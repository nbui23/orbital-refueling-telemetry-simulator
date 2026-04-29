"""Generate instruction data for RefuelGuard-LM.

Rows are derived from the Orbital Refueling Simulator's synthetic telemetry,
deterministic rule alerts, phase-aware ML scores, and attribution output. The
LLM is trained to explain those monitoring outputs, not to replace them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector import PhaseAwareDetector
from explainer import explain_window
from rules import RuleAlert, evaluate_rules
from simulator import ANOMALY_SCENARIOS, SCENARIO_DESCRIPTIONS, generate_telemetry

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UI helper
    tqdm = None


INSTRUCTION = "Explain the likely spacecraft refueling anomaly from the telemetry summary."
SAFETY_SUFFIX = (
    " This explanation is based on synthetic telemetry, rule alerts, and advisory ML "
    "scores; it is not flight-certified."
)

SCENARIO_SIGNALS: dict[str, list[str]] = {
    "nominal": ["no sustained abnormal signal"],
    "slow_leak": ["seal_pressure", "flow_rate", "line_pressure"],
    "partial_blockage": ["flow_rate", "line_pressure", "pump_current"],
    "pump_degradation": ["pump_current", "flow_rate", "line_temperature"],
    "arm_misalignment": ["end_effector_position_error", "interface_force", "interface_torque"],
    "sensor_drift": ["line_pressure", "donor_tank_pressure"],
    "sensor_dropout": ["line_pressure", "receiver_tank_pressure"],
    "stuck_at_pressure": ["line_pressure"],
    "bias_oscillation": ["line_pressure", "donor_tank_pressure"],
    "unstable_slosh": ["flow_rate", "line_pressure", "propellant_temperature"],
}

SCENARIO_FAMILY: dict[str, str] = {
    "nominal": "nominal behavior",
    "slow_leak": "seal or leak fault",
    "partial_blockage": "transfer-line blockage",
    "pump_degradation": "pump performance degradation",
    "arm_misalignment": "robotic arm alignment fault",
    "sensor_drift": "pressure sensor drift",
    "sensor_dropout": "sensor dropout",
    "stuck_at_pressure": "stuck pressure sensor",
    "bias_oscillation": "oscillating pressure sensor bias",
    "unstable_slosh": "propellant slosh instability",
}

FOCUS_PHASE: dict[str, str] = {
    "nominal": "main_transfer",
    "slow_leak": "leak_check",
    "partial_blockage": "main_transfer",
    "pump_degradation": "main_transfer",
    "arm_misalignment": "docking",
    "sensor_drift": "main_transfer",
    "sensor_dropout": "main_transfer",
    "stuck_at_pressure": "main_transfer",
    "bias_oscillation": "main_transfer",
    "unstable_slosh": "main_transfer",
}


def _highest_severity(alerts: Iterable[RuleAlert]) -> str:
    severities = {alert.severity.upper() for alert in alerts}
    if "CRITICAL" in severities:
        return "critical"
    if "WARNING" in severities:
        return "warning"
    return "none"


def _signal_trends(df: pd.DataFrame, nominal_df: pd.DataFrame, scenario: str, phase: str) -> list[str]:
    trends: list[str] = []
    for signal in SCENARIO_SIGNALS[scenario]:
        if signal not in df.columns:
            trends.append(signal)
            continue
        observed_series = df.loc[df["phase"] == phase, signal]
        baseline_series = nominal_df.loc[nominal_df["phase"] == phase, signal]
        observed = float(observed_series.mean())
        baseline = float(baseline_series.mean())
        observed_std = float(observed_series.std())
        baseline_std = float(baseline_series.std())
        delta = observed - baseline
        threshold = max(abs(baseline) * 0.08, 0.02)
        if scenario in {"bias_oscillation", "unstable_slosh"} and observed_std > baseline_std * 1.8:
            trends.append(f"{signal} oscillating")
        elif scenario == "stuck_at_pressure" and observed_std < max(baseline_std * 0.4, 0.02):
            trends.append(f"{signal} held nearly constant")
        elif delta > threshold:
            trends.append(f"{signal} increased")
        elif delta < -threshold:
            trends.append(f"{signal} decreased")
        else:
            trends.append(f"{signal} near nominal")
    return trends


def _top_contributors(
    detector: PhaseAwareDetector,
    df: pd.DataFrame,
    scores: np.ndarray,
    fallback: list[str],
) -> list[str]:
    threshold = max(0.35, float(np.percentile(scores, 92)))
    contribs = explain_window(detector, df, scores, threshold=threshold, n_top=3, max_samples=16)
    top = [name for name, value in contribs.items() if value > 0]
    expected = [name for name in top if name in fallback]
    combined = expected + [name for name in fallback if name not in expected]
    return combined[:3]


def _rule_names(alerts: list[RuleAlert]) -> str:
    names = sorted({alert.rule_name for alert in alerts})
    return ", ".join(names[:4]) if names else "none"


def _format_input(
    *,
    phase: str,
    scenario: str,
    trends: list[str],
    severity: str,
    ml_score: float,
    top_contributors: list[str],
    rule_names: str,
) -> str:
    return "\n".join(
        [
            f"Phase: {phase}",
            f"Scenario: {scenario}",
            f"Signals: {', '.join(trends)}",
            f"Rule alerts: {severity}",
            f"Rule IDs: {rule_names}",
            f"ML anomaly score: {ml_score:.2f}",
            f"Top contributors: {', '.join(top_contributors)}",
        ]
    )


def _outputs_for_task(
    task_type: str,
    scenario: str,
    trends: list[str],
    severity: str,
    ml_score: float,
    top_contributors: list[str],
) -> str:
    family = SCENARIO_FAMILY[scenario]
    description = SCENARIO_DESCRIPTIONS[scenario]
    signals = ", ".join(top_contributors)
    trend_text = ", ".join(trends)

    if task_type == "classification":
        return (
            f"The most likely anomaly family is {family}. The simulator label is {scenario}; "
            f"the supporting pattern is {trend_text}. ML score {ml_score:.2f} is advisory, "
            f"while rule severity is {severity}."
            + SAFETY_SUFFIX
        )
    if task_type == "attribution":
        return (
            f"The main attribution signals are {signals}. They matter because {description.lower()} "
            f"The contributors explain why the phase-aware ML detector treated this window as unusual; "
            f"they are not causal proof by themselves."
            + SAFETY_SUFFIX
        )
    if task_type == "rule_vs_ml":
        return (
            f"Rules provide hard-limit evidence and currently report {severity}. The ML score "
            f"of {ml_score:.2f} is advisory evidence of multivariate behavior. Use rules for "
            f"explicit safety thresholds, then use ML and attribution to explain pattern context."
            + SAFETY_SUFFIX
        )
    if task_type == "safety_uncertainty":
        return (
            f"This should be described as a likely {family}, not a certified diagnosis. The wording "
            f"should mention uncertainty because the data are synthetic and the ML score is advisory. "
            f"Recommended phrasing: the pattern is consistent with {scenario} given {trend_text}."
            + SAFETY_SUFFIX
        )
    return (
        f"This pattern is most consistent with {scenario} during the refueling sequence. "
        f"{trend_text.capitalize()} match the simulator behavior: {description.lower()} "
        f"The rule alert level is {severity}, and the ML score of {ml_score:.2f} adds advisory "
        f"evidence of abnormal multivariate behavior."
        + SAFETY_SUFFIX
    )


def build_examples(n_examples: int, seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    print("Training nominal phase-aware detector for dataset generation...")
    detector = PhaseAwareDetector(contamination=0.02, n_estimators=80, random_state=seed).fit(
        generate_telemetry("nominal", seed=seed)
    )
    task_types = [
        "explanation",
        "classification",
        "attribution",
        "rule_vs_ml",
        "safety_uncertainty",
    ]

    rows: list[dict[str, str]] = []
    iterator = range(n_examples)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Generating instruction rows", unit="row")
    for i in iterator:
        scenario = str(rng.choice(ANOMALY_SCENARIOS))
        sim_seed = int(rng.integers(0, 1_000_000))
        phase = FOCUS_PHASE[scenario]
        df = generate_telemetry(scenario, seed=sim_seed)
        nominal_df = generate_telemetry("nominal", seed=sim_seed)
        scores = detector.score(df)
        focus_mask = df["phase"] == phase
        focus_scores = scores[focus_mask.to_numpy()]
        ml_score = float(np.percentile(focus_scores, 95)) if len(focus_scores) else float(scores.max())
        alerts = evaluate_rules(df)
        focus_alerts = [a for a in alerts if a.phase == phase] or alerts
        severity = _highest_severity(focus_alerts)
        trends = _signal_trends(df, nominal_df, scenario, phase)
        top = _top_contributors(detector, df, scores, SCENARIO_SIGNALS[scenario])
        task_type = task_types[i % len(task_types)]

        rows.append(
            {
                "task_type": task_type,
                "instruction": INSTRUCTION,
                "input": _format_input(
                    phase=phase,
                    scenario=scenario,
                    trends=trends,
                    severity=severity,
                    ml_score=ml_score,
                    top_contributors=top,
                    rule_names=_rule_names(focus_alerts),
                ),
                "output": _outputs_for_task(task_type, scenario, trends, severity, ml_score, top),
            }
        )
    return rows


def split_examples(
    examples: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected --train-ratio > 0, --val-ratio >= 0, and train+val < 1.")
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return examples[:train_end], examples[train_end:val_end], examples[val_end:]


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RefuelGuard-LM instruction data.")
    parser.add_argument("--output-dir", type=Path, default=Path("llm/data"))
    parser.add_argument("--n-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_examples < 10:
        raise ValueError("--n-examples should be at least 10 so all task types are represented.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = build_examples(args.n_examples, args.seed)
    train, val, test = split_examples(examples, args.train_ratio, args.val_ratio)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)
    write_jsonl(args.output_dir / "test.jsonl", test)
    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test rows to {args.output_dir}")


if __name__ == "__main__":
    main()
