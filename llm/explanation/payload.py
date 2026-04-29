"""Structured explanation payloads for deterministic and LLM explainers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from simulator import SCENARIO_DESCRIPTIONS, generate_telemetry


SAFETY_NOTE = (
    "Synthetic telemetry only. Deterministic rules and ML scores are monitoring outputs, "
    "not flight-certified spacecraft diagnoses."
)


@dataclass(frozen=True)
class ExplanationPayload:
    """Grounded simulator outputs allowed into an explanation model."""

    phase: str
    scenario: str
    rule_alert_level: str
    rule_names: tuple[str, ...]
    ml_anomaly_score: float
    top_contributing_signals: tuple[str, ...]
    signal_changes: tuple[str, ...]
    current_telemetry_values: dict[str, float]
    safety_note: str = SAFETY_NOTE

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "scenario": self.scenario,
            "rule_alert_level": self.rule_alert_level,
            "rule_names": list(self.rule_names),
            "ml_anomaly_score": self.ml_anomaly_score,
            "top_contributing_signals": list(self.top_contributing_signals),
            "signal_changes": list(self.signal_changes),
            "current_telemetry_values": self.current_telemetry_values,
            "safety_note": self.safety_note,
        }


def _highest_rule_level(grouped_alerts: pd.DataFrame) -> str:
    if grouped_alerts is None or grouped_alerts.empty or "severity" not in grouped_alerts:
        return "none"
    severities = {str(value).upper() for value in grouped_alerts["severity"].dropna()}
    if "CRITICAL" in severities:
        return "critical"
    if "WARNING" in severities:
        return "warning"
    return "none"


def _rule_names(grouped_alerts: pd.DataFrame) -> tuple[str, ...]:
    if grouped_alerts is None or grouped_alerts.empty:
        return ()
    source_col = "rule_id" if "rule_id" in grouped_alerts.columns else "rule"
    if source_col not in grouped_alerts.columns:
        return ()
    return tuple(sorted(str(value) for value in grouped_alerts[source_col].dropna().unique()))


def _focus_index(df: pd.DataFrame, scores: np.ndarray) -> int:
    if len(scores) == 0:
        return 0
    return int(np.argmax(scores))


def _signal_changes(
    scenario: str,
    phase: str,
    df: pd.DataFrame,
    signals: tuple[str, ...],
    nominal_seed: int,
) -> tuple[str, ...]:
    if not signals:
        return ()
    nominal = generate_telemetry("nominal", seed=nominal_seed)
    changes: list[str] = []
    for signal in signals:
        if signal not in df.columns or signal not in nominal.columns:
            continue
        observed = df.loc[df["phase"] == phase, signal]
        baseline = nominal.loc[nominal["phase"] == phase, signal]
        if observed.empty or baseline.empty:
            continue
        observed_mean = float(observed.mean())
        baseline_mean = float(baseline.mean())
        observed_std = float(observed.std())
        baseline_std = float(baseline.std())
        threshold = max(abs(baseline_mean) * 0.08, 0.02)

        if scenario in {"bias_oscillation", "unstable_slosh"} and observed_std > baseline_std * 1.8:
            changes.append(f"{signal} oscillating")
        elif scenario == "stuck_at_pressure" and observed_std < max(baseline_std * 0.4, 0.02):
            changes.append(f"{signal} held nearly constant")
        elif observed_mean - baseline_mean > threshold:
            changes.append(f"{signal} increased")
        elif observed_mean - baseline_mean < -threshold:
            changes.append(f"{signal} decreased")
        else:
            changes.append(f"{signal} near nominal")
    return tuple(changes)


def _current_values(df: pd.DataFrame, idx: int, signals: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, float] = {}
    if df.empty:
        return values
    row = df.iloc[idx]
    for signal in signals:
        if signal in row and pd.notna(row[signal]):
            values[signal] = round(float(row[signal]), 4)
    return values


def build_explanation_payload(
    *,
    scenario: str,
    df: pd.DataFrame,
    scores: np.ndarray,
    grouped_alerts: pd.DataFrame,
    contributions: dict[str, float],
    nominal_seed: int = 42,
) -> ExplanationPayload:
    """Build a bounded, grounded payload for explanation rendering."""
    idx = _focus_index(df, scores)
    phase = str(df.iloc[idx]["phase"]) if not df.empty else "unknown"
    top_signals = tuple(contributions.keys())[:6]
    if not top_signals and scenario == "nominal":
        top_signals = ("no sustained abnormal signal",)
    elif not top_signals:
        top_signals = tuple()
    signal_changes = _signal_changes(scenario, phase, df, top_signals, nominal_seed)
    ml_score = float(np.max(scores)) if len(scores) else 0.0

    return ExplanationPayload(
        phase=phase,
        scenario=scenario,
        rule_alert_level=_highest_rule_level(grouped_alerts),
        rule_names=_rule_names(grouped_alerts),
        ml_anomaly_score=round(ml_score, 4),
        top_contributing_signals=top_signals,
        signal_changes=signal_changes,
        current_telemetry_values=_current_values(df, idx, top_signals),
        safety_note=SAFETY_NOTE,
    )


def scenario_description(scenario: str) -> str:
    return SCENARIO_DESCRIPTIONS.get(scenario, "No simulator description available.")
