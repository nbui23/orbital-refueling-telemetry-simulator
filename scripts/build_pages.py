"""Build static GitHub Pages assets for the telemetry simulator."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
OUTPUT = PUBLIC / "site-data.js"
sys.path.insert(0, str(ROOT))

from detector import PhaseAwareDetector
from rules import evaluate_rules, group_alerts
from simulator import ANOMALY_SCENARIOS, SCENARIO_DESCRIPTIONS, generate_telemetry

SIGNALS = [
    "flow_rate",
    "line_pressure",
    "seal_pressure",
    "pump_current",
    "end_effector_position_error",
    "interface_force",
    "attitude_error",
    "reaction_wheel_speed",
    "bus_voltage",
]


def _clean_float(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


def _severity(alerts: list[dict[str, Any]]) -> str:
    severities = {str(alert.get("severity", "")) for alert in alerts}
    if "CRITICAL" in severities:
        return "CRITICAL"
    if "WARNING" in severities:
        return "WARNING"
    return "NONE"


def _scenario_payload(detector: PhaseAwareDetector, scenario: str) -> dict[str, Any]:
    df = generate_telemetry(scenario=scenario)
    scores = detector.score(df)
    df = df.copy()
    df["ml_score"] = scores * 100.0

    grouped = group_alerts(evaluate_rules(df))
    alerts = grouped.to_dict(orient="records")

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "time": _clean_float(row["time"], 1),
                "phase": str(row["phase"]),
                "ml_score": _clean_float(row["ml_score"], 2),
                **{signal: _clean_float(row[signal], 4) for signal in SIGNALS},
            }
        )

    top_score_idx = int(df["ml_score"].idxmax())
    top_score_row = df.loc[top_score_idx]

    return {
        "description": SCENARIO_DESCRIPTIONS[scenario],
        "summary": {
            "max_ml_score": _clean_float(df["ml_score"].max(), 1),
            "mean_ml_score": _clean_float(df["ml_score"].mean(), 1),
            "alert_count": len(alerts),
            "highest_severity": _severity(alerts),
            "top_phase": str(top_score_row["phase"]),
            "top_time": _clean_float(top_score_row["time"], 1),
        },
        "alerts": alerts,
        "rows": rows,
    }


def build() -> None:
    nominal = generate_telemetry(scenario="nominal", seed=42)
    detector = PhaseAwareDetector().fit(nominal)
    payload = {
        "generated_by": "scripts/build_pages.py",
        "scenarios": {
            scenario: _scenario_payload(detector, scenario)
            for scenario in ANOMALY_SCENARIOS
        },
        "signals": SIGNALS,
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "window.TELEMETRY_SITE_DATA = "
        + json.dumps(payload, separators=(",", ":"), sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
