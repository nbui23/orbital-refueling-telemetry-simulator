"""Deterministic fallback explanation renderer."""
from __future__ import annotations

from .payload import ExplanationPayload, scenario_description


def _humanize(value: str) -> str:
    return value.replace("_", " ")


def build_deterministic_explanation(payload: ExplanationPayload) -> str:
    """Render a concise explanation without using an LLM."""
    scenario = _humanize(payload.scenario)
    rule_text = (
        f"Deterministic rules report {payload.rule_alert_level}"
        + (f" via {', '.join(payload.rule_names)}." if payload.rule_names else ".")
    )
    contributors = ", ".join(_humanize(signal) for signal in payload.top_contributing_signals) or "none"
    changes = ", ".join(_humanize(change) for change in payload.signal_changes) or "no strong signal change"
    return (
        f"This pattern is consistent with {scenario} during {payload.phase.replace('_', ' ')}. "
        f"{rule_text} The ML anomaly score is {payload.ml_anomaly_score:.2f} and is advisory only. "
        f"Top contributing signals are {contributors}; observed changes are {changes}. "
        f"Simulator context: {scenario_description(payload.scenario)} "
        f"This is a synthetic, uncertain explanation and is not flight-certified."
    )
