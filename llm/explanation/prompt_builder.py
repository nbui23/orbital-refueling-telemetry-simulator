"""Prompt builder for RefuelGuard-LM explanations."""
from __future__ import annotations

from .payload import ExplanationPayload


INSTRUCTION = (
    "Explain the likely spacecraft refueling anomaly from the structured telemetry summary. "
    "Use only the provided fields. Distinguish deterministic rule alerts from advisory ML "
    "anomaly scores. Include uncertainty and avoid unsupported claims. State that the "
    "explanation is based on synthetic telemetry and is not flight-certified. Keep the "
    "explanation concise."
)


def _join_or_none(items: tuple[str, ...]) -> str:
    return ", ".join(items) if items else "none"


def build_prompt_input(payload: ExplanationPayload) -> str:
    value_text = ", ".join(
        f"{name}={value}" for name, value in payload.current_telemetry_values.items()
    ) or "none"
    return "\n".join(
        [
            f"Phase: {payload.phase}",
            f"Scenario: {payload.scenario}",
            f"Rule alerts: {payload.rule_alert_level}",
            f"Triggered rules: {_join_or_none(payload.rule_names)}",
            f"ML anomaly score: {payload.ml_anomaly_score:.2f}",
            f"Top contributors: {_join_or_none(payload.top_contributing_signals)}",
            f"Signal changes: {_join_or_none(payload.signal_changes)}",
            f"Current telemetry values: {value_text}",
            f"Safety note: {payload.safety_note}",
        ]
    )


def build_instruction_prompt(payload: ExplanationPayload) -> str:
    return f"Instruction:\n{INSTRUCTION}\n\nInput:\n{build_prompt_input(payload)}\n\nResponse:\n"
