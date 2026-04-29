"""Tests for optional RefuelGuard-LM explanation integration."""
from __future__ import annotations

from llm.explanation import (
    LLM_MODE,
    SIDE_BY_SIDE_MODE,
    ExplanationPayload,
    build_instruction_prompt,
    resolve_explanation_mode,
)
from llm.explanation.llm_explainer import LLMExplanation, LocalLLMExplainer


def _payload() -> ExplanationPayload:
    return ExplanationPayload(
        phase="main_transfer",
        scenario="partial_blockage",
        rule_alert_level="warning",
        rule_names=("LINE_PRESSURE_WARNING",),
        ml_anomaly_score=0.82,
        top_contributing_signals=("line_pressure", "flow_rate"),
        signal_changes=("line_pressure increased", "flow_rate decreased"),
        current_telemetry_values={"line_pressure": 185.2, "flow_rate": 0.21},
    )


def test_prompt_builder_includes_required_fields() -> None:
    prompt = build_instruction_prompt(_payload())

    assert "Phase: main_transfer" in prompt
    assert "Scenario: partial_blockage" in prompt
    assert "Rule alerts: warning" in prompt
    assert "Triggered rules: LINE_PRESSURE_WARNING" in prompt
    assert "ML anomaly score: 0.82" in prompt
    assert "Top contributors: line_pressure, flow_rate" in prompt
    assert "Signal changes: line_pressure increased, flow_rate decreased" in prompt


def test_prompt_builder_includes_safety_instructions() -> None:
    prompt = build_instruction_prompt(_payload()).lower()

    assert "use only the provided fields" in prompt
    assert "deterministic rule alerts" in prompt
    assert "advisory ml" in prompt
    assert "uncertainty" in prompt
    assert "not flight-certified" in prompt


def test_llm_explainer_falls_back_when_adapter_missing() -> None:
    explainer = LocalLLMExplainer(
        base_model="/missing/base/model",
        adapter_path="/missing/adapter/path",
        local_files_only=True,
    )

    result = explainer.explain(_payload())

    assert not result.available
    assert result.text is None
    assert result.error is not None
    assert "adapter not found" in result.error.lower()


class _UnavailableExplainer:
    def explain(self, payload: ExplanationPayload) -> LLMExplanation:
        return LLMExplanation(text=None, available=False, error="mock load failure")


class _AvailableExplainer:
    def explain(self, payload: ExplanationPayload) -> LLMExplanation:
        return LLMExplanation(text="Likely partial blockage. Synthetic and not flight-certified.", available=True)


def test_side_by_side_mode_falls_back_when_llm_unavailable() -> None:
    result = resolve_explanation_mode(
        mode=SIDE_BY_SIDE_MODE,
        payload=_payload(),
        deterministic_text="deterministic fallback",
        adapter_exists=True,
        llm_explainer=_UnavailableExplainer(),
    )

    assert result.llm_text is None
    assert result.deterministic_text == "deterministic fallback"
    assert not result.show_side_by_side
    assert result.warning is not None


def test_side_by_side_mode_returns_both_when_llm_available() -> None:
    result = resolve_explanation_mode(
        mode=SIDE_BY_SIDE_MODE,
        payload=_payload(),
        deterministic_text="deterministic fallback",
        adapter_exists=True,
        llm_explainer=_AvailableExplainer(),
    )

    assert result.deterministic_text == "deterministic fallback"
    assert result.llm_text is not None
    assert result.show_side_by_side
    assert result.warning is None


def test_llm_mode_falls_back_when_adapter_missing() -> None:
    result = resolve_explanation_mode(
        mode=LLM_MODE,
        payload=_payload(),
        deterministic_text="deterministic fallback",
        adapter_exists=False,
        llm_explainer=None,
    )

    assert result.llm_text is None
    assert result.deterministic_text == "deterministic fallback"
    assert "adapter not found" in (result.warning or "").lower()
