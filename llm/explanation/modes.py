"""Pure explanation-mode resolution helpers for UI and tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .llm_explainer import LLMExplanation
from .payload import ExplanationPayload


DETERMINISTIC_MODE = "Deterministic explanation"
LLM_MODE = "Fine-tuned LLM explanation"
SIDE_BY_SIDE_MODE = "Side-by-side comparison"


class ExplainerLike(Protocol):
    def explain(self, payload: ExplanationPayload) -> LLMExplanation:
        ...


@dataclass(frozen=True)
class ExplanationModeResult:
    deterministic_text: str
    llm_text: str | None
    warning: str | None
    show_side_by_side: bool


def resolve_explanation_mode(
    *,
    mode: str,
    payload: ExplanationPayload,
    deterministic_text: str,
    adapter_exists: bool,
    llm_explainer: ExplainerLike | None,
) -> ExplanationModeResult:
    if mode == DETERMINISTIC_MODE:
        return ExplanationModeResult(deterministic_text, None, None, False)

    if not adapter_exists:
        return ExplanationModeResult(
            deterministic_text,
            None,
            "Fine-tuned adapter not found. Showing deterministic explanation instead.",
            False,
        )

    if llm_explainer is None:
        return ExplanationModeResult(
            deterministic_text,
            None,
            "Fine-tuned LLM explainer unavailable. Showing deterministic explanation instead.",
            False,
        )

    llm_result = llm_explainer.explain(payload)
    if not llm_result.available or not llm_result.text:
        error = f" {llm_result.error}" if llm_result.error else ""
        return ExplanationModeResult(
            deterministic_text,
            None,
            f"Fine-tuned LLM explainer unavailable. Showing deterministic explanation instead.{error}",
            False,
        )

    return ExplanationModeResult(
        deterministic_text,
        llm_result.text,
        None,
        mode == SIDE_BY_SIDE_MODE,
    )
