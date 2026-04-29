"""Optional explanation helpers for RefuelGuard-LM integration."""

from .deterministic_explainer import build_deterministic_explanation
from .modes import (
    DETERMINISTIC_MODE,
    LLM_MODE,
    SIDE_BY_SIDE_MODE,
    resolve_explanation_mode,
)
from .payload import ExplanationPayload, build_explanation_payload
from .prompt_builder import build_instruction_prompt

__all__ = [
    "DETERMINISTIC_MODE",
    "LLM_MODE",
    "SIDE_BY_SIDE_MODE",
    "ExplanationPayload",
    "build_explanation_payload",
    "build_deterministic_explanation",
    "build_instruction_prompt",
    "resolve_explanation_mode",
]
