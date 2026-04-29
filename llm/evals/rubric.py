"""Deterministic rubric for RefuelGuard-LM generated explanations."""
from __future__ import annotations

import re
from dataclasses import dataclass


ANOMALY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "nominal": ("nominal", "no anomaly", "normal"),
    "slow_leak": ("leak", "seal"),
    "partial_blockage": ("blockage", "blocked", "restriction", "obstruction"),
    "pump_degradation": ("pump", "degradation", "overcurrent"),
    "arm_misalignment": ("misalignment", "alignment", "end effector", "interface force"),
    "sensor_drift": ("drift", "bias"),
    "sensor_dropout": ("dropout", "zero", "data gap"),
    "stuck_at_pressure": ("stuck", "frozen", "freezes", "constant pressure"),
    "bias_oscillation": ("oscillation", "oscillating", "bias"),
    "unstable_slosh": ("slosh", "oscillation", "unstable flow"),
}

UNSUPPORTED_STRONG_CLAIMS = (
    "definitely",
    "certainly",
    "guaranteed",
    "confirmed",
    "is flight-certified",
    "flight certified for",
    "must abort because the ml",
    "ml proves",
)

NEGATION_TERMS = ("not", "no", "never", "without", "cannot", "can't", "isn't", "not a")
UNCERTAINTY_TERMS = ("likely", "consistent with", "suggests", "advisory", "synthetic", "not flight-certified")


@dataclass(frozen=True)
class RubricScore:
    total: int
    correct_anomaly_family: int
    relevant_signals_mentioned: int
    no_unsupported_diagnosis: int
    rules_vs_ml_distinction: int
    uncertainty_language: int


def _scenario_from_input(prompt_input: str) -> str:
    match = re.search(r"^Scenario:\s*([a-z0-9_]+)", prompt_input, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _contributors_from_input(prompt_input: str) -> list[str]:
    match = re.search(r"^Top contributors:\s*(.+)$", prompt_input, flags=re.MULTILINE)
    if not match:
        return []
    return [item.strip().lower() for item in match.group(1).split(",") if item.strip()]


def _has_unsupported_claim(text: str) -> bool:
    """Return True for strong unsupported claims, but ignore local negations.

    Examples that should not be penalized:
    - "not guaranteed"
    - "not confirmed"
    - "cannot be confirmed"
    - "no guaranteed diagnosis"
    """
    normalized = re.sub(r"[^a-z0-9'-]+", " ", text.lower())
    tokens = normalized.split()
    claim_roots = tuple(claim.split()[0] for claim in UNSUPPORTED_STRONG_CLAIMS if " " not in claim)

    for phrase in (claim for claim in UNSUPPORTED_STRONG_CLAIMS if " " in claim):
        if phrase in normalized:
            return True

    for idx, token in enumerate(tokens):
        if not any(token.startswith(root) for root in claim_roots):
            continue
        window = tokens[max(0, idx - 4) : idx]
        if any(term in window for term in NEGATION_TERMS):
            continue
        return True
    return False


def score_output(prompt_input: str, output: str) -> RubricScore:
    text = output.lower()
    scenario = _scenario_from_input(prompt_input)
    expected_terms = ANOMALY_KEYWORDS.get(scenario, ())
    contributors = _contributors_from_input(prompt_input)

    correct_family = int(any(term in text for term in expected_terms))
    relevant_signals = int(sum(1 for signal in contributors if signal.replace("_", " ") in text or signal in text) >= 1)
    no_unsupported = int(not _has_unsupported_claim(text))
    rules_vs_ml = int(("rule" in text and "ml" in text) and ("advisory" in text or "threshold" in text))
    uncertainty = int(any(term in text for term in UNCERTAINTY_TERMS))
    total = correct_family + relevant_signals + no_unsupported + rules_vs_ml + uncertainty
    return RubricScore(total, correct_family, relevant_signals, no_unsupported, rules_vs_ml, uncertainty)
