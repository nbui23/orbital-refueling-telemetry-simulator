"""Local optional LLM explanation adapter for RefuelGuard-LM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .payload import ExplanationPayload
from .prompt_builder import build_instruction_prompt


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER_PATH = Path("llm/training/outputs/refuelguard-lm-lora")


@dataclass(frozen=True)
class LLMExplanation:
    text: str | None
    available: bool
    error: str | None = None


class LocalLLMExplainer:
    """Loads a local/cached base model plus optional LoRA adapter.

    No paid APIs are used. If loading fails, callers receive a structured error
    and can fall back to deterministic explanation text.
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str | Path | None = DEFAULT_ADAPTER_PATH,
        *,
        local_files_only: bool = True,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        do_sample: bool = False,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_error: str | None = None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def is_available(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True
        self._load()
        return self._model is not None and self._tokenizer is not None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self._load_error is not None:
            return
        if self.adapter_path is not None and not self.adapter_path.exists():
            self._load_error = f"Fine-tuned adapter not found: {self.adapter_path}"
            return

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            if self.adapter_path is not None:
                model = PeftModel.from_pretrained(
                    model,
                    str(self.adapter_path),
                    local_files_only=self.local_files_only,
                )
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
        except Exception as exc:  # pragma: no cover - exact HF errors vary by host
            self._load_error = f"Unable to load local LLM explainer: {exc}"

    def explain(self, payload: ExplanationPayload) -> LLMExplanation:
        if not self.is_available:
            return LLMExplanation(text=None, available=False, error=self._load_error)

        try:
            import torch

            prompt = build_instruction_prompt(payload)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if self.do_sample:
                generation_kwargs["temperature"] = self.temperature
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **generation_kwargs)
            new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            return LLMExplanation(text=_ensure_safety_wording(text), available=True)
        except Exception as exc:  # pragma: no cover - exact HF errors vary by host
            return LLMExplanation(text=None, available=False, error=f"LLM generation failed: {exc}")


def _ensure_safety_wording(text: str) -> str:
    lower = text.lower()
    additions: list[str] = []
    if "synthetic" not in lower:
        additions.append("based on synthetic telemetry")
    if "not flight-certified" not in lower and "not flight certified" not in lower:
        additions.append("not flight-certified")
    if "likely" not in lower and "consistent with" not in lower and "suggest" not in lower:
        additions.append("uncertain")
    if not additions:
        return text
    return f"{text} This explanation is {', '.join(additions)}."
