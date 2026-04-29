"""Compare base model and optional RefuelGuard-LM LoRA adapter on test JSONL."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rubric import score_output

ROOT = Path(__file__).resolve().parents[2]

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UI helper
    tqdm = None


def format_prompt(instruction: str, user_input: str) -> str:
    return (
        "<|system|>\n"
        "You explain synthetic spacecraft refueling telemetry anomalies. "
        "Do not replace deterministic rules or ML detectors.\n"
        "<|user|>\n"
        f"{instruction}\n\n{user_input}\n"
        "<|assistant|>\n"
    )


def load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_model(base_model: str, adapter_path: Path | None = None):
    label = f"{base_model}"
    if adapter_path:
        label += f" + adapter {adapter_path}"
    print(f"Loading model: {label}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return tokenizer, model


def generate_batch(tokenizer, model, prompts: list[str], max_new_tokens: int) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
        for ids in output_ids
    ]


def evaluate_label(
    label: str,
    tokenizer,
    model,
    rows: list[dict[str, str]],
    max_new_tokens: int,
    batch_size: int,
) -> list[dict]:
    results: list[dict] = []
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    iterator = enumerate(batches)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(batches), desc=f"Evaluating {label}", unit="batch")
    for batch_idx, batch in iterator:
        prompts = [format_prompt(row["instruction"], row["input"]) for row in batch]
        predictions = generate_batch(tokenizer, model, prompts, max_new_tokens)
        for offset, (row, prediction) in enumerate(zip(batch, predictions)):
            score = score_output(row["input"], prediction)
            results.append(
                {
                    "row_id": batch_idx * batch_size + offset,
                    "model": label,
                    "task_type": row.get("task_type", ""),
                    "input": row["input"],
                    "reference": row["output"],
                    "prediction": prediction,
                    **score.__dict__,
                }
        )
    return results


def print_summary(results: list[dict]) -> None:
    labels = sorted({row["model"] for row in results})
    print("\nmodel, n, mean_total")
    for label in labels:
        subset = [row for row in results if row["model"] == label]
        mean_total = sum(row["total"] for row in subset) / max(len(subset), 1)
        print(f"{label}, {len(subset)}, {mean_total:.2f}")
    print("\ncriterion means")
    criteria = [
        "correct_anomaly_family",
        "relevant_signals_mentioned",
        "no_unsupported_diagnosis",
        "rules_vs_ml_distinction",
        "uncertainty_language",
    ]
    for label in labels:
        subset = [row for row in results if row["model"] == label]
        parts = [f"{criterion}={sum(row[criterion] for row in subset) / max(len(subset), 1):.2f}" for criterion in criteria]
        print(f"{label}: " + ", ".join(parts))
    print("\ntask means")
    for label in labels:
        task_types = sorted({row["task_type"] for row in results if row["model"] == label})
        for task_type in task_types:
            subset = [row for row in results if row["model"] == label and row["task_type"] == task_type]
            mean_total = sum(row["total"] for row in subset) / max(len(subset), 1)
            print(f"{label}, {task_type}, {len(subset)}, {mean_total:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RefuelGuard-LM outputs with deterministic rubric.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--test-file", type=Path, default=ROOT / "llm" / "data" / "test.jsonl")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "llm" / "evals" / "results.csv")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum test rows to evaluate. Omit or pass 0 to evaluate the full test file.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit == 0:
        args.limit = None
    rows = load_rows(args.test_file, args.limit)
    if not rows:
        raise ValueError(f"No rows loaded from {args.test_file}")
    print(f"Loaded {len(rows)} test rows from {args.test_file}")
    all_results: list[dict] = []

    base_tokenizer, base = load_model(args.base_model)
    all_results.extend(evaluate_label("base", base_tokenizer, base, rows, args.max_new_tokens, args.batch_size))
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.adapter_path:
        tuned_tokenizer, tuned = load_model(args.base_model, args.adapter_path)
        all_results.extend(evaluate_label("finetuned", tuned_tokenizer, tuned, rows, args.max_new_tokens, args.batch_size))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print_summary(all_results)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
