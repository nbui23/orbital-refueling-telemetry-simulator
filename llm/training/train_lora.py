"""LoRA fine-tuning entry point for RefuelGuard-LM.

Designed for local CPU smoke tests and free GPU notebooks. For larger runs,
use a Colab/Kaggle T4 and keep the default Qwen2.5-0.5B model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "llm" / "training" / "outputs" / "refuelguard-lm-lora"


def format_prompt(instruction: str, user_input: str) -> str:
    return (
        "<|system|>\n"
        "You explain synthetic spacecraft refueling telemetry anomalies. "
        "Do not replace deterministic rules or ML detectors.\n"
        "<|user|>\n"
        f"{instruction}\n\n{user_input}\n"
        "<|assistant|>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small open LLM with LoRA.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=ROOT / "llm" / "data" / "train.jsonl")
    parser.add_argument("--val-file", type=Path, default=ROOT / "llm" / "data" / "val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--use-4bit", action="store_true", help="Requires bitsandbytes and CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_file.exists():
        sys.exit(f"Missing train file: {args.train_file}. Run llm/data/generate_instruction_data.py first.")

    data_files = {"train": str(args.train_file)}
    if args.val_file.exists():
        data_files["validation"] = str(args.val_file)
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(example: dict) -> dict:
        prompt = format_prompt(example["instruction"], example["input"])
        answer = example["output"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[: args.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: args.max_length]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if "validation" in tokenized else "no",
        report_to="none",
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
