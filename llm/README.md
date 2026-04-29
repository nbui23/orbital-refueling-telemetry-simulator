# RefuelGuard-LM

RefuelGuard-LM is an optional research extension for Orbital Refueling Simulator. It fine-tunes a small open-source language model to explain synthetic spacecraft refueling telemetry anomalies.

The LLM does not replace deterministic rules or the phase-aware ML anomaly detector. It consumes their outputs: scenario summaries, rule alerts, ML anomaly scores, and attribution signals.

## Layout

| Path | Purpose |
|---|---|
| `data/generate_instruction_data.py` | Builds deterministic JSONL instruction data from simulator outputs |
| `data/schema.md` | Dataset fields, task types, labels, and limitations |
| `training/train_lora.py` | Hugging Face + PEFT LoRA training script |
| `training/finetune_qwen_lora.ipynb` | Free-tier notebook workflow |
| `evals/rubric.py` | Deterministic scoring rubric |
| `evals/evaluate_base_vs_finetuned.py` | Base vs adapter evaluation script |
| `reports/` | Research report and model card templates |

## Generate Data

```bash
python3 llm/data/generate_instruction_data.py \
  --output-dir llm/data \
  --n-examples 300 \
  --seed 42
```

This writes `train.jsonl`, `val.jsonl`, and `test.jsonl`. Small `sample_*.jsonl` files are committed for inspection only.

## Train LoRA Adapter

```bash
pip install -r llm/training/requirements.txt
python3 llm/training/train_lora.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --train-file llm/data/train.jsonl \
  --val-file llm/data/val.jsonl
```

Default output:

```text
llm/training/outputs/refuelguard-lm-lora
```

For Colab or Kaggle, add `--use-4bit` if CUDA and bitsandbytes are available.

## Evaluate

```bash
python3 llm/evals/evaluate_base_vs_finetuned.py \
  --test-file llm/data/test.jsonl \
  --adapter-path llm/training/outputs/refuelguard-lm-lora \
  --limit 20
```

The evaluator uses a deterministic rubric. It checks whether the model names the likely anomaly family, mentions relevant signals, avoids unsupported claims, distinguishes rules from ML, and uses uncertainty language.

## Use In Streamlit

After training, the dashboard can use the adapter as an optional local explanation renderer:

```bash
pip install -r requirements.txt
pip install -r llm/training/requirements.txt
streamlit run app.py
```

Select `Fine-tuned LLM explanation` or `Side-by-side comparison` in the sidebar. The app builds a structured payload from simulator outputs and sends only grounded fields into the prompt. If the adapter at `llm/training/outputs/refuelguard-lm-lora` or the cached base model cannot load locally, the UI falls back to deterministic explanation text.

On Streamlit Community Cloud, local LLM loading is disabled by default to avoid memory/resource-limit failures. The dashboard remains usable with deterministic explanations. On a larger host with local model assets, opt in with:

```bash
REFUELGUARD_ENABLE_LOCAL_LLM=1 streamlit run app.py
```

## Safety Scope

All data are synthetic. Outputs are educational research explanations, not spacecraft diagnoses, not flight software, and not certified safety decisions.
