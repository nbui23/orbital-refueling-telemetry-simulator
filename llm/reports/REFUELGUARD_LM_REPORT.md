# RefuelGuard-LM Research Report

## Abstract

RefuelGuard-LM fine-tunes a small open-source instruction model to explain synthetic orbital refueling telemetry anomalies. The model is trained on summaries generated from Orbital Refueling Simulator, including rule alerts, phase-aware ML anomaly scores, and feature attribution.

## Motivation

Hybrid monitoring systems can detect hard-limit violations with rules and subtle multivariate patterns with ML. Operators still need concise explanations that preserve the difference between deterministic safety evidence and advisory anomaly scores.

## System Overview

Orbital Refueling Simulator generates synthetic mission telemetry across approach, docking, transfer, leak check, disconnect, and retreat phases. Existing modules provide deterministic rule alerts, phase-aware `IsolationForest` anomaly scores, and perturbation-based top contributors. RefuelGuard-LM consumes those outputs and generates explanations.

## Dataset Generation

Instruction rows are created by `llm/data/generate_instruction_data.py`. Each row includes a task type, instruction, telemetry summary input, and reference output. Task types cover anomaly explanation, classification, attribution, rule-vs-ML distinction, and uncertainty wording.

## Fine-Tuning Setup

Default base model: `Qwen/Qwen2.5-0.5B-Instruct`.

Method: LoRA adapter tuning with Hugging Face Transformers and PEFT. Optional QLoRA-style 4-bit loading is available on CUDA systems with bitsandbytes.

## Evaluation

Evaluation compares base model outputs against optional fine-tuned adapter outputs on the held-out synthetic test split. The latest run used 100 held-out examples generated with seed `20260429` and decoded with greedy generation capped at 96 new tokens. The deterministic rubric scores:

- correct anomaly family
- relevant signal mention
- no unsupported diagnosis
- rule vs ML distinction
- uncertainty language

## Results

| Model | N | Mean total score | Anomaly family | Signals | No unsupported diagnosis | Rules vs ML | Uncertainty |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct base | 100 | 3.69 / 5 | 0.86 | 0.87 | 1.00 | 0.00 | 0.96 |
| RefuelGuard-LM LoRA | 100 | 4.86 / 5 | 1.00 | 0.92 | 1.00 | 0.94 | 1.00 |

Task-level mean scores:

| Model | Explanation | Classification | Attribution | Rule vs ML | Safety uncertainty |
|---|---:|---:|---:|---:|---:|
| Base | 3.70 | 3.65 | 3.65 | 3.70 | 3.75 |
| RefuelGuard-LM LoRA | 5.00 | 5.00 | 4.90 | 4.60 | 4.80 |

## Error Analysis

The base model usually produced plausible spacecraft-monitoring prose, but it did not use the project-specific rule-vs-ML distinction required by the rubric. It failed that criterion on all 100 rows. It also missed the expected anomaly family on 14 rows and omitted relevant top-contributor signals on 13 rows.

The LoRA adapter preserved the intended safety framing and named the correct anomaly family on all 100 rows. Remaining failures were concentrated in `pump_degradation` prompts, where some outputs summarized the scenario without explicitly saying that ML scores are advisory or naming the top contributors. Two `sensor_dropout` attribution rows missed contributor wording. No fine-tuned outputs were penalized for unsupported diagnosis after the negation-aware rubric update.

## Limitations

- Synthetic data only.
- Template-generated labels.
- No real spacecraft validation.
- LLM output is explanatory only and must not drive safety actions.
- Rubric is deterministic and incomplete.

## Future Work

- Add multi-fault synthetic scenarios.
- Add human-written reference explanations.
- Compare Qwen, TinyLlama, and Gemma small adapters.
- Add calibration-focused evaluation for uncertainty wording.
