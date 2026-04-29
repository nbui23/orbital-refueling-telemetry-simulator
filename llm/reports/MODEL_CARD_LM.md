# RefuelGuard-LM Model Card

## Model Name

RefuelGuard-LM LoRA adapter.

## Base Model

Default: `Qwen/Qwen2.5-0.5B-Instruct`.

Other small open-source bases may be tested, including Qwen2.5-1.5B-Instruct, TinyLlama, and small Gemma variants.

## Intended Use

Generate concise explanations of synthetic orbital refueling anomaly summaries produced by Orbital Refueling Simulator. Inputs should include mission phase, simulator scenario, signal trends, deterministic rule alerts, advisory ML score, and top attribution signals.

## Out-of-Scope Use

- Real spacecraft diagnosis.
- Flight software.
- Autonomous abort or control decisions.
- Replacing deterministic rules or anomaly detectors.
- Medical, legal, financial, or other unrelated decision support.

## Training Data

Synthetic instruction rows generated from Orbital Refueling Simulator using:

- `generate_telemetry`
- `evaluate_rules`
- `PhaseAwareDetector`
- `explain_window`

## Synthetic Limitations

Telemetry is procedurally generated with simplified physics. Rule thresholds are illustrative. ML scores are advisory and not real-world calibrated. Reference explanations are template-generated.

## Safety Notes

Model outputs should include uncertainty language and should distinguish deterministic rule alerts from advisory ML anomaly scores. Any deployment should keep rules and anomaly detection as separate upstream systems.

## Evaluation Summary

Use `llm/evals/evaluate_base_vs_finetuned.py` to score held-out examples. Report mean total score plus failure categories before publishing results.
