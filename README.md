# Orbital Refueling Telemetry Simulator

Autonomous orbital refueling anomaly detection: a synthetic Python prototype demonstrating hybrid deterministic and machine-learning monitoring for spacecraft autonomy.

Orbital Refueling Telemetry Simulator is an educational portfolio project. It is not flight software, not an autonomous abort system, not a real spacecraft diagnosis tool, and not based on real spacecraft telemetry. All telemetry is procedurally generated with simplified physics.

## Why This Project Exists

Autonomous propellant transfer between spacecraft is a high-stakes operation with several subsystems moving at once: robotic alignment, docking contact loads, seal pressure, transfer pressure, pump behavior, fluid flow, attitude stability, thermal health, and electrical health.

Some faults are obvious because one signal crosses a hard engineering limit. Others are subtle: a pressure sensor can drift, a pump can degrade gradually, or flow and pressure can move in a correlated pattern that is individually borderline but collectively unusual.

Orbital Refueling Telemetry Simulator demonstrates a monitoring architecture for that split:

- deterministic rules for explicit hard-limit checks
- a phase-aware ML anomaly detector for advisory early warning
- lightweight feature attribution to explain which signals influenced the ML score
- a Streamlit dashboard for scenario exploration and validation

## Demo Highlights

- Full 410-second synthetic orbital refueling mission sampled at 2 Hz
- Nine mission phases, from approach through retreat
- Ten selectable scenarios: nominal plus nine anomaly cases
- Deterministic rule alerts with severity and recommended actions
- One `IsolationForest` model per mission phase
- ML uncertainty bands from a nominal-seed detector ensemble
- Optional line-pressure state estimate and residual chart
- Experimental rolling-window sequence score in validation
- CSV replay ingestion and drift summary scaffolding
- Perturbation-based signal contribution estimates
- Beginner Mode with plain-English chart captions and a signal glossary
- Scenario validation script for repeatable regression checks

## Why Hybrid Monitoring Matters

| Failure type | Example | Monitoring layer |
|---|---|---|
| Hard-limit breach | Pump current spikes above the safety threshold | Deterministic rule alert |
| Subtle multivariate drift | Flow, pressure, and pump current move into an unusual combined pattern | ML anomaly score |

Rules alone may miss gradual correlated degradation. ML alone is not appropriate as the authority for safety-critical decisions. Orbital Refueling Telemetry Simulator keeps both layers independent: rules provide hard engineering checks, while ML provides advisory pattern recognition.

## Refueling Phases

| Phase | Duration | What happens |
|---|---:|---|
| `approach` | 60 s | Servicer closes distance; attitude stabilizes |
| `arm_alignment` | 45 s | Robotic arm moves toward the target port |
| `docking` | 30 s | End effector makes contact; interface loads rise |
| `seal_check` | 20 s | Seal pressure ramps from 0 to 5 bar |
| `pressure_equalization` | 40 s | Transfer lines reach operating pressure |
| `main_transfer` | 120 s | Propellant flows at nominal 0.5 kg/s |
| `leak_check` | 30 s | Flow is halted; residual flow indicates a breach |
| `disconnect` | 20 s | Seal vents; arm retracts |
| `retreat` | 45 s | Servicer departs |

## Anomaly Scenarios

| Scenario | Injected behavior | Key signals |
|---|---|---|
| `nominal` | Clean baseline | None |
| `slow_leak` | Seal pressure drifts downward; leak-check flow rises | `seal_pressure`, `flow_rate` |
| `partial_blockage` | Flow drops while line pressure and pump current rise | `flow_rate`, `line_pressure`, `pump_current` |
| `pump_degradation` | Current spikes, flow becomes erratic, line heats up | `pump_current`, `flow_rate`, `line_temperature` |
| `arm_misalignment` | End-effector error stays high; interface loads rise | `end_effector_position_error`, `interface_force` |
| `sensor_drift` | Pressure readings drift upward over the mission | `line_pressure`, `donor_tank_pressure` |
| `sensor_dropout` | Pressure telemetry briefly drops to zero while other signals continue | `line_pressure`, `receiver_tank_pressure` |
| `stuck_at_pressure` | Line pressure freezes at a plausible value during transfer | `line_pressure` |
| `bias_oscillation` | Pressure sensor bias oscillates without crossing hard limits | `line_pressure`, `donor_tank_pressure` |
| `unstable_slosh` | Flow and line pressure oscillate during transfer | `flow_rate`, `line_pressure`, `propellant_temperature` |

## Project Structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `simulator.py` | Synthetic mission telemetry and anomaly injection |
| `rules.py` | Deterministic engineering rules and grouped alerts |
| `detector.py` | Phase-aware `IsolationForest` anomaly detector and nominal-seed uncertainty ensemble |
| `estimator.py` | Lightweight line-pressure estimator and residual calculation |
| `sequence_detector.py` | Experimental rolling-window temporal anomaly detector |
| `replay.py` | CSV replay schema validation and drift summary helpers |
| `explainer.py` | Perturbation-based feature attribution |
| `examples/replay_sample.csv` | Minimal replay CSV with expected schema |
| `scripts/validate_scenarios.py` | Scenario validation summary |
| `tests/` | Regression tests for simulator, rules, detector, and explanations |
| `docs/` | Architecture and implementation notes |
| `llm/` | Optional RefuelGuard-LM research extension for fine-tuned telemetry explanations |

## How To Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. The detector trains on nominal synthetic data on first load and then reuses Streamlit's cache for scenario switches.

Requirements: Python 3.10+, no GPU needed.

## Run Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -v
```

The test suite covers telemetry generation, rule logic, detector scoring, and explanation behavior.

Continuous integration runs the same test suite on Python 3.10, 3.11, and 3.12 through GitHub Actions.

## Run Scenario Validation

```bash
python3 scripts/validate_scenarios.py
```

The script trains or loads the default phase-aware detector, runs every scenario, prints a compact summary table, and writes `outputs/scenario_validation.csv`.

Key validation fields:

| Column | Meaning |
|---|---|
| `max_ml_score` | Highest phase-aware ML anomaly score observed |
| `mean_ml_score` | Average ML anomaly score across the mission |
| `mean_ml_uncertainty` | Average spread across the nominal-seed detector ensemble |
| `max_ml_uncertainty` | Largest ensemble score spread in the mission |
| `max_sequence_score` | Highest experimental rolling-window temporal anomaly score |
| `mean_sequence_score` | Average experimental temporal anomaly score |
| `max_pressure_residual` | Largest observed-vs-estimated line-pressure residual |
| `ml_anomaly_rate_pct` | Percent of rows above the dashboard ML threshold |
| `raw_rule_alert_count` | Count of raw deterministic rule alert rows |
| `grouped_rule_alert_count` | Count of grouped dashboard alert events |
| `highest_rule_severity` | Highest deterministic severity: `NONE`, `WARNING`, or `CRITICAL` |
| `top_contributing_signal` | Largest perturbation-based contributor to high ML scores |

Expected validation story:

- `nominal` remains quiet, with low ML scores and no rule alerts.
- `partial_blockage` triggers both ML scoring and deterministic pressure/current rules.
- `sensor_drift` raises ML scores while deterministic rules can remain quiet.
- `sensor_dropout` and `stuck_at_pressure` highlight sensor-fault behavior through ML scores and estimator residuals.
- `bias_oscillation` demonstrates temporal/sensor behavior that can stay below deterministic pressure limits.
- `unstable_slosh` raises ML scores through coupled flow, pressure, and thermal behavior without necessarily crossing hard thresholds.

These numbers are regression evidence for synthetic scenarios, not real-world performance estimates.

## RefuelGuard-LM: Fine-Tuned Telemetry Explanation Model

RefuelGuard-LM is an optional research extension under `llm/`. It uses Orbital Refueling Simulator synthetic telemetry to train a small open-source LLM adapter that explains anomaly summaries.

The LLM does not replace deterministic rules or the phase-aware anomaly detector. It explains their outputs: mission phase, scenario context, rule alerts, advisory ML anomaly score, and top contributing signals.

Generate instruction data:

```bash
python3 llm/data/generate_instruction_data.py --output-dir llm/data --n-examples 300 --seed 42
```

Train a LoRA adapter with a small open model:

```bash
pip install -r llm/training/requirements.txt
python3 llm/training/train_lora.py --base-model Qwen/Qwen2.5-0.5B-Instruct
```

Evaluate base versus fine-tuned outputs:

```bash
python3 llm/evals/evaluate_base_vs_finetuned.py \
  --adapter-path llm/training/outputs/refuelguard-lm-lora
```

This extension is synthetic-data research only. It is not flight software, not a certified diagnosis model, and not a safety decision authority.

## Optional Fine-Tuned LLM Explainer

The Streamlit dashboard can optionally use the fine-tuned RefuelGuard-LM adapter as an explanation mode. The architecture stays hybrid and conservative:

```text
Telemetry
  -> deterministic rules + phase-aware ML anomaly score
  -> structured explanation payload
  -> deterministic explainer and/or local fine-tuned LLM explainer
  -> Streamlit explanation text
```

The LLM does not detect anomalies, set alert severity, override deterministic rules, or replace the ML score. It only rewrites grounded fields from the simulator: phase, scenario, rule alerts, ML score, top contributing signals, signal changes, and current telemetry values.

To use it locally:

```bash
pip install -r requirements.txt
pip install -r llm/training/requirements.txt
streamlit run app.py
```

Then choose one of the dashboard explanation modes:

- `Deterministic explanation`
- `Fine-tuned LLM explanation`
- `Side-by-side comparison`

By default the app looks for the LoRA adapter at:

```text
llm/training/outputs/refuelguard-lm-lora
```

If the adapter or model cannot be loaded, the dashboard shows a warning and falls back to deterministic explanation text instead of crashing. LLM explanations are local-only and should include uncertainty wording, synthetic-data scope, and “not flight-certified” safety language.

### Streamlit Community Cloud behavior

The public Community Cloud deployment should be treated as a deterministic dashboard demo. The app disables local LLM loading on Community Cloud by default because the free resource envelope is not a good fit for loading Qwen + PEFT adapters. If a viewer selects `Fine-tuned LLM explanation`, the app shows a warning and falls back to deterministic explanation text.

On a larger host with local model assets, set this environment variable to opt in:

```bash
REFUELGUARD_ENABLE_LOCAL_LLM=1 streamlit run app.py
```

## Documentation

This repository keeps the public overview in `README.md` and `MODEL_CARD.md`. Extended local notes and walkthrough material are intentionally excluded from the public GitHub repository.

## Limitations

- Synthetic data only; the simulator is not a high-fidelity spacecraft model.
- Not flight software, an autonomous abort system, or a certified safety interlock.
- Rule thresholds are illustrative and not derived from real spacecraft engineering limits.
- `IsolationForest` scores each row independently; the rolling-window detector is experimental and only used as a comparison layer.
- Perturbation attribution is approximate and not causal proof.
- The detector is trained offline and does not perform online learning or drift adaptation.
- Uncertainty bands reflect sensitivity to synthetic nominal seeds, not calibrated confidence for real spacecraft telemetry.

## Future Improvements

- Calibrate uncertainty intervals on independent nominal data.
- Expand replay ingestion into streaming/online drift monitoring.
- Compare rolling-window temporal scores against stronger sequence models when a deep-learning dependency is justified.
- Add richer multi-fault scenarios and phase-label fault injection.
