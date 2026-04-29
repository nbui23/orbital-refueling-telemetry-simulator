# RefuelGuard-LM Instruction Dataset Schema

Rows are JSONL objects derived from Orbital Refueling Simulator synthetic runs.

## Fields

| Field | Type | Description |
|---|---|---|
| `task_type` | string | One of `explanation`, `classification`, `attribution`, `rule_vs_ml`, or `safety_uncertainty` |
| `instruction` | string | Common user instruction for the LLM |
| `input` | string | Telemetry summary produced from simulator, rule engine, ML detector, and attribution |
| `output` | string | Reference explanation generated from deterministic templates and simulator labels |

## Input Format

`input` contains:

- `Phase`: mission phase most relevant to the anomaly
- `Scenario`: simulator scenario label
- `Signals`: trend summary against a same-seed nominal run
- `Rule alerts`: highest deterministic rule severity in the relevant phase
- `Rule IDs`: matching rule names when present
- `ML anomaly score`: phase-aware detector score summary
- `Top contributors`: perturbation-based feature attribution signals

## Task Types

| Task type | Goal |
|---|---|
| `explanation` | Explain likely anomaly pattern in plain technical language |
| `classification` | Name the anomaly family and supporting evidence |
| `attribution` | Explain why top signals influenced the ML score |
| `rule_vs_ml` | Distinguish hard rule alerts from advisory ML scores |
| `safety_uncertainty` | Use careful uncertainty and synthetic-data safety wording |

## Example

```json
{
  "task_type": "explanation",
  "instruction": "Explain the likely spacecraft refueling anomaly from the telemetry summary.",
  "input": "Phase: main_transfer\nScenario: partial_blockage\nSignals: flow_rate decreased, line_pressure increased, pump_current increased\nRule alerts: warning\nRule IDs: LINE_PRESSURE_WARNING, PUMP_OVERCURRENT_WARNING\nML anomaly score: 0.82\nTop contributors: line_pressure, flow_rate, pump_current",
  "output": "This pattern is most consistent with partial_blockage during the refueling sequence. Flow_rate decreased, line_pressure increased, pump_current increased match the simulator behavior: transfer line partially blocked. Flow reduced; upstream pressure elevated; pump works harder. The rule alert level is warning, and the ML score of 0.82 adds advisory evidence of abnormal multivariate behavior. This explanation is based on synthetic telemetry, rule alerts, and advisory ML scores; it is not flight-certified."
}
```

## Label Generation

Labels come from simulator scenario names and deterministic templates. Signal trends are computed by comparing the scenario run to a same-seed nominal run for the focus phase. Rule severity comes from `evaluate_rules`. ML scores come from `PhaseAwareDetector` trained on nominal telemetry. Attribution comes from `explain_window`.

## Synthetic Limitations

- Telemetry is procedurally generated and simplified.
- Rule thresholds are illustrative.
- ML scores are advisory and not calibrated for real spacecraft.
- Attribution is perturbation-based and not causal proof.
- Dataset labels are template-generated, so they should not be treated as independent expert annotations.
