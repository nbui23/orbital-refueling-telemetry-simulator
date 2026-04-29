"""Orbital Refueling Telemetry Simulator dashboard.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from detector import EnsemblePhaseAwareDetector, PhaseAwareDetector
from estimator import estimate_line_pressure
from explainer import explain_window
from llm.explanation import (
    DETERMINISTIC_MODE,
    LLM_MODE,
    SIDE_BY_SIDE_MODE,
    build_deterministic_explanation,
    build_explanation_payload,
    resolve_explanation_mode,
)
from llm.explanation.llm_explainer import DEFAULT_ADAPTER_PATH, LocalLLMExplainer
from rules import alerts_to_dataframe, evaluate_rules, group_alerts
from simulator import (
    ANOMALY_SCENARIOS,
    PHASE_DURATIONS,
    PHASES,
    SCENARIO_DESCRIPTIONS,
    generate_telemetry,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orbital Refueling Telemetry Simulator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
PHASE_COLORS: dict[str, str] = {
    "approach":              "#4895ef",
    "arm_alignment":         "#f4a261",
    "docking":               "#2a9d8f",
    "seal_check":            "#e9c46a",
    "pressure_equalization": "#9b5de5",
    "main_transfer":         "#e63946",
    "leak_check":            "#fb8500",
    "disconnect":            "#8d99ae",
    "retreat":               "#70c1b3",
}

SEVERITY_BG: dict[str, str] = {"CRITICAL": "#e63946", "WARNING": "#f4a261"}
LINE_COLORS: list[str] = ["#4895ef", "#e63946", "#2a9d8f", "#f4a261", "#9b5de5", "#fb8500"]
PLOT_TEMPLATE = "plotly_dark"
_BG = "#0e1117"

CHART_HEIGHT = 300
SCORE_HEIGHT = 330
LLM_ADAPTER_PATH = Path(DEFAULT_ADAPTER_PATH)
_LOCAL_LLM_FLAG = "REFUELGUARD_ENABLE_LOCAL_LLM"


_GLOSSARY: dict[str, dict[str, str]] = {
    "phase": {
        "meaning": "The current step of the refueling operation, such as approach, docking, transfer, or retreat.",
        "matters": "Normal telemetry changes by phase. Flow is expected during transfer, but suspicious during leak check.",
        "suspicious": "A signal that looks normal in one phase may be unusual in another phase.",
    },
    "flow_rate": {
        "meaning": "How quickly propellant is moving through the transfer line.",
        "matters": "It shows whether fuel is moving as expected during the main transfer phase.",
        "suspicious": "Low flow during transfer may indicate blockage; non-zero flow during leak check may indicate leakage.",
    },
    "line_pressure": {
        "meaning": "Pressure inside the fuel transfer line.",
        "matters": "It helps show whether the line is operating within expected pressure bounds.",
        "suspicious": "High pressure may indicate blockage; slow drift may indicate a sensor or calibration issue.",
    },
    "seal_pressure": {
        "meaning": "Pressure holding the sealed connection between the two vehicles.",
        "matters": "A stable seal is needed before and during transfer.",
        "suspicious": "Falling seal pressure may indicate a seal problem or leak risk.",
    },
    "donor_tank_pressure": {
        "meaning": "Pressure in the vehicle giving propellant.",
        "matters": "It should change predictably as propellant leaves the donor tank.",
        "suspicious": "Unexpected drift may suggest sensor mismatch or abnormal transfer behavior.",
    },
    "receiver_tank_pressure": {
        "meaning": "Pressure in the vehicle receiving propellant.",
        "matters": "It should rise in a controlled way as propellant is received.",
        "suspicious": "Unexpected movement may suggest transfer imbalance or sensor inconsistency.",
    },
    "total_mass_transferred": {
        "meaning": "Running total of synthetic propellant transferred.",
        "matters": "It gives a simple progress indicator for the refueling operation.",
        "suspicious": "A flat or slow increase during transfer may suggest poor flow.",
    },
    "pump_current": {
        "meaning": "Electrical current used by the transfer pump.",
        "matters": "A struggling pump often draws more current.",
        "suspicious": "High current may indicate blockage, pump degradation, or abnormal load.",
    },
    "propellant_temperature": {
        "meaning": "Temperature of the synthetic propellant.",
        "matters": "Temperature changes can accompany fluid instability or transfer stress.",
        "suspicious": "Oscillation or unexpected heating may be consistent with slosh or thermal stress.",
    },
    "line_temperature": {
        "meaning": "Temperature of the transfer line.",
        "matters": "Line heating can accompany pump stress or abnormal transfer conditions.",
        "suspicious": "Rising temperature may indicate pump degradation or thermal load.",
    },
    "arm_joint_angle": {
        "meaning": "Position angle of the robotic arm joint.",
        "matters": "The arm should move toward a stable docking configuration.",
        "suspicious": "Unexpected angle behavior may suggest the arm is not following the expected approach path.",
    },
    "arm_joint_velocity": {
        "meaning": "How fast the robotic arm joint is moving.",
        "matters": "The arm should slow down and settle during contact.",
        "suspicious": "Motion when the arm should be locked may suggest unstable contact or control stress.",
    },
    "arm_motor_current": {
        "meaning": "Electrical current used by the robotic arm motor.",
        "matters": "It reflects how hard the arm is working.",
        "suspicious": "Elevated current may suggest mechanical load or resistance.",
    },
    "end_effector_position_error": {
        "meaning": "How far the arm tip is from the intended docking position.",
        "matters": "Small error means the arm is aligned; large error means contact may be unsafe.",
        "suspicious": "Persistent error may indicate misalignment.",
    },
    "interface_force": {
        "meaning": "Contact force at the docking interface.",
        "matters": "The vehicles should not press together too hard.",
        "suspicious": "High force may indicate misalignment or structural stress.",
    },
    "interface_torque": {
        "meaning": "Twisting load at the docking interface.",
        "matters": "Twisting can show whether contact is uneven.",
        "suspicious": "High torque may indicate rotational misalignment.",
    },
    "attitude_error": {
        "meaning": "How far the vehicle pointing is from the intended orientation.",
        "matters": "Stable pointing helps keep the refueling interface aligned.",
        "suspicious": "Large error may indicate attitude-control stress.",
    },
    "reaction_wheel_speed": {
        "meaning": "Speed of the wheels used to control vehicle orientation.",
        "matters": "It reflects how hard the vehicle is working to stay stable.",
        "suspicious": "Unexpected speed changes may suggest attitude-control effort.",
    },
    "bus_voltage": {
        "meaning": "Main electrical bus voltage.",
        "matters": "The system needs stable power for pumps, sensors, and control systems.",
        "suspicious": "Low voltage may indicate power stress.",
    },
    "deterministic rule": {
        "meaning": "A fixed engineering check, such as pressure above a limit or voltage below a limit.",
        "matters": "Rules are simple, explicit, and independent of ML.",
        "suspicious": "A rule alert means a specific threshold was crossed.",
    },
    "ML anomaly score": {
        "meaning": "An advisory dashboard score from 0 to 100 showing how unlike nominal telemetry the current pattern looks.",
        "matters": "It can surface subtle multivariate patterns before a single hard limit is crossed.",
        "suspicious": "Scores above 45 warrant attention in this dashboard, but are not automatic actions.",
    },
    "grouped alert event": {
        "meaning": "Repeated rule alerts merged into one readable event window.",
        "matters": "It keeps the dashboard readable when one issue triggers many timestamp-level alerts.",
        "suspicious": "A grouped event summarizes one sustained rule violation window.",
    },
    "synthetic/demo units": {
        "meaning": "Generated values with plausible scale, not measurements from a real spacecraft.",
        "matters": "They keep the dashboard realistic enough to explain patterns without claiming real flight accuracy.",
        "suspicious": "The exact number is less important than the trend, timing, and relationship between signals.",
    },
    "phase-aware": {
        "meaning": "The detector compares each row against normal behavior for that specific refueling phase.",
        "matters": "Normal changes by phase: flow is expected during transfer but suspicious during leak check.",
        "suspicious": "Using the wrong phase context can make normal behavior look unusual, or unusual behavior look normal.",
    },
    "end effector": {
        "meaning": "The tool or tip of the robotic arm that connects to the target interface.",
        "matters": "It must align accurately before a stable transfer connection is possible.",
        "suspicious": "Large position error may suggest the arm tip is not seated correctly.",
    },
    "attitude": {
        "meaning": "The spacecraft's orientation in space.",
        "matters": "Stable orientation helps keep the docking and transfer interface aligned.",
        "suspicious": "Large attitude error may suggest vehicle stability stress.",
    },
    "reaction wheel": {
        "meaning": "A spinning wheel used to help control spacecraft orientation.",
        "matters": "Wheel speed reflects how hard the spacecraft is working to stay stable.",
        "suspicious": "Rising wheel speed together with attitude error may suggest disturbance or control stress.",
    },
}

_SIGNAL_UNITS: dict[str, str] = {
    "attitude_error": "deg",
    "reaction_wheel_speed": "RPM",
    "arm_joint_angle": "deg",
    "arm_joint_velocity": "deg/s",
    "arm_motor_current": "A",
    "end_effector_position_error": "mm",
    "interface_force": "N",
    "interface_torque": "Nm",
    "seal_pressure": "bar",
    "donor_tank_pressure": "bar",
    "receiver_tank_pressure": "bar",
    "line_pressure": "bar",
    "flow_rate": "kg/s",
    "total_mass_transferred": "kg",
    "pump_current": "A",
    "propellant_temperature": "deg C",
    "line_temperature": "deg C",
    "bus_voltage": "V",
}

_SCENARIO_INTERPRETATION: dict[str, dict[str, str]] = {
    "nominal": {
        "noticed": "Telemetry mostly follows the nominal synthetic profile.",
        "may_indicate": "A healthy baseline for this synthetic scenario.",
        "check": "Use this view as the quiet baseline: no grouped rule events and low ML scoring.",
    },
    "slow_leak": {
        "noticed": "Seal pressure and leak-check flow become less consistent with nominal behavior.",
        "may_indicate": "A developing leak or seal-integrity issue in the synthetic scenario.",
        "check": "Inspect seal pressure trend, leak-check flow, and grouped seal/leak rule events.",
    },
    "partial_blockage": {
        "noticed": "Flow, line pressure, and pump current move together in an unusual transfer pattern.",
        "may_indicate": "A partial transfer-line blockage or pump load increase.",
        "check": "Inspect grouped pressure/current rule events and compare flow against pump current.",
    },
    "pump_degradation": {
        "noticed": "Pump current, flow stability, and line temperature are less nominal-like during transfer.",
        "may_indicate": "Pump degradation or pump thermal stress.",
        "check": "Inspect pump current spikes, transfer flow, and line temperature.",
    },
    "arm_misalignment": {
        "noticed": "Arm position error remains high and contact loads are elevated.",
        "may_indicate": "Robotic arm misalignment or docking contact stress.",
        "check": "Inspect end-effector error, interface force, interface torque, and grouped arm/contact rule events.",
    },
    "sensor_drift": {
        "noticed": "Pressure readings drift away from the usual multivariate pattern while rules remain quiet.",
        "may_indicate": "A sensor drift pattern rather than a hard pressure-limit breach.",
        "check": "Compare line pressure with donor/receiver tank pressure and verify sensor calibration assumptions.",
    },
    "sensor_dropout": {
        "noticed": "Pressure readings briefly collapse while other transfer signals continue.",
        "may_indicate": "A sensor dropout or telemetry acquisition gap rather than a physical pressure loss.",
        "check": "Compare pressure dropouts with flow rate, pump current, and estimator residuals.",
    },
    "stuck_at_pressure": {
        "noticed": "Line pressure becomes too flat during phases where small variation is expected.",
        "may_indicate": "A stuck-at pressure sensor fault.",
        "check": "Compare line pressure against flow, pump current, and pressure estimator residuals.",
    },
    "bias_oscillation": {
        "noticed": "Pressure readings oscillate with a sensor-like bias pattern without necessarily crossing hard limits.",
        "may_indicate": "Sensor calibration oscillation or signal-conditioning fault.",
        "check": "Compare pressure oscillations with flow, donor pressure, and temporal ML score.",
    },
    "unstable_slosh": {
        "noticed": "Flow and pressure oscillate during transfer instead of staying smooth.",
        "may_indicate": "Fluid slosh or attitude-control stress in the synthetic transfer scenario.",
        "check": "Inspect flow, line pressure, propellant temperature, attitude error, and reaction wheel speed.",
    },
}

_CHART_HELP: dict[str, dict[str, object]] = {
    "mission_phase": {
        "title": "Mission Phase Timeline",
        "axis_explanation": (
            "The x-axis is mission elapsed time in seconds. The y-axis lists the active refueling phase."
        ),
        "synthetic_generation": (
            "The simulator runs a fixed 410-second mission script: approach, arm alignment, docking, seal check, pressure equalization, "
            "main transfer, leak check, disconnect, and retreat."
        ),
        "rule_vs_ml": (
            "Phase context is relevant to both layers. Rules are scoped to phases where a check is meaningful, and the ML detector compares telemetry "
            "against nominal examples from the same phase."
        ),
        "operational_relevance": (
            "A value can be normal in one phase and suspicious in another. This timeline provides the context needed to interpret every other chart."
        ),
        "signals": {
            "phase": {
                "meaning": "The current step of the refueling operation.",
                "up": "Moving downward through the listed phases means the mission is progressing.",
                "down": "Moving backward is not expected in this fixed synthetic replay.",
                "possible_issue": "Unexpected phase context would make telemetry interpretation unreliable.",
            },
        },
    },
    "vehicle_stability": {
        "title": "Vehicle Stability During Refueling",
        "axis_explanation": (
            "The x-axis is mission elapsed time in seconds. The y-axis uses synthetic/demo units: "
            "attitude error is in degrees and reaction wheel speed is in RPM."
        ),
        "synthetic_generation": (
            "The simulator generates a stable approach, docking, and transfer profile with small random noise. "
            "Different phases have different expected stability behavior."
        ),
        "rule_vs_ml": (
            "This chart is checked by both layers. Deterministic rules catch obvious attitude-control stress. "
            "ML can surface oscillatory or phase-dependent stability changes that are less obvious from one threshold."
        ),
        "operational_relevance": (
            "This chart indicates whether the vehicle is staying pointed correctly during refueling. "
            "If attitude error and reaction wheel speed rise together, it may suggest the transfer operation is disturbing vehicle stability."
        ),
        "signals": {
            "attitude_error": {
                "meaning": "How far the vehicle orientation is from where it should be.",
                "up": "Rising attitude error means pointing is getting worse.",
                "down": "Falling attitude error means the vehicle is settling closer to the intended orientation.",
                "possible_issue": "High or rising values may suggest attitude-control stress.",
            },
            "reaction_wheel_speed": {
                "meaning": "How hard the spacecraft is working to stay stable.",
                "up": "Rising wheel speed means the attitude-control system is working harder.",
                "down": "Falling speed can mean the vehicle is settling or using less control effort.",
                "possible_issue": "Rising speed together with attitude error may suggest disturbance or control stress.",
            },
        },
    },
    "robotic_arm": {
        "title": "Robotic Arm Alignment and Contact",
        "axis_explanation": (
            "The x-axis is mission elapsed time in seconds. The y-axis mixes synthetic/demo units such as degrees, "
            "millimetres, amps, newtons, and newton-metres depending on the signal."
        ),
        "synthetic_generation": (
            "The simulator moves the arm through alignment, docking, contact, transfer, and disconnect phases. "
            "Nominal data converges toward low position error with controlled contact loads."
        ),
        "rule_vs_ml": (
            "This chart is checked by both layers. Deterministic rules catch high alignment error or force spikes. "
            "ML can catch unusual combinations of arm effort, force, torque, and phase."
        ),
        "operational_relevance": (
            "This is the mechanical interface view. The arm tip should converge toward the target, and contact loads should stay controlled. "
            "If the arm is misaligned, the error and contact loads can rise together."
        ),
        "signals": {
            "end_effector_position_error": {
                "meaning": "How far the robotic arm tip is from the desired docking target.",
                "up": "Rising error means the arm tip is farther from the target.",
                "down": "Lower is better during docking because it means alignment is improving.",
                "possible_issue": "Persistent high error may suggest arm misalignment.",
            },
            "arm_joint_angle": {
                "meaning": "The arm joint's position angle.",
                "up": "Rising angle usually reflects the arm moving into docking position.",
                "down": "Falling angle can be expected during disconnect or retreat.",
                "possible_issue": "Unexpected movement for the current phase may suggest the arm is not following the expected path.",
            },
            "arm_joint_velocity": {
                "meaning": "How fast the arm joint is moving.",
                "up": "Higher velocity means faster arm movement.",
                "down": "Near zero is expected when the arm is locked or settled.",
                "possible_issue": "Motion when the arm should be settled may suggest unstable contact.",
            },
            "arm_motor_current": {
                "meaning": "How hard the arm motor is electrically working.",
                "up": "Rising current may mean the arm is working harder.",
                "down": "Lower current often means lower mechanical load or idle behavior.",
                "possible_issue": "High current with force/torque changes may suggest mechanical resistance.",
            },
            "interface_force": {
                "meaning": "Contact force at the docking interface.",
                "up": "Rising force means stronger contact loads.",
                "down": "Lower force can mean the interface is unloaded or separating.",
                "possible_issue": "Spikes may suggest rough contact, over-stress, or misalignment.",
            },
            "interface_torque": {
                "meaning": "Twisting load at the docking interface.",
                "up": "Rising torque means more twisting contact load.",
                "down": "Lower torque suggests less rotational stress.",
                "possible_issue": "High torque may suggest rotational misalignment.",
            },
        },
    },
    "fuel_transfer": {
        "title": "Fuel Transfer Health",
        "axis_explanation": (
            "The x-axis is mission elapsed time in seconds. The y-axis uses synthetic/demo units for pressure, flow, or mass depending on the selected signal."
        ),
        "synthetic_generation": (
            "The simulator creates phase-specific pressure and flow profiles: little or no flow before transfer, rising flow during transfer, "
            "and changing donor/receiver pressures as synthetic propellant moves."
        ),
        "rule_vs_ml": (
            "This chart is checked by both layers. Deterministic rules can catch strong pressure/flow mismatches. "
            "ML can catch weaker multivariate patterns before a hard rule fires."
        ),
        "operational_relevance": (
            "This is the primary transfer-performance chart. During main transfer, flow should rise and receiver tank pressure should increase. "
            "If flow drops while line pressure rises, that can be consistent with blockage or restriction."
        ),
        "signals": {
            "flow_rate": {
                "meaning": "How quickly synthetic propellant is moving through the line.",
                "up": "Rising flow during main_transfer is expected.",
                "down": "Dropping flow during transfer may suggest transfer is restricted.",
                "possible_issue": "Flow dropping while line pressure rises may suggest blockage or restriction.",
            },
            "line_pressure": {
                "meaning": "Pressure inside the transfer line.",
                "up": "Rising pressure can be normal during pressure_equalization, but high pressure during transfer can be concerning.",
                "down": "Falling pressure may be expected during disconnect, but instability can matter during transfer.",
                "possible_issue": "Pressure instability may suggest blockage, leak, sensor drift, or transfer instability.",
            },
            "donor_tank_pressure": {
                "meaning": "Pressure in the vehicle giving propellant.",
                "up": "Unexpected rising donor pressure may suggest sensor mismatch in this synthetic setup.",
                "down": "A controlled decrease can be expected as propellant leaves the donor side.",
                "possible_issue": "Unexpected drift may be consistent with sensor drift or abnormal transfer behavior.",
            },
            "receiver_tank_pressure": {
                "meaning": "Pressure in the vehicle receiving propellant.",
                "up": "Rising receiver pressure during transfer is expected.",
                "down": "Falling or flat pressure during transfer may suggest poor receipt of propellant.",
                "possible_issue": "Unexpected movement may suggest transfer imbalance or sensor inconsistency.",
            },
            "total_mass_transferred": {
                "meaning": "Running total of synthetic propellant moved.",
                "up": "Rising mass during main_transfer is expected.",
                "down": "This should not meaningfully go down in this synthetic replay.",
                "possible_issue": "A flat or slow rise during transfer may suggest reduced flow.",
            },
        },
    },
    "seal_leak": {
        "title": "Seal and Leak Integrity",
        "axis_explanation": (
            "The x-axis is mission elapsed time in seconds. The y-axis uses synthetic/demo pressure units such as bar."
        ),
        "synthetic_generation": (
            "The simulator ramps seal pressure during seal_check, holds it during transfer and leak_check, then vents it during disconnect."
        ),
        "rule_vs_ml": (
            "This chart is checked by both layers. Deterministic rules are especially appropriate for clear leak-check failures. "
            "ML may help identify subtle pressure behavior that does not cross a hard threshold."
        ),
        "operational_relevance": (
            "The connection must hold pressure before, during, and after transfer. Falling seal pressure or pressure instability may suggest leakage or poor seal integrity."
        ),
        "signals": {
            "seal_pressure": {
                "meaning": "Pressure holding the refueling connection sealed.",
                "up": "Rising seal pressure is expected during seal_check.",
                "down": "Falling seal pressure after the seal is established may suggest seal loss.",
                "possible_issue": "Falling or unstable seal pressure may suggest leakage or poor seal integrity.",
            },
            "line_pressure": {
                "meaning": "Pressure inside the transfer line.",
                "up": "Rising pressure can be expected during pressure_equalization, but sustained high pressure can be concerning.",
                "down": "Falling pressure may be expected during disconnect, but an early drop can matter during transfer or leak check.",
                "possible_issue": "Pressure instability may suggest leak risk, blockage, or transfer instability.",
            },
            "receiver_tank_pressure": {
                "meaning": "Pressure in the receiving vehicle tank.",
                "up": "Rising receiver pressure during transfer is expected.",
                "down": "Unexpected falling pressure can suggest the receiving side is not holding expected conditions.",
                "possible_issue": "Unexpected movement may suggest transfer imbalance, leakage, or sensor inconsistency.",
            },
        },
    },
    "thermal_power": {
        "title": "Thermal and Power Health",
        "axis_explanation": (
            "The x-axis is mission elapsed time in seconds. The y-axis mixes synthetic/demo units: amps, degrees Celsius, and volts."
        ),
        "synthetic_generation": (
            "The simulator adds small thermal and electrical variation around nominal values, with scenario-specific pump or line heating when relevant."
        ),
        "rule_vs_ml": (
            "This chart is checked by both layers. Deterministic rules handle hard low-voltage or high-current conditions. "
            "ML can catch subtler patterns like pump current drifting upward over time."
        ),
        "operational_relevance": (
            "This chart connects pump effort, thermal response, and power health. Pump current shows how hard the pump is working, temperature shows thermal response, "
            "and bus voltage shows whether the power system remains healthy."
        ),
        "signals": {
            "pump_current": {
                "meaning": "How hard the pump is electrically working.",
                "up": "Rising pump current means the pump is drawing more power.",
                "down": "Falling current can be expected when the pump is off or lightly loaded.",
                "possible_issue": "Pump current rising while flow drops may suggest pump degradation or blockage.",
            },
            "propellant_temperature": {
                "meaning": "Synthetic propellant temperature.",
                "up": "Rising temperature may reflect transfer or thermal load.",
                "down": "Falling temperature may reflect returning toward baseline.",
                "possible_issue": "Temperature rising too quickly may suggest abnormal fluid or pump behavior.",
            },
            "line_temperature": {
                "meaning": "Synthetic transfer-line temperature.",
                "up": "Rising line temperature may suggest heating during stressed transfer.",
                "down": "Falling line temperature may reflect lower load or post-transfer cooling.",
                "possible_issue": "Line heating with pump-current changes may suggest abnormal pump/line behavior.",
            },
            "bus_voltage": {
                "meaning": "Main synthetic electrical bus voltage.",
                "up": "Small upward movement is usually not the main concern here.",
                "down": "Dropping bus voltage may suggest power concern.",
                "possible_issue": "Low voltage may indicate power-system stress.",
            },
        },
    },
    "ml_score": {
        "title": "ML Early-Warning Score",
        "axis_explanation": (
            "The x-axis is mission elapsed time in seconds. The y-axis is an advisory anomaly score from 0 to 100."
        ),
        "synthetic_generation": (
            "The score comes from a phase-aware IsolationForest trained on synthetic nominal runs. "
            "The model compares the current telemetry pattern against normal examples for the same phase."
        ),
        "rule_vs_ml": (
            "This chart is the ML advisory layer. It is not a diagnosis and not an abort command. "
            "Deterministic rules remain responsible for hard engineering checks."
        ),
        "operational_relevance": (
            "This chart provides advisory early warning. A higher score means the telemetry looks less like nominal behavior for that phase. "
            "If the ML score rises before grouped rule events appear, the combined telemetry pattern may be changing before a hard rule fires."
        ),
        "signals": {
            "ML anomaly score": {
                "meaning": "How unlike nominal synthetic telemetry the current pattern looks for the current phase.",
                "up": "Higher score means less nominal-like behavior and more reason to investigate.",
                "down": "Lower score means the telemetry looks more consistent with nominal examples.",
                "possible_issue": "High score may suggest a subtle multivariate anomaly, but it does not identify a root cause by itself.",
            },
            "threshold line": {
                "meaning": "The dashboard attention threshold, shown at 45 out of 100.",
                "up": "Not applicable: the line is fixed.",
                "down": "Not applicable: the line is fixed.",
                "possible_issue": "Scores above the line warrant attention; they are not automatic actions.",
            },
            "rule markers": {
                "meaning": "Triangle markers show deterministic rule alerts on the same timeline.",
                "up": "Not applicable: markers indicate alert timing, not a measured value.",
                "down": "Not applicable: markers indicate alert timing, not a measured value.",
                "possible_issue": "Markers show hard threshold crossings that should be reviewed independently of ML.",
            },
        },
    },
    "attribution": {
        "title": "Signals Contributing Most to the ML Score",
        "axis_explanation": (
            "The x-axis is an attribution cue: how much the ML score drops when a signal is replaced with its nominal phase average. "
            "The y-axis lists the signals with the largest score effect."
        ),
        "synthetic_generation": (
            "For high-score rows, the dashboard tests one signal at a time against the detector's nominal phase baseline. "
            "This is a lightweight approximation, not a causal explanation."
        ),
        "rule_vs_ml": (
            "This chart explains the ML advisory layer only. Deterministic rules still provide explicit hard engineering checks."
        ),
        "operational_relevance": (
            "The chart highlights which signals are most consistent with the elevated ML score, so follow-up review can focus on the relevant subsystem."
        ),
        "signals": {
            "attribution cue": {
                "meaning": "A relative score-drop measure for one signal at a time.",
                "up": "A larger bar means that signal contributed more to the ML score in the sampled high-score rows.",
                "down": "A smaller bar means that signal had less effect on the score.",
                "possible_issue": "Large bars identify signals worth reviewing, but they do not confirm root cause.",
            },
        },
    },
}


# ── Cached computation ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training anomaly detector on nominal baseline…")
def _load_detector() -> PhaseAwareDetector:
    nom = generate_telemetry("nominal", seed=0)
    return PhaseAwareDetector(contamination=0.02, n_estimators=200).fit(nom)


@st.cache_resource(show_spinner="Training ML uncertainty ensemble on nominal seeds...")
def _load_ensemble() -> EnsemblePhaseAwareDetector:
    return EnsemblePhaseAwareDetector(seeds=(0, 1, 2, 3, 4), n_estimators=80).fit()


@st.cache_data(show_spinner=False)
def _run_scenario(
    scenario: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    detector = _load_detector()
    ensemble = _load_ensemble()
    df = generate_telemetry(scenario, seed=42)
    scores = detector.score(df)
    score_summary = ensemble.score_summary(df)
    raw_alerts = evaluate_rules(df)
    alerts_df = alerts_to_dataframe(raw_alerts)      # timestamp-level (score plot overlay)
    grouped_df = group_alerts(raw_alerts, gap_s=15.0) # event windows (dashboard display)
    contribs = explain_window(detector, df, scores, threshold=0.45, n_top=8)
    estimates = estimate_line_pressure(df)
    return df, scores, alerts_df, grouped_df, contribs, score_summary, estimates


@st.cache_resource(show_spinner="Loading local fine-tuned LLM explainer...")
def _load_llm_explainer(adapter_path: str) -> LocalLLMExplainer:
    return LocalLLMExplainer(adapter_path=adapter_path, local_files_only=True)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _phase_bands(fig: go.Figure, df: pd.DataFrame) -> None:
    for phase in PHASES:
        mask = df["phase"] == phase
        if not mask.any():
            continue
        fig.add_vrect(
            x0=float(df.loc[mask, "time"].min()),
            x1=float(df.loc[mask, "time"].max()),
            fillcolor=PHASE_COLORS[phase],
            opacity=0.07,
            line_width=0,
        )


def _label_for(signal: str) -> str:
    return signal.replace("_", " ").title()


def _unit_for(signal: str) -> str:
    return _SIGNAL_UNITS.get(signal, "demo units")


def _base_layout(title: str, yaxis_label: str, height: int = 230) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13, color="#cccccc")),
        height=height,
        margin=dict(l=12, r=16, t=64, b=44),
        xaxis=dict(title="Mission time (s)", showgrid=False, color="#888"),
        yaxis=dict(title=yaxis_label, showgrid=True, gridcolor="#1e1e2e", color="#888"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        hovermode="x unified",
        template=PLOT_TEMPLATE,
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
    )


def _signal_plot(
    df: pd.DataFrame,
    signals: list[tuple[str, str]],
    title: str,
    yaxis_label: str = "",
    height: int = CHART_HEIGHT,
) -> go.Figure:
    fig = go.Figure()
    _phase_bands(fig, df)
    for i, (col, label) in enumerate(signals):
        unit = _unit_for(col)
        hover_label = label.replace(" (synthetic/demo units)", "")
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df[col],
            mode="lines",
            name=label,
            line=dict(width=1.6, color=LINE_COLORS[i % len(LINE_COLORS)]),
            customdata=df["phase"].str.replace("_", " ").str.title(),
            hovertemplate=(
                f"<b>{hover_label}</b><br>"
                "Time: %{x:.1f} s<br>"
                "Phase: %{customdata}<br>"
                f"Value: %{{y:.3f}} {unit}"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(**_base_layout(title, yaxis_label, height))
    return fig


def _score_plot(
    df: pd.DataFrame,
    scores: np.ndarray,
    score_summary: pd.DataFrame,
    alerts_df: pd.DataFrame,
) -> go.Figure:
    smoothed = pd.Series(scores).rolling(5, center=True, min_periods=1).mean().values
    score_pct = scores * 100.0
    smoothed_pct = smoothed * 100.0

    fig = go.Figure()
    _phase_bands(fig, df)
    fig.add_trace(go.Scatter(
        x=pd.concat([df["time"], df["time"].iloc[::-1]]),
        y=pd.concat([
            score_summary["ml_score_high"] * 100.0,
            (score_summary["ml_score_low"] * 100.0).iloc[::-1],
        ]),
        fill="toself",
        fillcolor="rgba(244, 162, 97, 0.16)",
        line=dict(color="rgba(244, 162, 97, 0)"),
        name="ML uncertainty band",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=df["time"], y=score_pct,
        mode="lines", name="Raw ML score",
        line=dict(width=0.8, color="#4895ef", dash="dot"),
        opacity=0.40,
        customdata=df["phase"].str.replace("_", " ").str.title(),
        hovertemplate=(
            "<b>Raw ML score</b><br>"
            "Time: %{x:.1f} s<br>"
            "Phase: %{customdata}<br>"
            "Score: %{y:.1f} / 100<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=smoothed_pct,
        mode="lines", name="Smoothed ML score",
        line=dict(width=2.2, color="#4895ef"),
        fill="tozeroy",
        fillcolor="rgba(72,149,239,0.10)",
        customdata=df["phase"].str.replace("_", " ").str.title(),
        hovertemplate=(
            "<b>Smoothed ML score</b><br>"
            "Time: %{x:.1f} s<br>"
            "Phase: %{customdata}<br>"
            "Score: %{y:.1f} / 100<extra></extra>"
        ),
    ))
    fig.add_hline(
        y=45,
        line=dict(color="#e63946", dash="dash", width=1.5),
        annotation_text="Attention threshold (45/100)",
        annotation_position="top right",
        annotation_font=dict(color="#e63946", size=11),
    )

    if len(alerts_df):
        for sev in ("CRITICAL", "WARNING"):
            sub = alerts_df[alerts_df["severity"] == sev]
            if len(sub):
                fig.add_trace(go.Scatter(
                    x=sub["time"],
                    y=[96] * len(sub),
                    mode="markers",
                    name=f"Rule {sev.lower()}",
                    marker=dict(symbol="triangle-down", size=10, color=SEVERITY_BG[sev]),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Time: %{x:.1f} s<br>"
                        "Severity: %{customdata[1]}"
                        "<extra></extra>"
                    ),
                    customdata=np.stack([sub["rule"].values, sub["severity"].values], axis=-1),
                ))

    layout = _base_layout("ML Early-Warning Score", "Advisory score (0-100)", height=SCORE_HEIGHT)
    layout["yaxis"] = dict(
        title="Advisory score (0-100)",
        range=[0, 100],
        showgrid=True,
        gridcolor="#1e1e2e",
        color="#888",
    )
    fig.update_layout(**layout)
    return fig


def _estimator_plot(estimates: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _phase_bands(fig, estimates.rename(columns={"line_pressure_observed": "line_pressure"}))
    fig.add_trace(go.Scatter(
        x=estimates["time"],
        y=estimates["line_pressure_observed"],
        mode="lines",
        name="Observed line pressure",
        line=dict(width=1.4, color="#4895ef"),
    ))
    fig.add_trace(go.Scatter(
        x=estimates["time"],
        y=estimates["line_pressure_estimated"],
        mode="lines",
        name="Estimated line pressure",
        line=dict(width=2.0, color="#2a9d8f"),
    ))
    fig.add_trace(go.Scatter(
        x=estimates["time"],
        y=estimates["line_pressure_residual"],
        mode="lines",
        name="Residual",
        line=dict(width=1.0, color="#f4a261", dash="dot"),
        yaxis="y2",
    ))
    layout = _base_layout("Line Pressure Estimate and Residual", "line pressure (bar)", height=320)
    layout["yaxis2"] = dict(
        title="residual (bar)",
        overlaying="y",
        side="right",
        showgrid=False,
        color="#888",
    )
    fig.update_layout(**layout)
    return fig


def _phase_timeline_plot(df: pd.DataFrame) -> go.Figure:
    phase_to_y = {phase: idx for idx, phase in enumerate(PHASES)}
    y_values = df["phase"].map(phase_to_y)
    phase_labels = df["phase"].str.replace("_", " ").str.title()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=y_values,
        mode="lines",
        line=dict(width=10, color="#4895ef"),
        customdata=phase_labels,
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "Mission time: %{x:.1f} s"
            "<extra></extra>"
        ),
        name="Active phase",
    ))
    for phase in PHASES:
        mask = df["phase"] == phase
        if not mask.any():
            continue
        fig.add_vrect(
            x0=float(df.loc[mask, "time"].min()),
            x1=float(df.loc[mask, "time"].max()),
            fillcolor=PHASE_COLORS[phase],
            opacity=0.10,
            line_width=0,
        )
    fig.update_layout(
        title=dict(text="Mission Phase Timeline", font=dict(size=13, color="#cccccc")),
        height=240,
        margin=dict(l=12, r=16, t=54, b=38),
        xaxis=dict(title="Mission time (s)", showgrid=False, color="#888"),
        yaxis=dict(
            title="Active phase",
            tickmode="array",
            tickvals=list(phase_to_y.values()),
            ticktext=[phase.replace("_", " ").title() for phase in PHASES],
            autorange="reversed",
            showgrid=False,
            color="#888",
        ),
        showlegend=False,
        template=PLOT_TEMPLATE,
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
    )
    return fig


def _contribution_plot(contribs: dict[str, float]) -> go.Figure:
    if not contribs:
        fig = go.Figure()
        fig.update_layout(
            title="No anomalous rows above threshold — try an anomaly scenario",
            template=PLOT_TEMPLATE,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            height=220,
        )
        return fig

    labels = [_label_for(k) for k in contribs]
    values = list(contribs.values())
    bar_colors = [
        "#e63946" if v > 0.06 else "#f4a261" if v > 0.01 else "#8d99ae"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{v:+.3f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Signals Contributing Most to the ML Score", font=dict(size=13, color="#cccccc")),
        height=max(240, len(contribs) * 36 + 90),
        margin=dict(l=12, r=84, t=58, b=42),
        xaxis=dict(title="Attribution cue (score drop when set to nominal)", showgrid=False, color="#888"),
        yaxis=dict(showgrid=False, color="#ccc"),
        template=PLOT_TEMPLATE,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
    )
    return fig


# ── Signal descriptions ───────────────────────────────────────────────────────
_SIGNAL_DESC: dict[str, str] = {
    "pump_current":                "Pump drive current — spikes may indicate degradation or blockage",
    "flow_rate":                   "Propellant mass flow — sustained drop may indicate blockage; non-zero during leak check may indicate breach",
    "seal_pressure":               "Interface seal pressure — downward drift may indicate seal failure",
    "line_pressure":               "Transfer line pressure — over-range may indicate blockage; gradual drift may indicate sensor fault",
    "end_effector_position_error": "Arm end-effector alignment error — persistently high values may indicate misalignment",
    "interface_force":             "Docking port contact force — elevated values may indicate misalignment or over-stress",
    "interface_torque":            "Docking port torque — elevated values may indicate rotational misalignment",
    "propellant_temperature":      "Propellant temperature — oscillating patterns may indicate slosh instability",
    "line_temperature":            "Transfer line temperature — elevated values may indicate thermal fault or pump heating",
    "attitude_error":              "Spacecraft pointing error — engineering limit is 2°; operation halts above 5°",
    "donor_tank_pressure":         "Donor tank pressure — tracks propellant depletion during transfer",
    "receiver_tank_pressure":      "Receiver tank pressure — tracks propellant fill level during transfer",
    "arm_joint_angle":             "Robotic arm joint angle — converges to ~45° during docking phase",
    "arm_motor_current":           "Arm actuator current — elevated during active motion, near idle when locked",
    "reaction_wheel_speed":        "Reaction wheel speed — reflects attitude control effort",
    "bus_voltage":                 "Spacecraft power bus — engineering limit is 27 V",
    "arm_joint_velocity":          "Arm joint angular velocity — near zero when arm is locked in position",
}


def _what_to_look_for(text: str, beginner_mode: bool) -> None:
    if beginner_mode:
        st.caption(f"What to look for: {text}")


def _render_signal_help(signals: list[str], beginner_mode: bool) -> None:
    if not beginner_mode:
        return
    help_bits = []
    for signal in signals:
        entry = _GLOSSARY.get(signal)
        if entry:
            help_bits.append(f"**{_label_for(signal)}:** {entry['meaning']}")
    if help_bits:
        st.caption("  \n".join(help_bits))


def _render_signal_interpretation_table(signals: dict[str, dict[str, str]]) -> None:
    rows = []
    for signal, info in signals.items():
        rows.append({
            "signal": _label_for(signal),
            "meaning": info["meaning"],
            "if it goes up": info["up"],
            "if it goes down": info["down"],
            "pattern may suggest": info["possible_issue"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_chart_help(chart_key: str, beginner_mode: bool) -> None:
    help_data = _CHART_HELP[chart_key]
    expanded = beginner_mode
    with st.expander(f"How to read this chart: {help_data['title']}", expanded=expanded):
        st.markdown(f"**Axes:** {help_data['axis_explanation']}")
        st.markdown(f"**How the synthetic data is generated:** {help_data['synthetic_generation']}")
        st.markdown("**How to read each signal:**")
        _render_signal_interpretation_table(help_data["signals"])  # type: ignore[arg-type]
        st.markdown(f"**Rules vs ML:** {help_data['rule_vs_ml']}")
        st.info(f"Operational relevance: {help_data['operational_relevance']}", icon="ℹ️")


def _top_phase(df: pd.DataFrame, scores: np.ndarray) -> str:
    high = df.loc[scores > 0.45, "phase"]
    if len(high) == 0:
        return str(df["phase"].iloc[0])
    return str(high.mode().iloc[0])


def _render_glossary(beginner_mode: bool) -> None:
    st.markdown("### Learn the Signals")
    st.caption(
        "Plain-English guide to the telemetry and alert terms used in this synthetic dashboard."
    )
    st.info(
        "**Synthetic/demo units** means the numbers are plausible generated values, not real spacecraft measurements. "
        "Some charts place signals with different units together so you can compare timing and shape. "
        "The important clue is often the pattern across signals, not the exact numeric value.",
        icon="ℹ️",
    )

    terms = list(_GLOSSARY.keys())
    if beginner_mode:
        for term in terms:
            entry = _GLOSSARY[term]
            with st.expander(_label_for(term), expanded=term in ("ML anomaly score", "deterministic rule", "grouped alert event")):
                st.markdown(f"**Meaning:** {entry['meaning']}")
                st.markdown(f"**Why it matters:** {entry['matters']}")
                st.markdown(f"**Suspicious behavior may suggest:** {entry['suspicious']}")
    else:
        glossary_df = pd.DataFrame(
            [
                {
                    "term": _label_for(term),
                    "meaning": entry["meaning"],
                    "why it matters": entry["matters"],
                    "suspicious behavior may suggest": entry["suspicious"],
                }
                for term, entry in _GLOSSARY.items()
            ]
        )
        st.dataframe(glossary_df, use_container_width=True, hide_index=True)


def _is_streamlit_community_cloud() -> bool:
    """Best-effort detection for Streamlit Community Cloud deployments."""
    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_CLOUD"):
        return True
    return str(Path.cwd()).startswith("/mount/src/")


def _local_llm_enabled() -> bool:
    flag = os.environ.get(_LOCAL_LLM_FLAG, "auto").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return not _is_streamlit_community_cloud()


def _render_llm_explanation_panel(
    payload,
    deterministic_text: str,
    mode: str,
) -> None:
    st.markdown("#### Explanation Mode")
    st.caption(
        "LLM mode is optional and local. It explains structured rule, ML, and attribution outputs; "
        "it does not replace anomaly detection."
    )

    llm_enabled = _local_llm_enabled()
    explainer = (
        _load_llm_explainer(str(LLM_ADAPTER_PATH))
        if llm_enabled and LLM_ADAPTER_PATH.exists() and mode != DETERMINISTIC_MODE
        else None
    )
    result = resolve_explanation_mode(
        mode=mode,
        payload=payload,
        deterministic_text=deterministic_text,
        adapter_exists=llm_enabled and LLM_ADAPTER_PATH.exists(),
        llm_explainer=explainer,
    )
    if mode != DETERMINISTIC_MODE and not llm_enabled:
        st.warning(
            "Local LLM loading is disabled for this deployment. Showing deterministic explanation instead. "
            f"Set `{_LOCAL_LLM_FLAG}=1` only on a host with enough memory and local model assets.",
            icon="⚠️",
        )
    elif result.warning:
        st.warning(result.warning, icon="⚠️")

    if result.show_side_by_side and result.llm_text:
        left, right = st.columns(2)
        with left:
            st.markdown("**Deterministic**")
            st.info(result.deterministic_text, icon="ℹ️")
        with right:
            st.markdown("**Fine-tuned LLM**")
            st.success(result.llm_text, icon="🤖")
    elif result.llm_text:
        st.success(result.llm_text, icon="🤖")
    else:
        st.info(result.deterministic_text, icon="ℹ️")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛰️ Orbital Refueling Telemetry Simulator")
    st.caption("Synthetic orbital transfer monitor")
    st.markdown("---")

    scenario = st.radio(
        "Anomaly scenario",
        ANOMALY_SCENARIOS,
        format_func=lambda s: s.replace("_", " ").title(),
    )
    st.info(SCENARIO_DESCRIPTIONS[scenario], icon="ℹ️")
    beginner_mode = st.toggle(
        "Beginner Mode",
        value=True,
        help="Show extra plain-English guidance, captions, and glossary details.",
    )
    show_estimator = st.toggle(
        "Show estimator residuals",
        value=False,
        help="Optional lightweight state-estimation chart for observed vs estimated line pressure.",
    )
    explanation_mode = st.radio(
        "Explanation mode",
        [
            DETERMINISTIC_MODE,
            LLM_MODE,
            SIDE_BY_SIDE_MODE,
        ],
        help=(
            "Optional local LLM explanation layer. Rules and ML scores remain the source of truth; "
            "the LLM only rewrites structured simulator outputs."
        ),
    )
    st.markdown("---")

    st.markdown("**Mission phases**")
    for phase, color in PHASE_COLORS.items():
        dur = PHASE_DURATIONS[phase]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
            f'<div style="width:12px;height:12px;background:{color};'
            f'border-radius:2px;flex-shrink:0"></div>'
            f'<span style="font-size:0.82rem">'
            f'{phase.replace("_", " ").title()}'
            f' <span style="color:#666">({dur}s)</span>'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption(
        "Synthetic prototype. All telemetry is procedurally generated. "
        "Not based on real spacecraft data or operational software."
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("## 🛰️ Orbital Refueling Telemetry Simulator")
st.markdown(
    "**Hybrid safety architecture:** "
    "**Deterministic rules** enforce hard engineering limits (fixed thresholds, always authoritative). "
    "**ML early warning** (IsolationForest) surfaces subtle multivariate drift before any single threshold is breached — advisory only. "
    "Select a scenario in the sidebar to explore."
)
st.caption(
    "⚠️ Synthetic prototype — all telemetry is procedurally generated. "
    "Not flight software, not an autonomous abort system, and not based on real spacecraft data or any specific mission."
)

guide_expanded = beginner_mode
with st.expander("How to read this dashboard", expanded=guide_expanded):
    st.markdown(
        "- **Deterministic rules** are explicit engineering checks, such as pressure or voltage crossing a fixed limit.\n"
        "- **ML anomaly score** is advisory early warning: it asks whether the combined telemetry pattern looks unlike nominal synthetic runs.\n"
        "- **Grouped alert events** merge repeated timestamp-level rule alerts into readable event windows.\n"
        "- **Phase awareness matters** because normal behavior changes across approach, docking, transfer, leak check, and retreat.\n"
        "- **Synthetic telemetry only:** this is not flight software, not an autonomous abort system, and not real spacecraft diagnosis."
    )
st.markdown("---")

# ── Run ───────────────────────────────────────────────────────────────────────

with st.spinner(f"Running **{scenario}** scenario…"):
    df, scores, alerts_df, grouped_df, contribs, score_summary, estimates = _run_scenario(scenario)

# ── Metrics ───────────────────────────────────────────────────────────────────

mass = float(df["total_mass_transferred"].max())
anomaly_rate = float((scores > 0.45).mean() * 100)
mean_s = float(scores.mean())
mean_unc = float(score_summary["ml_score_uncertainty"].mean())
n_crit = int((grouped_df["severity"] == "CRITICAL").sum()) if len(grouped_df) else 0
n_warn = int((grouped_df["severity"] == "WARNING").sum()) if len(grouped_df) else 0

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Mass Transferred", f"{mass:.1f} kg")
m2.metric("ML Early-Warning Rate", f"{anomaly_rate:.1f}%",
          help="% of timesteps where ML score exceeds 0.45 alert threshold")
m3.metric("Mean ML Score", f"{mean_s * 100:.1f}/100",
          help="Average advisory anomaly score across full mission (0 = nominal-like, 100 = strongly unusual)")
m4.metric("Critical Rule Events", str(n_crit), help="Grouped deterministic rule events with CRITICAL severity")
m5.metric("Warning Rule Events", str(n_warn), help="Grouped deterministic rule events with WARNING severity")
m6.metric("Mean ML Uncertainty", f"{mean_unc * 100:.1f}/100",
          help="Average spread across nominal-seed detector ensemble. Higher means calibration-sensitive score.")

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Mission Telemetry",
    "ML + Rule Alerts",
    "Why It Flagged",
    "Learn the Signals",
])

# ── Tab 1: Telemetry ──────────────────────────────────────────────────────────

with tab1:
    st.caption(
        "Shaded bands mark refueling phases. Hover any chart for the timestamp, phase, and synthetic value."
    )

    st.plotly_chart(_phase_timeline_plot(df), use_container_width=True)
    _what_to_look_for(
        "phase context explains why the same value can be normal at one point and suspicious later.",
        beginner_mode,
    )
    _render_chart_help("mission_phase", beginner_mode)

    st.plotly_chart(
        _signal_plot(
            df,
            [("end_effector_position_error", "Position error (mm)"),
             ("arm_joint_angle", "Joint angle (°)"),
             ("arm_joint_velocity", "Joint velocity (°/s)"),
             ("arm_motor_current", "Arm motor (A)"),
             ("interface_force", "Force (N)"),
             ("interface_torque", "Torque (Nm)")],
            "Robotic Arm Alignment and Contact",
            "synthetic/demo units",
        ),
        use_container_width=True,
    )
    _what_to_look_for(
        "alignment error should fall as the arm docks; force, torque, or motor-current spikes may suggest rough contact or misalignment.",
        beginner_mode,
    )
    _render_signal_help([
        "end_effector_position_error", "arm_joint_angle", "arm_joint_velocity",
        "arm_motor_current", "interface_force", "interface_torque",
    ], beginner_mode)
    _render_chart_help("robotic_arm", beginner_mode)

    st.plotly_chart(
        _signal_plot(
            df,
            [("flow_rate", "Flow (kg/s)"),
             ("line_pressure", "Line pressure (bar)"),
             ("donor_tank_pressure", "Donor tank (bar)"),
             ("receiver_tank_pressure", "Receiver tank (bar)"),
             ("total_mass_transferred", "Transferred mass (kg)")],
            "Fuel Transfer Health",
            "synthetic/demo units",
        ),
        use_container_width=True,
    )
    _what_to_look_for(
        "during main transfer, flow and received mass should rise smoothly; low flow with high line pressure may suggest restriction.",
        beginner_mode,
    )
    _render_signal_help([
        "flow_rate", "line_pressure", "donor_tank_pressure",
        "receiver_tank_pressure", "total_mass_transferred",
    ], beginner_mode)
    _render_chart_help("fuel_transfer", beginner_mode)

    st.plotly_chart(
        _signal_plot(
            df,
            [("seal_pressure", "Seal pressure (bar)"),
             ("line_pressure", "Line pressure (bar)"),
             ("receiver_tank_pressure", "Receiver tank (bar)")],
            "Seal and Leak Integrity",
            "bar",
        ),
        use_container_width=True,
    )
    _what_to_look_for(
        "after the seal is established, falling or unstable seal pressure may suggest leakage or poor seal integrity.",
        beginner_mode,
    )
    _render_signal_help(["seal_pressure", "line_pressure", "receiver_tank_pressure"], beginner_mode)
    _render_chart_help("seal_leak", beginner_mode)

    st.plotly_chart(
        _signal_plot(
            df,
            [("pump_current", "Pump current (A)"),
             ("propellant_temperature", "Propellant (deg C)"),
             ("line_temperature", "Line (deg C)"),
             ("bus_voltage", "Bus voltage (V)")],
            "Thermal and Power Health",
            "synthetic/demo units",
        ),
        use_container_width=True,
    )
    _what_to_look_for(
        "temperature or voltage changes may suggest thermal stress, pump stress, or power-system issues.",
        beginner_mode,
    )
    _render_signal_help(["pump_current", "propellant_temperature", "line_temperature", "bus_voltage"], beginner_mode)
    _render_chart_help("thermal_power", beginner_mode)

    if show_estimator:
        st.plotly_chart(_estimator_plot(estimates), use_container_width=True)
        _what_to_look_for(
            "large residuals mean observed line pressure diverged from the simple one-signal state estimate.",
            beginner_mode,
        )

    st.plotly_chart(
        _signal_plot(
            df,
            [("attitude_error", "Attitude error (°)"),
             ("reaction_wheel_speed", "Wheel speed (RPM)")],
            "Vehicle Stability During Refueling",
            "synthetic/demo units",
        ),
        use_container_width=True,
    )
    _what_to_look_for(
        "attitude error and reaction-wheel speed rising together may suggest vehicle stability stress.",
        beginner_mode,
    )
    _render_signal_help(["attitude_error", "reaction_wheel_speed"], beginner_mode)
    _render_chart_help("vehicle_stability", beginner_mode)

# ── Tab 2: Anomaly Detection ──────────────────────────────────────────────────

with tab2:
    st.markdown(
        "**ML early warning** (blue curve): advisory score reflecting multivariate deviation from nominal. "
        "Trained on nominal data only; higher score = less nominal-like. "
        "Scores above 45/100 warrant attention but are not automatic actions.  \n"
        "**Deterministic rule violations** (▼ markers): engineering safety checks with fixed thresholds. "
        "These represent hard safety/engineering checks; ML never suppresses or overrides them."
    )
    if beginner_mode:
        st.info(
            "Read this tab from top to bottom: first check whether the ML score rises, then check whether any deterministic rule events were grouped below.",
            icon="ℹ️",
        )
    st.plotly_chart(_score_plot(df, scores, score_summary, alerts_df), use_container_width=True)
    _what_to_look_for(
        "blue score rising above 45 suggests the combined signal pattern is less like nominal synthetic telemetry; triangle markers show rule threshold crossings.",
        beginner_mode,
    )
    _render_chart_help("ml_score", beginner_mode)

    st.markdown("### Grouped Rule Alert Events")
    st.caption(
        "Default view shows grouped event windows so sustained issues appear as a few readable events. Raw timestamp-level rows stay collapsed below."
    )
    if len(grouped_df) == 0:
        st.success("No rule violations detected for this scenario.", icon="✅")
    else:
        for sev, icon in [("CRITICAL", "🔴"), ("WARNING", "🟡")]:
            sub = grouped_df[grouped_df["severity"] == sev].reset_index(drop=True)
            if len(sub) == 0:
                continue
            with st.expander(
                f"{icon} {sev} — {len(sub)} event(s)",
                expanded=(sev == "CRITICAL"),
            ):
                display_cols = [
                    "title", "phase", "start_elapsed_s", "end_elapsed_s",
                    "duration_s", "peak_value", "affected_signals",
                    "recommended_action", "number_of_points",
                ]
                st.dataframe(
                    sub[display_cols].rename(columns={
                        "start_elapsed_s": "start (s)",
                        "end_elapsed_s":   "end (s)",
                        "duration_s":      "duration (s)",
                        "peak_value":      "peak value",
                        "affected_signals":"signal",
                        "number_of_points":"# samples",
                        "recommended_action": "action",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        # Raw timestamp-level detail (collapsed by default)
        with st.expander("Raw timestamp-level alerts (collapsed detail)", expanded=False):
            if len(alerts_df) == 0:
                st.write("None.")
            else:
                st.dataframe(
                    alerts_df[["time", "phase", "severity", "rule", "signal", "value"]],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"{len(alerts_df)} raw alert rows → grouped into "
                    f"{len(grouped_df)} event windows (gap threshold: 15 s)"
                )

# ── Tab 3: Explanation ────────────────────────────────────────────────────────

with tab3:
    st.markdown(
        "**Lightweight attribution cue:** each signal in high-score rows is individually "
        "replaced with its nominal phase mean. If the score drops, that signal likely contributed "
        "to the ML score. This is not causal proof."
    )

    col_chart, col_info = st.columns([3, 2])

    with col_chart:
        st.plotly_chart(_contribution_plot(contribs), use_container_width=True)
        _what_to_look_for(
            "longer bars suggest signals that contributed more to the high ML score in this scenario.",
            beginner_mode,
        )
        _render_chart_help("attribution", beginner_mode)

    with col_info:
        st.markdown("#### Explanation Summary")
        explanation_payload = build_explanation_payload(
            scenario=scenario,
            df=df,
            scores=scores,
            grouped_alerts=grouped_df,
            contributions=contribs,
            nominal_seed=42,
        )
        deterministic_text = build_deterministic_explanation(explanation_payload)
        _render_llm_explanation_panel(
            explanation_payload,
            deterministic_text,
            explanation_mode,
        )
        st.markdown("---")
        if not contribs:
            st.info(
                "No rows exceeded the anomaly threshold (0.45). "
                "Select an anomaly scenario from the sidebar.",
                icon="ℹ️",
            )
        else:
            top_feat = list(contribs.keys())[0]
            top_val = list(contribs.values())[0]
            top_phase = _top_phase(df, scores).replace("_", " ").title()
            interpretation = _SCENARIO_INTERPRETATION[scenario]
            top_help = _GLOSSARY.get(top_feat, {})

            st.markdown("**1. What the system noticed**")
            st.write(
                f"The ML score was most associated with **{_label_for(top_feat)}** "
                f"(attribution cue `{top_val:+.3f}`). {interpretation['noticed']}"
            )

            st.markdown("**2. Why this is unusual for the current phase**")
            st.write(
                f"Most high-score rows occur around **{top_phase}**. "
                f"During each phase, the detector compares telemetry against that phase's nominal synthetic pattern. "
                f"For this signal: {top_help.get('matters', 'the value is compared with phase-specific nominal behavior')}"
            )

            st.markdown("**3. What it may indicate**")
            st.write(
                f"This pattern is consistent with **{scenario.replace('_', ' ')}**. "
                f"It may indicate: {interpretation['may_indicate']} "
                "It does not confirm a physical root cause."
            )

            st.markdown("**4. What an operator might check next**")
            st.write(interpretation["check"])

            st.markdown("---")
            st.markdown("**Top contributing signals**")
            for feat in list(contribs.keys())[:6]:
                desc = _SIGNAL_DESC.get(feat, "")
                val = contribs[feat]
                badge = "High" if val > 0.06 else "Medium" if val > 0.01 else "Low"
                st.markdown(f"- **{_label_for(feat)}** (`{badge}`, `{val:+.3f}`): {desc}")

        st.markdown("---")
        st.markdown(
            "**Why ML + rules?**  \n"
            "Rules cover hard-limit breaches with deterministic checks. "
            "ML may surface patterns that are individually within limits but collectively "
            "unusual. ML provides advisory early warning; rules provide hard safety/engineering checks."
        )

with tab4:
    _render_glossary(beginner_mode)
    if beginner_mode:
        st.markdown("---")
        st.markdown("### Quick Mental Model")
        st.markdown(
            "- **Rules:** explicit threshold checks. Useful for hard limits.\n"
            "- **ML score:** advisory pattern check. Useful when several signals look subtly unusual together.\n"
            "- **Phase:** the current refueling step. It changes what normal looks like.\n"
            "- **Grouped alerts:** readable summaries of repeated rule crossings."
        )
