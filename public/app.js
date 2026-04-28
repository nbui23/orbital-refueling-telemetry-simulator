const data = window.TELEMETRY_SITE_DATA;

const signalLabels = {
  flow_rate: "Flow rate",
  line_pressure: "Line pressure",
  seal_pressure: "Seal pressure",
  pump_current: "Pump current",
  end_effector_position_error: "Arm position error",
  interface_force: "Interface force",
  attitude_error: "Attitude error",
  reaction_wheel_speed: "Reaction wheel speed",
  bus_voltage: "Bus voltage",
};

const signalUnits = {
  flow_rate: "kg/s",
  line_pressure: "bar",
  seal_pressure: "bar",
  pump_current: "A",
  end_effector_position_error: "mm",
  interface_force: "N",
  attitude_error: "deg",
  reaction_wheel_speed: "RPM",
  bus_voltage: "V",
};

const chartColors = {
  ml_score: "#f2b84b",
  flow_rate: "#42c2a8",
  line_pressure: "#d65f5f",
  seal_pressure: "#7aa7ff",
  pump_current: "#c78cff",
  end_effector_position_error: "#ff8a5b",
  interface_force: "#65d0ff",
  attitude_error: "#e7df76",
  reaction_wheel_speed: "#93d36f",
  bus_voltage: "#b8c4d9",
};

const phaseColors = {
  approach: "#315f93",
  arm_alignment: "#8a6234",
  docking: "#2c7367",
  seal_check: "#8d7931",
  pressure_equalization: "#69519a",
  main_transfer: "#923743",
  leak_check: "#9a5b25",
  disconnect: "#66707f",
  retreat: "#4c7d78",
};

const scenarioSelect = document.querySelector("#scenario");
const signalSelect = document.querySelector("#signal");
const alertTable = document.querySelector("#alert-table tbody");
const phaseRail = document.querySelector("#phase-rail");

for (const scenario of Object.keys(data.scenarios)) {
  const option = document.createElement("option");
  option.value = scenario;
  option.textContent = scenario.replaceAll("_", " ");
  scenarioSelect.append(option);
}

for (const signal of data.signals) {
  const option = document.createElement("option");
  option.value = signal;
  option.textContent = `${signalLabels[signal]} (${signalUnits[signal]})`;
  signalSelect.append(option);
}

scenarioSelect.value = "partial_blockage";
signalSelect.value = "line_pressure";

function fmt(value) {
  return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : value;
}

function statusClass(severity) {
  if (severity === "CRITICAL") return "critical";
  if (severity === "WARNING") return "warning";
  return "none";
}

function setMetric(id, value, detail) {
  document.querySelector(`#${id} .metric-value`).textContent = value;
  document.querySelector(`#${id} .metric-detail`).textContent = detail;
}

function drawChart(canvas, rows, series, options = {}) {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const pad = { left: 48, right: 18, top: 18, bottom: 34 };
  const width = rect.width - pad.left - pad.right;
  const height = rect.height - pad.top - pad.bottom;
  const times = rows.map((row) => row.time);
  const maxTime = Math.max(...times);
  const minTime = Math.min(...times);
  const values = rows.flatMap((row) => series.map((s) => row[s.key]));
  let minY = options.minY ?? Math.min(...values);
  let maxY = options.maxY ?? Math.max(...values);
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }

  ctx.fillStyle = "#101827";
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "#263244";
  ctx.lineWidth = 1;
  ctx.font = "12px system-ui, -apple-system, Segoe UI, sans-serif";
  ctx.fillStyle = "#9fb0c8";

  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (height * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + width, y);
    ctx.stroke();
    const value = maxY - ((maxY - minY) * i) / 4;
    ctx.fillText(fmt(value), 8, y + 4);
  }

  for (let i = 0; i <= 4; i += 1) {
    const x = pad.left + (width * i) / 4;
    const value = minTime + ((maxTime - minTime) * i) / 4;
    ctx.fillText(`${fmt(value)}s`, x - 16, rect.height - 10);
  }

  const xFor = (time) => pad.left + ((time - minTime) / (maxTime - minTime)) * width;
  const yFor = (value) => pad.top + ((maxY - value) / (maxY - minY)) * height;

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((row, idx) => {
      const x = xFor(row.time);
      const y = yFor(row[s.key]);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  if (options.threshold !== undefined) {
    const y = yFor(options.threshold);
    ctx.strokeStyle = "#f05f6b";
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + width, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function renderPhases(rows) {
  phaseRail.innerHTML = "";
  const total = rows.at(-1).time - rows[0].time;
  let start = 0;
  while (start < rows.length) {
    let end = start;
    while (end + 1 < rows.length && rows[end + 1].phase === rows[start].phase) end += 1;
    const segment = document.createElement("div");
    segment.className = "phase-segment";
    segment.style.background = phaseColors[rows[start].phase] || "#536070";
    segment.style.width = `${((rows[end].time - rows[start].time + 0.5) / total) * 100}%`;
    segment.title = rows[start].phase.replaceAll("_", " ");
    segment.textContent = rows[start].phase.replaceAll("_", " ");
    phaseRail.append(segment);
    start = end + 1;
  }
}

function renderAlerts(alerts) {
  alertTable.innerHTML = "";
  if (alerts.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="5" class="muted">No grouped deterministic rule alerts for this scenario.</td>';
    alertTable.append(row);
    return;
  }

  for (const alert of alerts) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><span class="pill ${statusClass(alert.severity)}">${alert.severity}</span></td>
      <td>${alert.title}</td>
      <td>${alert.phase.replaceAll("_", " ")}</td>
      <td>${fmt(alert.start_elapsed_s)}-${fmt(alert.end_elapsed_s)}s</td>
      <td>${alert.recommended_action}</td>
    `;
    alertTable.append(row);
  }
}

function render() {
  const scenario = data.scenarios[scenarioSelect.value];
  const rows = scenario.rows;
  const signal = signalSelect.value;

  document.querySelector("#scenario-description").textContent = scenario.description;
  setMetric("max-score", fmt(scenario.summary.max_ml_score), `peak at ${fmt(scenario.summary.top_time)}s during ${scenario.summary.top_phase.replaceAll("_", " ")}`);
  setMetric("mean-score", fmt(scenario.summary.mean_ml_score), "average advisory ML score");
  setMetric("alert-count", scenario.summary.alert_count, "grouped deterministic events");
  setMetric("highest-severity", scenario.summary.highest_severity, "highest rule severity");
  document.querySelector("#highest-severity .metric-value").className = `metric-value ${statusClass(scenario.summary.highest_severity)}`;

  renderPhases(rows);
  renderAlerts(scenario.alerts);
  drawChart(document.querySelector("#ml-chart"), rows, [{ key: "ml_score", color: chartColors.ml_score }], { minY: 0, maxY: 100, threshold: 45 });
  drawChart(document.querySelector("#signal-chart"), rows, [{ key: signal, color: chartColors[signal] }]);
  document.querySelector("#signal-title").textContent = `${signalLabels[signal]} (${signalUnits[signal]})`;
}

scenarioSelect.addEventListener("change", render);
signalSelect.addEventListener("change", render);
window.addEventListener("resize", render);
render();
