"""HTML template for the llmwatch web dashboard."""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llmwatch Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #f8f9fa;
    color: #1e293b;
    min-height: 100vh;
  }

  /* * Top bar */
  .topbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 24px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .topbar-brand {
    font-size: 18px;
    font-weight: 700;
    color: #2563eb;
    letter-spacing: -0.5px;
  }

  .topbar-brand span {
    color: #64748b;
    font-weight: 400;
  }

  .topbar-controls {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .topbar-controls label {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
  }

  select {
    appearance: none;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 32px 6px 10px;
    font-size: 13px;
    color: #1e293b;
    cursor: pointer;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2364748b' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
  }

  select:focus {
    outline: none;
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
  }

  /* * Main layout */
  .main {
    max-width: 1280px;
    margin: 0 auto;
    padding: 24px 24px 48px;
  }

  /* * Status badge */
  .status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    font-size: 12px;
    color: #64748b;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
  }

  .status-dot.loading {
    background: #f59e0b;
    animation: pulse 1.2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* * Summary cards */
  .cards-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  @media (max-width: 900px) {
    .cards-row { grid-template-columns: repeat(2, 1fr); }
  }

  @media (max-width: 520px) {
    .cards-row { grid-template-columns: 1fr; }
  }

  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  }

  .card-label {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 8px;
  }

  .card-value {
    font-size: 26px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
  }

  .card-value.cost { color: #2563eb; }

  .card-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
  }

  /* * Charts row */
  .charts-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }

  @media (max-width: 720px) {
    .charts-row { grid-template-columns: 1fr; }
  }

  .chart-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  }

  .chart-title {
    font-size: 13px;
    font-weight: 600;
    color: #475569;
    margin-bottom: 16px;
  }

  .chart-container {
    position: relative;
    height: 240px;
  }

  /* * Tabs */
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .section-title {
    font-size: 14px;
    font-weight: 600;
    color: #1e293b;
  }

  .tabs {
    display: flex;
    gap: 4px;
    background: #f1f5f9;
    padding: 4px;
    border-radius: 8px;
  }

  .tab-btn {
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #64748b;
    background: transparent;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tab-btn:hover { color: #1e293b; }

  .tab-btn.active {
    background: #ffffff;
    color: #2563eb;
    font-weight: 600;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
  }

  /* * Tables */
  .table-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-bottom: 24px;
  }

  .table-card-header {
    padding: 16px 20px;
    border-bottom: 1px solid #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .table-scroll {
    overflow-x: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  thead th {
    background: #f8fafc;
    padding: 10px 16px;
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid #e2e8f0;
    position: sticky;
    top: 0;
    white-space: nowrap;
  }

  thead th.num { text-align: right; }

  tbody tr {
    border-bottom: 1px solid #f1f5f9;
    transition: background 0.1s;
  }

  tbody tr:last-child { border-bottom: none; }

  tbody tr:hover { background: #f8fafc; }

  tbody tr:nth-child(even) { background: #fafbfc; }
  tbody tr:nth-child(even):hover { background: #f1f5f9; }

  td {
    padding: 10px 16px;
    color: #334155;
    vertical-align: middle;
  }

  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.cost { text-align: right; color: #2563eb; font-weight: 600; font-variant-numeric: tabular-nums; }
  td.model { font-family: "SF Mono", "Fira Code", Consolas, monospace; font-size: 12px; }
  td.mono { font-family: "SF Mono", "Fira Code", Consolas, monospace; font-size: 11px; color: #64748b; }
  td.ts { font-size: 12px; color: #94a3b8; white-space: nowrap; }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
  }

  .badge-blue { background: #dbeafe; color: #1d4ed8; }
  .badge-purple { background: #ede9fe; color: #6d28d9; }
  .badge-green { background: #dcfce7; color: #15803d; }
  .badge-orange { background: #ffedd5; color: #c2410c; }

  /* * Empty state */
  .empty-state {
    padding: 48px 24px;
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
  }

  /* * Error banner */
  .error-banner {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #b91c1c;
    margin-bottom: 16px;
    display: none;
  }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-brand">llmwatch <span>/ dashboard</span></div>
  <div class="topbar-controls">
    <label for="period-select">Period</label>
    <select id="period-select">
      <option value="24h">Last 24 hours</option>
      <option value="7d">Last 7 days</option>
      <option value="30d" selected>Last 30 days</option>
      <option value="90d">Last 90 days</option>
    </select>
  </div>
</div>

<div class="main">
  <div class="status-bar">
    <div class="status-dot loading" id="status-dot"></div>
    <span id="status-text">Loading...</span>
  </div>

  <div class="error-banner" id="error-banner"></div>

  <!-- Summary cards -->
  <div class="cards-row">
    <div class="card">
      <div class="card-label">Total Cost</div>
      <div class="card-value cost" id="card-cost">-</div>
      <div class="card-sub" id="card-cost-sub">&nbsp;</div>
    </div>
    <div class="card">
      <div class="card-label">Total Requests</div>
      <div class="card-value" id="card-requests">-</div>
      <div class="card-sub" id="card-requests-sub">&nbsp;</div>
    </div>
    <div class="card">
      <div class="card-label">Top Model</div>
      <div class="card-value" style="font-size:16px;word-break:break-all;" id="card-top-model">-</div>
      <div class="card-sub" id="card-top-model-sub">&nbsp;</div>
    </div>
    <div class="card">
      <div class="card-label">Top Feature</div>
      <div class="card-value" style="font-size:16px;word-break:break-all;" id="card-top-feature">-</div>
      <div class="card-sub" id="card-top-feature-sub">&nbsp;</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-row">
    <div class="chart-card">
      <div class="chart-title" id="donut-title">Cost by Feature</div>
      <div class="chart-container">
        <canvas id="chart-donut"></canvas>
      </div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Cost by Model</div>
      <div class="chart-container">
        <canvas id="chart-bar"></canvas>
      </div>
    </div>
  </div>

  <!-- Breakdown table with group-by tabs -->
  <div class="table-card">
    <div class="table-card-header">
      <div class="section-title" id="breakdown-title">Breakdown</div>
      <div class="tabs" id="group-tabs">
        <button class="tab-btn active" data-group="feature">Feature</button>
        <button class="tab-btn" data-group="user_id">User</button>
        <button class="tab-btn" data-group="model">Model</button>
        <button class="tab-btn" data-group="provider">Provider</button>
        <button class="tab-btn" data-group="environment">Environment</button>
      </div>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th id="breakdown-col-header">Feature</th>
            <th class="num">Requests</th>
            <th class="num">Prompt Tokens</th>
            <th class="num">Completion Tokens</th>
            <th class="num">Cost (USD)</th>
          </tr>
        </thead>
        <tbody id="breakdown-tbody">
          <tr><td colspan="5" class="empty-state">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Recent records -->
  <div class="table-card">
    <div class="table-card-header">
      <div class="section-title">Recent Records</div>
      <span style="font-size:12px;color:#94a3b8;">Last 20 calls</span>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Model</th>
            <th>Provider</th>
            <th>Feature</th>
            <th>User</th>
            <th class="num">Prompt</th>
            <th class="num">Completion</th>
            <th class="num">Latency (ms)</th>
            <th class="num">Cost (USD)</th>
          </tr>
        </thead>
        <tbody id="records-tbody">
          <tr><td colspan="9" class="empty-state">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
// * Color palette (8 colors)
const PALETTE = [
  "#2563eb", "#7c3aed", "#0891b2", "#059669",
  "#d97706", "#dc2626", "#db2777", "#65a30d"
];

// * State
let currentPeriod = "30d";
let currentGroup = "feature";
let donutChart = null;
let barChart = null;

// * Utility functions
function esc(s) {
  if (s == null) return "";
  const el = document.createElement("span");
  el.textContent = String(s);
  return el.innerHTML;
}

function fmtCost(v) {
  if (v == null) return "-";
  return "$" + Number(v).toFixed(4);
}

function fmtNum(v) {
  if (v == null) return "-";
  return Number(v).toLocaleString();
}

function fmtTs(iso) {
  if (!iso) return "-";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit"
    });
  } catch { return iso; }
}

function truncate(s, n) {
  if (!s) return "-";
  return s.length > n ? s.slice(0, n) + "..." : s;
}

function providerBadge(provider) {
  const cls = {
    openai: "badge-green",
    anthropic: "badge-purple",
    google: "badge-blue",
  }[provider?.toLowerCase()] || "badge-orange";
  return `<span class="badge ${cls}">${esc(provider) || "-"}</span>`;
}

// * Status helpers
function setStatus(loading, text) {
  document.getElementById("status-dot").className = "status-dot" + (loading ? " loading" : "");
  document.getElementById("status-text").textContent = text;
}

function showError(msg) {
  const el = document.getElementById("error-banner");
  el.textContent = msg;
  el.style.display = "block";
}

function hideError() {
  document.getElementById("error-banner").style.display = "none";
}

// * Fetch helpers
async function apiFetch(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
}

// * Chart helpers
function destroyChart(instance) {
  if (instance) { try { instance.destroy(); } catch (_) {} }
}

function buildDonut(labels, values, title) {
  destroyChart(donutChart);
  const ctx = document.getElementById("chart-donut").getContext("2d");
  donutChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: PALETTE,
        borderWidth: 2,
        borderColor: "#ffffff",
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "right",
          labels: { font: { size: 11 }, padding: 12, boxWidth: 12, usePointStyle: true },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${fmtCost(ctx.parsed)}`,
          },
        },
      },
      cutout: "62%",
    },
  });
  document.getElementById("donut-title").textContent = title;
}

function buildBar(labels, values) {
  destroyChart(barChart);
  const ctx = document.getElementById("chart-bar").getContext("2d");
  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: PALETTE,
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${fmtCost(ctx.parsed.x)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: "#f1f5f9" },
          ticks: {
            font: { size: 11 },
            callback: v => "$" + Number(v).toFixed(4),
          },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } },
        },
      },
    },
  });
}

// * Render breakdown table
function renderBreakdown(breakdowns, groupBy) {
  const tbody = document.getElementById("breakdown-tbody");
  document.getElementById("breakdown-col-header").textContent =
    { feature: "Feature", user_id: "User ID", model: "Model", provider: "Provider" }[groupBy] || groupBy;
  document.getElementById("breakdown-title").textContent =
    "Breakdown by " + (groupBy === "user_id" ? "User" : groupBy.charAt(0).toUpperCase() + groupBy.slice(1));

  if (!breakdowns || breakdowns.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No data for this period.</td></tr>';
    return;
  }

  tbody.innerHTML = breakdowns.map(b => `
    <tr>
      <td>${esc(b.group_value) || "-"}</td>
      <td class="num">${fmtNum(b.total_requests)}</td>
      <td class="num">${fmtNum(b.total_prompt_tokens)}</td>
      <td class="num">${fmtNum(b.total_completion_tokens)}</td>
      <td class="cost">${fmtCost(b.total_cost_usd)}</td>
    </tr>
  `).join("");
}

// * Render records table
function renderRecords(records) {
  const tbody = document.getElementById("records-tbody");
  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state">No recent records.</td></tr>';
    return;
  }

  tbody.innerHTML = records.map(r => `
    <tr>
      <td class="ts">${esc(fmtTs(r.timestamp))}</td>
      <td class="model">${esc(truncate(r.model, 28))}</td>
      <td>${providerBadge(r.provider)}</td>
      <td>${esc(r.tags?.feature) || "<span style='color:#cbd5e1'>-</span>"}</td>
      <td>${esc(r.tags?.user_id) || "<span style='color:#cbd5e1'>-</span>"}</td>
      <td class="num">${fmtNum(r.token_usage?.prompt_tokens)}</td>
      <td class="num">${fmtNum(r.token_usage?.completion_tokens)}</td>
      <td class="num">${r.latency_ms != null ? fmtNum(Math.round(r.latency_ms)) : "-"}</td>
      <td class="cost">${fmtCost(r.cost_usd)}</td>
    </tr>
  `).join("");
}

// * Update overview cards
function renderOverview(data) {
  document.getElementById("card-cost").textContent = fmtCost(data.total_cost_usd);
  document.getElementById("card-cost-sub").textContent =
    `${fmtNum(data.total_prompt_tokens + data.total_completion_tokens)} total tokens`;
  document.getElementById("card-requests").textContent = fmtNum(data.total_requests);
  document.getElementById("card-requests-sub").textContent =
    data.total_prompt_tokens != null ? `${fmtNum(data.total_prompt_tokens)} prompt` : "";
  document.getElementById("card-top-model").textContent = data.top_model || "-";  // textContent is safe
  document.getElementById("card-top-model-sub").textContent =
    data.top_model_cost != null ? fmtCost(data.top_model_cost) : "";
  document.getElementById("card-top-feature").textContent = data.top_feature || "-";  // textContent is safe
  document.getElementById("card-top-feature-sub").textContent =
    data.top_feature_cost != null ? fmtCost(data.top_feature_cost) : "";
}

// * Main data load
async function loadAll() {
  setStatus(true, "Fetching data...");
  hideError();

  const period = currentPeriod;
  const group = currentGroup;

  try {
    const [overview, summary, modelSummary, records] = await Promise.all([
      apiFetch(`/api/overview?period=${period}`),
      apiFetch(`/api/summary?group_by=${group}&period=${period}`),
      apiFetch(`/api/summary?group_by=model&period=${period}`),
      apiFetch(`/api/records?limit=20`),
    ]);

    renderOverview(overview);

    // Donut chart — current group_by
    const donutLabels = summary.breakdowns.map(b => truncate(b.group_value, 20));
    const donutValues = summary.breakdowns.map(b => b.total_cost_usd);
    const donutTitle = "Cost by " +
      (group === "user_id" ? "User" : group.charAt(0).toUpperCase() + group.slice(1));
    buildDonut(donutLabels, donutValues, donutTitle);

    // Bar chart — by model (top 8)
    const top8 = modelSummary.breakdowns.slice(0, 8);
    buildBar(top8.map(b => truncate(b.group_value, 24)), top8.map(b => b.total_cost_usd));

    renderBreakdown(summary.breakdowns, group);
    renderRecords(records);

    const now = new Date().toLocaleTimeString();
    setStatus(false, `Last updated: ${now}`);
  } catch (err) {
    setStatus(false, "Error loading data");
    showError("Failed to load data: " + err.message);
    console.error(err);
  }
}

// * Load only breakdown + donut (tab switch)
async function loadGroupBy() {
  const group = currentGroup;
  const period = currentPeriod;

  try {
    const summary = await apiFetch(`/api/summary?group_by=${group}&period=${period}`);
    const donutLabels = summary.breakdowns.map(b => truncate(b.group_value, 20));
    const donutValues = summary.breakdowns.map(b => b.total_cost_usd);
    const donutTitle = "Cost by " +
      (group === "user_id" ? "User" : group.charAt(0).toUpperCase() + group.slice(1));
    buildDonut(donutLabels, donutValues, donutTitle);
    renderBreakdown(summary.breakdowns, group);
  } catch (err) {
    showError("Failed to load breakdown: " + err.message);
  }
}

// * Event listeners
document.getElementById("period-select").addEventListener("change", e => {
  currentPeriod = e.target.value;
  loadAll();
});

document.getElementById("group-tabs").addEventListener("click", e => {
  const btn = e.target.closest(".tab-btn");
  if (!btn) return;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentGroup = btn.dataset.group;
  loadGroupBy();
});

// * Initial load
loadAll();
</script>
</body>
</html>
"""
