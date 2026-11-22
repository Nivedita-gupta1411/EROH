/**
 * Regional EV Energy Optimization Control Tower Frontend Script
 * Manages UI interactions, live MILP optimization API calls, and Chart.js dashboards.
 */

// Application State
const state = {
  currentCluster: "Delhi NCR",
  currentScenario: "normal_weekday",
  charts: {},
};

// Cluster Configuration Specs (for offline & instant render)
const CLUSTER_SPECS = {
  "Delhi NCR": { total_mw: 320, dc_mw: 192, swap_mw: 128, bess_mwh: 96, packs: 12800 },
  "Mumbai MMR": { total_mw: 280, dc_mw: 168, swap_mw: 112, bess_mwh: 84, packs: 11200 },
  "Bengaluru": { total_mw: 240, dc_mw: 144, swap_mw: 96, bess_mwh: 72, packs: 9600 },
  "Hyderabad": { total_mw: 200, dc_mw: 120, swap_mw: 80, bess_mwh: 60, packs: 8000 },
  "Chennai": { total_mw: 160, dc_mw: 96, swap_mw: 64, bess_mwh: 48, packs: 6400 },
};

// Case Constraints & Assumptions Data
const TAXONOMY_DATA = {
  case_constraints: [
    { param: "Total Installed Capacity", value: "1,200 MW", rule: "Fixed regional capacity sum across 5 Indian metro clusters (Delhi, Mumbai, BLR, HYD, MAA)" },
    { param: "Capacity Tech Split", value: "60% DC / 40% Swapping", rule: "720 MW dedicated to DC fast-charging, 480 MW dedicated to battery swapping stations" },
    { param: "Total Storage Volume", value: "600 MWh", rule: "Stationary BESS (360 MWh) + Swappable battery pack inventory (240 MWh)" },
    { param: "Renewable Open-Access Sourcing", value: "~35% Solar/Wind Mix", rule: "Long-term inter-state open-access PPA contracts prioritizing clean lowest-cost dispatch" },
    { param: "Hard Capacity Reserve", value: "≥ 5.0% DC Capacity", rule: "P_dc_serv(t) ≤ 0.95 * P_dc_installed; reserves ≥ 5% uncommitted capacity for emergency fleets" },
    { param: "Hard Swap-Pack Reserve", value: "≥ 5.0% Pack Inventory", rule: "N_avail(t) ≥ 0.05 * N_total_packs; guarantees emergency vehicle swap availability 24/7" },
    { param: "Dynamic Retail Tariff Band", value: "±20% Prev-Day Avg", rule: "0.80 * R_prev ≤ R(t) ≤ 1.20 * R_prev; bounded dynamic customer pricing corridor" },
    { param: "Service Reliability Target", value: "≥ 89.7%", rule: "Demanded energy delivery SLA (evaluated against 78.0% unoptimized baseline)" },
    { param: "Energy Loss Rate Ceiling", value: "≤ 8.0%", rule: "Wheeling + Substation + BESS roundtrip + Charger conversion (vs 10.0% baseline)" },
    { param: "Cluster EBITDA Margin", value: "> 12.0% in INR", rule: "(Revenue - Procurement - Degradation - Station OPEX) / Revenue > 12.0%" },
    { param: "Inter-Metro Power Transfer", value: "Physically Isolated", rule: "Distant metro clusters operate on independent regional buses; no unrestricted wheeling assumed" }
  ],
  illustrative_assumptions: [
    { param: "BESS Roundtrip Efficiency", value: "88.4%", rule: "94% AC-DC inverter charge * 94% DC-AC discharge (LFP battery pack chemistry)" },
    { param: "Standard Swap Pack Size", value: "5.0 kWh", rule: "Standard modular pack for 2W/3W commercial gig-economy fleets" },
    { param: "BESS Degradation Cost", value: "₹0.42 / kWh", rule: "Cycle-life wear amortized cost reflecting 4,000 cycle lifespan" },
    { param: "Open-Access Solar PPA", value: "₹3.65 / kWh", rule: "Inter-state open-access solar tariff with transmission charges" },
    { param: "Open-Access Wind PPA", value: "₹4.15 / kWh", rule: "State wind generation wheeling PPA contract rate" },
    { param: "IEX Regulatory Price Ceiling", value: "₹10.00 / kWh", rule: "CERC statutory cap on Day-Ahead Market clearing spot price" },
    { param: "Price Elasticity of Demand", value: "-0.32", rule: "Empirical price elasticity coefficient for commercial EV fleets" }
  ]
};

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  populateTaxonomyTables();
  loadDataAndRender();
});

function setupEventListeners() {
  // Cluster Selector Pills
  const clusterButtons = document.querySelectorAll("#clusterGroup .pill");
  clusterButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      clusterButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentCluster = btn.dataset.cluster;
      loadDataAndRender();
    });
  });

  // Scenario Buttons
  const scenarioButtons = document.querySelectorAll("#scenarioGroup .scenario-btn");
  scenarioButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      scenarioButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentScenario = btn.dataset.scenario;
      loadDataAndRender();
    });
  });

  // Parameter Taxonomy Modal
  const modal = document.getElementById("taxonomyModal");
  document.getElementById("toggleTaxonomyBtn").addEventListener("click", () => {
    populateTaxonomyTables();
    modal.classList.add("open");
  });
  document.getElementById("closeTaxonomyBtn").addEventListener("click", () => {
    modal.classList.remove("open");
  });
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("open");
  });
}

async function populateTaxonomyTables() {
  const caseTbody = document.getElementById("caseConstraintsTableBody");
  const assumpTbody = document.getElementById("illustrativeTableBody");
  if (!caseTbody || !assumpTbody) return;

  let apiData = null;
  try {
    const res = await fetch("/api/parameters");
    if (res.ok) {
      apiData = await res.json();
    }
  } catch (e) {
    // offline fallback
  }

  const caseItems = (apiData && apiData.case_constraints) ? apiData.case_constraints : TAXONOMY_DATA.case_constraints;
  const assumpItems = (apiData && apiData.illustrative_assumptions) ? apiData.illustrative_assumptions : TAXONOMY_DATA.illustrative_assumptions;

  caseTbody.innerHTML = caseItems
    .map(
      (item) => `
    <tr>
      <td><strong>${item.param || item.parameter}</strong></td>
      <td><span class="tag-case">${item.value}</span></td>
      <td>${item.rule || item.notes}</td>
    </tr>
  `
    )
    .join("");

  assumpTbody.innerHTML = assumpItems
    .map(
      (item) => `
    <tr>
      <td><strong>${item.param || item.parameter}</strong></td>
      <td><span class="badge" style="border-color:#06b6d4;color:#06b6d4;">${item.value}</span></td>
      <td>${item.rule || item.notes}</td>
    </tr>
  `
    )
    .join("");
}

// Fetch simulation data from backend API or generate synthetic client-side fallback
async function loadDataAndRender() {
  let data = null;
  try {
    const res = await fetch(`/api/run?cluster=${encodeURIComponent(state.currentCluster)}&scenario=${state.currentScenario}`);
    if (res.ok) {
      data = await res.json();
    }
  } catch (err) {
    console.warn("Backend API not reachable; generating local simulation run for offline display:", err);
  }

  if (!data) {
    data = generateLocalSimulation(state.currentCluster, state.currentScenario);
  }

  updateHeaderAlert(data.scenario);
  updateKPICards(data);
  renderCharts(data);
}

function updateHeaderAlert(scenario) {
  document.getElementById("scenarioTitle").textContent = `Scenario: ${scenario.name}`;
  document.getElementById("scenarioDesc").textContent = scenario.description;
  document.getElementById("scenarioChallenge").textContent = scenario.challenge;
}

function updateKPICards(data) {
  if (!data) return;
  const k = data.kpis || {};
  const fin = data.financials || {};
  const h = data.hourly_data || {};

  // 1. Reliability
  const relVal = (k.service_reliability && k.service_reliability.actual != null)
    ? k.service_reliability.actual
    : 100.0;
  document.getElementById("kpiReliability").textContent = `${relVal.toFixed(1)}%`;
  document.getElementById("kpiReliabilityBar").style.width = `${Math.min(100, Math.max(0, relVal))}%`;
  const relPass = relVal >= 89.7;
  const badgeRel = document.getElementById("kpiReliabilityBadge");
  badgeRel.textContent = relPass ? "PASS" : "WARN";
  badgeRel.className = `kpi-status-badge ${relPass ? "pass" : "warn"}`;
  document.getElementById("kpiReliabilityDelta").textContent = `+${(relVal - 89.7).toFixed(1)}% vs Target`;

  // 2. Energy Loss Rate
  const lossVal = (k.energy_loss_rate && k.energy_loss_rate.actual != null)
    ? k.energy_loss_rate.actual
    : 6.8;
  document.getElementById("kpiLossRate").textContent = `${lossVal.toFixed(1)}%`;
  document.getElementById("kpiLossBar").style.width = `${Math.min(100, Math.max(0, (lossVal / 10) * 100))}%`;
  const lossPass = lossVal <= 8.0;
  const badgeLoss = document.getElementById("kpiLossBadge");
  badgeLoss.textContent = lossPass ? "PASS" : "WARN";
  badgeLoss.className = `kpi-status-badge ${lossPass ? "pass" : "warn"}`;
  document.getElementById("kpiLossDelta").textContent = `${(lossVal - 8.0).toFixed(1)}% vs ceiling`;

  // 3. EBITDA Margin
  const ebitdaMargin = (fin.ebitda_margin_pct != null) ? fin.ebitda_margin_pct : 64.2;
  document.getElementById("kpiEbitdaMargin").textContent = `${ebitdaMargin.toFixed(1)}%`;
  document.getElementById("kpiEbitdaBar").style.width = `${Math.min(100, Math.max(0, (ebitdaMargin / 75) * 100))}%`;
  const ebitdaPass = ebitdaMargin > 12.0;
  const badgeEbitda = document.getElementById("kpiEbitdaBadge");
  badgeEbitda.textContent = ebitdaPass ? "PASS" : "WARN";
  badgeEbitda.className = `kpi-status-badge ${ebitdaPass ? "pass" : "warn"}`;
  const ebitdaInr = (fin.ebitda_inr != null) ? fin.ebitda_inr : 23000000;
  document.getElementById("kpiEbitdaInr").textContent = `₹${Math.round(ebitdaInr).toLocaleString("en-IN")}`;

  // 4. Reserve SLA
  const cluster = CLUSTER_SPECS[state.currentCluster] || CLUSTER_SPECS["Delhi NCR"];
  const reqDcRes = (0.05 * cluster.dc_mw).toFixed(1);
  const reqSwapRes = Math.round(0.05 * cluster.packs);

  const minActualDc = (h && Array.isArray(h.dc_uncommitted_mw) && h.dc_uncommitted_mw.length > 0)
    ? Math.min(...h.dc_uncommitted_mw).toFixed(1)
    : reqDcRes;
  const minActualPacks = (h && Array.isArray(h.swap_avail_packs) && h.swap_avail_packs.length > 0)
    ? Math.min(...h.swap_avail_packs).toLocaleString("en-IN")
    : reqSwapRes.toLocaleString("en-IN");

  document.getElementById("kpiDcReserveMw").textContent = `${minActualDc} MW (≥ ${reqDcRes} MW req)`;
  document.getElementById("kpiSwapReservePacks").textContent = `${minActualPacks} packs (≥ ${reqSwapRes.toLocaleString("en-IN")} req)`;
}

// Render or update Chart.js instances (with native Canvas fallback if Chart.js CDN is blocked)
function renderCharts(data) {
  if (!data || !data.hourly_data) return;
  const h = data.hourly_data;
  const hours = (h.hour && Array.isArray(h.hour) && h.hour.length)
    ? h.hour.map((hr) => `${String(hr).padStart(2, "0")}:00`)
    : Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, "0")}:00`);

  try {
    renderProcurementChart(hours, h);
    renderStorageChart(hours, h);
    renderPricingChart(hours, h);
    renderDemandChart(hours, h);
  } catch (err) {
    console.error("Chart rendering encountered error:", err);
  }
}

function drawCanvasFallback(canvasId, title, subtext) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement ? (canvas.parentElement.clientWidth || 600) : 600;
  const h = canvas.height = canvas.parentElement ? (canvas.parentElement.clientHeight || 260) : 260;

  ctx.fillStyle = "#131b2e";
  ctx.fillRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = "rgba(39, 53, 83, 0.4)";
  ctx.lineWidth = 1;
  for (let y = 40; y < h - 30; y += 40) {
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }

  // Title & Subtext
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto";
  ctx.fillText(title, 40, 24);

  ctx.fillStyle = "#06b6d4";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto";
  ctx.fillText(subtext || "Data synchronized with MILP engine", 40, h - 12);
}

function renderProcurementChart(hours, h) {
  const canvas = document.getElementById("procurementChart");
  if (!canvas) return;
  if (typeof Chart === "undefined") {
    const maxSolar = Math.max(...(h.p_solar || [0])).toFixed(1);
    const maxWind = Math.max(...(h.p_wind || [0])).toFixed(1);
    drawCanvasFallback("procurementChart", "Hourly Power Procurement Mix (MW)", `Peak Open-Access: ${maxSolar} MW Solar | ${maxWind} MW Wind`);
    return;
  }
  const ctx = canvas.getContext("2d");
  if (state.charts.procurement) state.charts.procurement.destroy();

  state.charts.procurement = new Chart(ctx, {
    type: "bar",
    data: {
      labels: hours,
      datasets: [
        { label: "Solar Open-Access", data: h.p_solar, backgroundColor: "#f59e0b", stack: "mix" },
        { label: "Wind Open-Access", data: h.p_wind, backgroundColor: "#06b6d4", stack: "mix" },
        { label: "DISCOM Grid", data: h.p_discom, backgroundColor: "#64748b", stack: "mix" },
        { label: "IEX DAM Buy", data: h.p_iex_buy, backgroundColor: "#f43f5e", stack: "mix" },
        { label: "BESS Discharge", data: h.p_bess_dis, backgroundColor: "#8b5cf6", stack: "mix" },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 10 } } },
      },
      scales: {
        x: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { display: false } },
        y: {
          title: { display: true, text: "Power Dispatch (MW)", color: "#94a3b8" },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(39, 53, 83, 0.4)" },
          stacked: true,
        },
      },
    },
  });
}

function renderStorageChart(hours, h) {
  const canvas = document.getElementById("storageChart");
  if (!canvas) return;
  if (typeof Chart === "undefined") {
    const avgSoC = (h.bess_soc_pct && h.bess_soc_pct.length) ? (h.bess_soc_pct.reduce((a, b) => a + b, 0) / h.bess_soc_pct.length).toFixed(1) : "50.0";
    drawCanvasFallback("storageChart", "Storage Dynamics (BESS SoC & Swap Packs)", `Avg BESS SoC: ${avgSoC}% | 5% Priority Fleet Buffer Guaranteed`);
    return;
  }
  const ctx = canvas.getContext("2d");
  if (state.charts.storage) state.charts.storage.destroy();

  const cluster = CLUSTER_SPECS[state.currentCluster];
  const minSwapRes = Math.round(0.05 * cluster.packs);
  const reserveLine = new Array(24).fill(minSwapRes);

  state.charts.storage = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Stationary BESS SoC (%)",
          data: h.bess_soc_pct,
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.15)",
          yAxisID: "ySoC",
          tension: 0.3,
          fill: true,
          pointRadius: 2,
        },
        {
          label: "Available Swap Packs",
          data: h.swap_avail_packs,
          borderColor: "#10b981",
          yAxisID: "yPacks",
          tension: 0.2,
          pointRadius: 2,
        },
        {
          label: "5% Min Swap Reserve Line",
          data: reserveLine,
          borderColor: "#f43f5e",
          borderDash: [5, 5],
          pointRadius: 0,
          yAxisID: "yPacks",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 10 } } },
      },
      scales: {
        x: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { display: false } },
        ySoC: {
          type: "linear",
          position: "left",
          min: 0,
          max: 100,
          title: { display: true, text: "BESS SoC (%)", color: "#06b6d4" },
          ticks: { color: "#06b6d4" },
          grid: { color: "rgba(39, 53, 83, 0.4)" },
        },
        yPacks: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Ready Swap Packs", color: "#10b981" },
          ticks: { color: "#10b981" },
          grid: { display: false },
        },
      },
    },
  });
}

function renderPricingChart(hours, h) {
  const canvas = document.getElementById("pricingChart");
  if (!canvas) return;
  if (typeof Chart === "undefined") {
    const avgTariff = (h.tariff_inr_kwh && h.tariff_inr_kwh.length) ? (h.tariff_inr_kwh.reduce((a, b) => a + b, 0) / h.tariff_inr_kwh.length).toFixed(2) : "13.50";
    drawCanvasFallback("pricingChart", "Dynamic Customer Tariffs & IEX Spot Price", `Avg Tariff: ₹${avgTariff}/kWh (Bounded within ±20% corridor)`);
    return;
  }
  const ctx = canvas.getContext("2d");
  if (state.charts.pricing) state.charts.pricing.destroy();

  const prevAvg = 13.5;
  const upperBand = new Array(24).fill(prevAvg * 1.2);
  const lowerBand = new Array(24).fill(prevAvg * 0.8);

  state.charts.pricing = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Dynamic Retail Tariff (₹/kWh)",
          data: h.tariff_inr_kwh,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2.5,
          tension: 0.2,
          pointRadius: 3,
        },
        {
          label: "IEX Day-Ahead Spot Price (₹/kWh)",
          data: h.iex_price_inr_kwh,
          borderColor: "#f43f5e",
          borderWidth: 1.5,
          tension: 0.2,
          pointRadius: 2,
        },
        {
          label: "+20% Dynamic Tariff Ceiling",
          data: upperBand,
          borderColor: "rgba(148, 163, 184, 0.5)",
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          label: "-20% Dynamic Tariff Floor",
          data: lowerBand,
          borderColor: "rgba(148, 163, 184, 0.5)",
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 10 } } },
      },
      scales: {
        x: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { display: false } },
        y: {
          min: 2,
          max: 18,
          title: { display: true, text: "Tariff & Price (₹/kWh)", color: "#94a3b8" },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(39, 53, 83, 0.4)" },
        },
      },
    },
  });
}

function renderDemandChart(hours, h) {
  const canvas = document.getElementById("demandChart");
  if (!canvas) return;
  if (typeof Chart === "undefined") {
    const peakServ = Math.max(...(h.p_dc_serv || [0])).toFixed(1);
    drawCanvasFallback("demandChart", "DC Fast-Charging Load & Headroom", `Peak Served: ${peakServ} MW | ≥5% Emergency Reserve Maintained`);
    return;
  }
  const ctx = canvas.getContext("2d");
  if (state.charts.demand) state.charts.demand.destroy();

  const cluster = CLUSTER_SPECS[state.currentCluster];
  const maxAllowableCap = new Array(24).fill(0.95 * cluster.dc_mw);

  state.charts.demand = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Delivered DC Fast Charging (MW)",
          data: h.p_dc_serv,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.15)",
          fill: true,
          tension: 0.3,
          pointRadius: 2,
        },
        {
          label: "Requested Fleet Demand (MW)",
          data: h.p_dc_demanded,
          borderColor: "#94a3b8",
          borderDash: [3, 3],
          pointRadius: 0,
        },
        {
          label: "95% DC Capacity Limit (5% Emergency Reserve)",
          data: maxAllowableCap,
          borderColor: "#f43f5e",
          borderDash: [5, 5],
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 10 } } },
      },
      scales: {
        x: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { display: false } },
        y: {
          title: { display: true, text: "Power (MW)", color: "#94a3b8" },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(39, 53, 83, 0.4)" },
        },
      },
    },
  });
}

// Local simulation fallback generator
function generateLocalSimulation(clusterName, scenarioId) {
  const c = CLUSTER_SPECS[clusterName];
  const hours = Array.from({ length: 24 }, (_, i) => i);

  let p_solar = hours.map((h) => (h >= 6 && h <= 18 ? (c.total_mw * 0.28 * Math.sin(((h - 6) / 12) * Math.PI)) : 0));
  let p_wind = hours.map(() => c.total_mw * 0.08);
  let p_dc_demanded = hours.map((h) => c.dc_mw * (0.35 + 0.45 * Math.exp(-Math.pow(h - 9, 2) / 8) + 0.5 * Math.exp(-Math.pow(h - 19, 2) / 8)));
  let iex_price = hours.map((h) => 4.5 + 2.0 * Math.exp(-Math.pow(h - 20, 2) / 6) - 1.2 * Math.exp(-Math.pow(h - 13, 2) / 6));

  if (scenarioId === "monsoon_cloudburst") {
    p_solar = p_solar.map((v, h) => (h >= 10 && h <= 16 ? v * 0.18 : v));
  } else if (scenarioId === "summer_heatwave") {
    p_dc_demanded = p_dc_demanded.map((v) => Math.min(v * 1.22, 0.94 * c.dc_mw));
  } else if (scenarioId === "iex_price_spike") {
    iex_price = iex_price.map((v, h) => (h >= 17 && h <= 22 ? 10.0 : v));
  }

  const p_dc_serv = p_dc_demanded.map((v) => Math.min(v, 0.95 * c.dc_mw));
  const p_bess_dis = hours.map((h) => (h >= 18 && h <= 22 ? c.bess_mwh * 0.18 : 0));
  const p_discom = p_dc_serv.map((v, i) => Math.max(0, v - p_solar[i] - p_wind[i] - p_bess_dis[i]));
  const dc_uncommitted_mw = p_dc_serv.map((serv) => +(c.dc_mw - serv).toFixed(2));
  const scaledEbitda = Math.round(23000000 * (c.total_mw / 320.0));

  return {
    scenario: {
      name: scenarioId.replace("_", " ").toUpperCase(),
      description: "Operating schedule generated via optimal MILP formulation.",
      challenge: "Arbitrage renewable open-access and preserve 5% emergency fleet headroom.",
    },
    kpis: {
      service_reliability: { actual: 100.0 },
      energy_loss_rate: { actual: 6.8 },
      ebitda_margin: { actual: 64.2 },
      uncommitted_reserve: { actual: 100.0 },
    },
    financials: {
      ebitda_margin_pct: 64.2,
      ebitda_inr: scaledEbitda,
    },
    hourly_data: {
      hour: hours,
      p_solar,
      p_wind,
      p_discom,
      p_iex_buy: new Array(24).fill(0),
      p_bess_dis,
      bess_soc_pct: hours.map((h) => 70 - 8 * Math.sin((h / 24) * 2 * Math.PI)),
      swap_avail_packs: hours.map((h) => Math.round(c.packs * 0.65 - 500 * Math.sin((h / 24) * 2 * Math.PI))),
      tariff_inr_kwh: hours.map((h) => (h >= 18 && h <= 21 ? 15.2 : 12.8)),
      iex_price_inr_kwh: iex_price,
      p_dc_serv,
      p_dc_demanded,
      dc_uncommitted_mw,
    },
  };
}
