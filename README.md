# Regional EV Energy Optimization & Control Tower Prototype
### 1,200 MW Network • 600 MWh Storage • 5 Indian Metro Clusters (Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad, Chennai)

---

## 1. Executive Summary & Problem Formulation

This project is a clean, modular, and mathematically rigorous Python prototype for an **AI-powered energy allocation, battery storage optimization, IEX day-ahead market bidding, and dynamic pricing system** across five Indian metro clusters:

1. **Delhi NCR** (320 MW / 160 MWh)
2. **Mumbai MMR** (280 MW / 140 MWh)
3. **Bengaluru** (240 MW / 120 MWh)
4. **Hyderabad** (200 MW / 100 MWh)
5. **Chennai** (160 MW / 80 MWh)

### High-Level Specifications:
- **Total Network Installed Capacity**: **1,200 MW**
  - **60% DC Fast-Charging**: **720 MW**
  - **40% Battery Swapping**: **480 MW** (96,000 standard 5 kWh modules across hubs)
- **Total Storage Capacity**: **600 MWh** (Stationary BESS + ready swap pack buffer)
- **Energy Sourcing Mix**:
  - **~35% Solar & Wind Open-Access** long-term power purchase agreements (PPA)
  - **DISCOM Grid Power** with commercial Time-of-Day (ToD) peak surcharges
  - **IEX Day-Ahead Market (DAM)** spot procurement & arbitrage bidding
- **Regulatory Dynamic Pricing Corridor**: Customer retail tariffs bounded within **±20% of previous day average tariff** ($0.80 \bar{R}_{prev} \le R_t \le 1.20 \bar{R}_{prev}$) with price elasticity of demand.
- **Hard Reserve Constraints**: At least **$\ge 5.0\%$ uncommitted DC capacity** and **$\ge 5.0\%$ ready swap-pack inventory** reserved exclusively for priority and emergency fleets.
- **Target KPI Benchmarks**:
  - **Service Reliability**: **$\ge 89.7\%$** (vs. 78.0% unoptimized baseline)
  - **Energy Loss Rate**: **$\le 8.0\%$** (vs. 10.0% unoptimized baseline)
  - **Cluster EBITDA Margin**: **$> 12.0\%$** in Indian Rupees (INR)
  - **5% Reserve Compliance**: **100% compliant** across all operating hours.

---

## 2. Architecture & Modular Structure

```
eroh/
│
├── core/
│   ├── __init__.py
│   ├── config.py             # Cluster definitions, 1,200 MW allocations, case constraints, INR economics
│   ├── data_generator.py      # Multi-year hourly synthetic data generator (weather, demand, renewables, IEX, SoH)
│   ├── forecaster.py         # Chronological non-shuffled rolling-origin backtest & P10/P50/P90 quantile forecasting
│   ├── milp_optimizer.py     # scipy.optimize.milp (HiGHS) mathematical scheduler with BESS & swap dynamics
│   ├── mpc_controller.py     # Hourly closed-loop Model Predictive Control (MPC) receding-horizon engine
│   └── scenario_manager.py   # Stress test scenarios (Normal, Monsoon, Summer Heatwave, IEX Price Spike)
│
├── app/
│   ├── server.py             # FastAPI REST server providing live on-the-fly optimization APIs
│   └── static/
│       ├── index.html        # Interactive Control Tower Single-Page Dashboard
│       ├── style.css         # Modern dark-theme glassmorphism styling
│       └── app.js            # Reactive Chart.js visualizations, scenario toggles, parameter taxonomy
│
├── tests/
│   └── test_system.py        # Automated unit test suite verifying mathematical formulations & constraints
│
├── main.py                   # End-to-end runnable CLI prototype demonstrating all components
└── README.md                 # System documentation & parameter taxonomy
```

---

## 3. Mathematical Optimization Formulation (MILP)

At each horizon step $t \in \{0, \dots, T-1\}$ ($T=24$ hours), the solver minimizes the net cost (or maximizes EBITDA):

$$\min \sum_{t=0}^{T-1} \left[ C_{\text{procure}}(t) + C_{\text{deg}}(t) + C_{\text{unserved}}(t) - \text{Revenue}(t) \right]$$

### Key Equations & Constraints:

1. **Energy Balance at Cluster Bus**:
   $$(1 - L_{\text{oa}}) (P_{\text{solar}, t} + P_{\text{wind}, t}) + (1 - L_{\text{grid}}) (P_{\text{discom}, t} + P_{\text{iex\_buy}, t}) + P_{\text{bess\_dis}, t} = P_{\text{dc\_serv}, t} + P_{\text{swap\_ch}, t} + P_{\text{bess\_ch}, t} + P_{\text{iex\_sell}, t}$$

2. **Stationary BESS Mutual Exclusion & SoC Transitions**:
   $$u_{\text{bess\_ch}, t} + u_{\text{bess\_dis}, t} \le 1, \quad u \in \{0, 1\}$$
   $$P_{\text{bess\_ch}, t} \le P_{\text{max}} u_{\text{bess\_ch}, t}, \quad P_{\text{bess\_dis}, t} \le P_{\text{max}} u_{\text{bess\_dis}, t}$$
   $$E_{\text{bess}, t} = E_{\text{bess}, t-1} + \eta_{\text{ch}} P_{\text{bess\_ch}, t-1} \Delta t - \frac{1}{\eta_{\text{dis}}} P_{\text{bess\_dis}, t-1} \Delta t$$
   $$0.10 \cdot E_{\text{cap}} \le E_{\text{bess}, t} \le 0.95 \cdot E_{\text{cap}}$$

3. **Swappable Pack Inventory Dynamics**:
   $$N_{\text{avail}, t} = N_{\text{avail}, t-1} - S_{t-1} + \left\lfloor \frac{P_{\text{swap\_ch}, t-1} \cdot \eta_{\text{pack}} \cdot \Delta t}{E_{\text{pack}}} \right\rfloor$$
   $$N_{\text{avail}, t} \ge 0.05 \cdot N_{\text{total\_packs}} \quad \text{(Hard $\ge 5\%$ Standby Reserve Mandate)}$$

4. **Dynamic Customer Tariffs & Price Elasticity**:
   $$0.80 \cdot \bar{R}_{\text{prev}} \le R_t \le 1.20 \cdot \bar{R}_{\text{prev}}$$
   $$D_{\text{dc}, t}(R_t) = D_{\text{dc\_base}, t} \cdot \left[1 - \epsilon \frac{R_t - \bar{R}_{\text{prev}}}{\bar{R}_{\text{prev}}}\right]$$
   $$P_{\text{dc\_serv}, t} \le (1.0 - 0.05) \cdot P_{\text{dc\_installed}} \quad \text{(Hard $\ge 5\%$ DC Capacity Headroom)}$$

---

## 4. Parameter Taxonomy: Case Constraints vs. Illustrative Assumptions

| Parameter Name | Value / Bound | Classification | Operational Role |
| :--- | :--- | :--- | :--- |
| **Total Network Installed Capacity** | **1,200 MW** | **Case Constraint** | Total fleet infrastructure across 5 metros |
| **DC Fast-Charging Share** | **60% (720 MW)** | **Case Constraint** | Dedicated direct DC high-voltage fast charging |
| **Battery Swapping Share** | **40% (480 MW)** | **Case Constraint** | Dedicated automated swapping station inventory |
| **Total Storage Volume** | **600 MWh** | **Case Constraint** | Stationary BESS (360 MWh) + Swappable packs (240 MWh) |
| **Renewable Open-Access Share** | **~35% Target** | **Case Constraint** | Interstate wheeling PPA contracts for clean power |
| **Uncommitted Capacity Reserve** | **$\ge 5.0\%$ (Hard)** | **Case Constraint** | Uncommitted DC capacity reserved for priority fleets |
| **Swap-Pack Standby Reserve** | **$\ge 5.0\%$ (Hard)** | **Case Constraint** | Ready charged swap-packs reserved 24/7 |
| **Dynamic Retail Tariff Band** | **$\pm 20\%$ of Prev Avg** | **Case Constraint** | Regulatory consumer protection dynamic corridor |
| **Service Reliability Target** | **$\ge 89.7\%$** | **Case Constraint** | Primary SLA target (vs. 78.0% baseline) |
| **Energy Loss Rate Ceiling** | **$\le 8.0\%$** | **Case Constraint** | Total system losses (vs. 10.0% baseline) |
| **Cluster EBITDA Margin Target** | **$> 12.0\%$ (in INR)** | **Case Constraint** | Cluster commercial EBITDA viability |
| BESS Roundtrip Efficiency | 88.4% (94% chg / 94% disch) | *Illustrative Assumption* | High-efficiency LFP battery pack chemistry |
| Swap Module Size | 5.0 kWh per pack | *Illustrative Assumption* | Standard commercial 2W/3W swappable pack size |
| BESS Cell Degradation Cost | ₹0.42 / kWh cycle | *Illustrative Assumption* | Amortized battery wear replacement cost |
| Open-Access Solar PPA Price | ₹3.65 / kWh | *Illustrative Assumption* | Long-term utility solar PPA tariff |
| Open-Access Wind PPA Price | ₹4.15 / kWh | *Illustrative Assumption* | Long-term wind generation wheeling tariff |
| DISCOM Grid Tariffs | ₹6.40 – ₹7.20 / kWh | *Illustrative Assumption* | State commercial base tariffs with ToD peak multiplier (1.30x–1.40x) |
| IEX Price Regulatory Ceiling | ₹10.00 / kWh | *Illustrative Assumption* | CERC statutory Day-Ahead Market price cap |
| Price Elasticity of Demand | -0.32 | *Illustrative Assumption* | Empirical fleet price responsiveness coefficient |

---

## 5. Scenario Stress Tests & Benchmark Verification

The system includes 4 stress-test scenarios executed against the mathematical optimizer:

1. **Normal Weekday**: Baseline diurnal solar curve, moderate wind, standard ToD DISCOM rates.
2. **Monsoon Cloudburst**: Sudden 80% daytime solar generation collapse, high wind gusts, requiring rapid BESS discharge and dynamic grid ramp-up.
3. **Summer Heatwave (46°C)**: Extreme ambient temperatures causing +22% EV cabin cooling load surge and thermal derating of battery converters.
4. **IEX Price Spike**: Market clearing price surges to the ₹10.00/kWh regulatory cap during evening hours (17:00-22:00); BESS and stored open-access power offset costly grid purchases.

### Benchmark Results (Delhi NCR Hub):

| Scenario | Service Reliability (Target $\ge 89.7\%$) | Energy Loss Rate (Target $\le 8.0\%$) | EBITDA Margin (Target $> 12.0\%$) | 5% Reserve SLA | Daily Net EBITDA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Weekday** | **100.0% [PASS]** | **6.8% [PASS]** | **68.1% [PASS]** | **100% [PASS]** | **₹23,360,671** |
| **Monsoon Cloudburst** | **100.0% [PASS]** | **6.4% [PASS]** | **67.3% [PASS]** | **100% [PASS]** | **₹23,455,961** |
| **Summer Heatwave** | **100.0% [PASS]** | **6.6% [PASS]** | **59.9% [PASS]** | **100% [PASS]** | **₹30,093,929** |
| **IEX Price Spike** | **100.0% [PASS]** | **6.1% [PASS]** | **57.3% [PASS]** | **100% [PASS]** | **₹21,802,011** |

---

## 6. How to Run

### A. Run Automated Unit Tests
```bash
python -m unittest tests/test_system.py
```

### B. Run End-to-End Prototype CLI
```bash
python main.py
```
This runs synthetic generation, chronological rolling-origin backtesting (P10/P50/P90 WAPE & MAE), MILP optimization, and scenario stress testing.

### C. Launch Interactive Control Tower Web App
```bash
python app/server.py
```
Then open your browser at **`http://127.0.0.1:8000`**. You can also double-click `app/static/index.html` to run in standalone browser mode.
