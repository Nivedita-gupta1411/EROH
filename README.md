# EROH – Regional EV Energy Optimization & Control Tower

**AI-Powered Energy Allocation, Battery Storage Optimization, IEX Market Bidding & Dynamic Pricing**

<p align="center">
  <a href="https://eroh-3.onrender.com"><strong>🚀 Live Demo – Open EROH</strong></a>
</p>

**Network Scale:** 1,200 MW · 600 MWh · 5 Indian Metro Clusters

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://eroh-3.onrender.com)
![Python](https://img.shields.io/badge/Python-3-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Optimization](https://img.shields.io/badge/Optimization-MILP-orange?style=for-the-badge)

---

## 1. Overview

EROH (EV Energy Optimization & Control Tower) is a modular Python-based prototype designed to optimize energy procurement, EV charging allocation, battery storage utilization, electricity market participation, and dynamic customer pricing across five major Indian metropolitan clusters.

The system combines forecasting, Mixed-Integer Linear Programming (MILP), Model Predictive Control (MPC), and scenario-based stress testing to demonstrate how a large-scale EV charging network can manage energy demand, operational constraints, and procurement costs.

### Network Coverage

| Metro Cluster | Installed Capacity | Storage Capacity |
|---|---:|---:|
| Delhi NCR | 320 MW | 160 MWh |
| Mumbai MMR | 280 MW | 140 MWh |
| Bengaluru | 240 MW | 120 MWh |
| Hyderabad | 200 MW | 100 MWh |
| Chennai | 160 MW | 80 MWh |
| **Total** | **1,200 MW** | **600 MWh** |

### Key Infrastructure Specifications

| Parameter | Specification |
|---|---:|
| Total installed network capacity | 1,200 MW |
| DC fast-charging allocation | 720 MW (60%) |
| Battery swapping allocation | 480 MW (40%) |
| Total storage capacity | 600 MWh |
| Renewable open-access target | Approximately 35% |
| DC capacity reserve | Minimum 5% |
| Ready swap-pack reserve | Minimum 5% |
| Dynamic tariff corridor | ±20% of previous-day average tariff |

**Energy sourcing strategy:**
- Solar and wind open-access procurement through long-term PPAs.
- DISCOM grid electricity with commercial Time-of-Day (ToD) pricing.
- Indian Energy Exchange (IEX) Day-Ahead Market (DAM) procurement and arbitrage.

---

## 2. Problem Statement

Large-scale EV charging networks face several interconnected challenges:

1. **Demand variability:** EV charging demand fluctuates throughout the day and across geographical clusters.
2. **Electricity procurement costs:** Grid tariffs and market prices vary, particularly during peak demand hours.
3. **Battery management:** Charging and discharging decisions must respect battery capacity, efficiency, and operational constraints.
4. **Renewable intermittency:** Solar and wind generation can fluctuate due to weather conditions.
5. **Service reliability:** Charging infrastructure must maintain sufficient capacity for customer demand and priority fleets.
6. **Dynamic pricing:** Retail tariffs must balance operational economics with customer affordability and demand responsiveness.

EROH addresses these challenges through an integrated forecasting and optimization framework.

---

## 3. System Architecture

The project follows a modular architecture with separate components for data generation, forecasting, optimization, control, scenario management, and web-based visualization.

```text
EROH/
│
├── core/
│   ├── __init__.py
│   ├── config.py
│   │   └── Cluster definitions, network allocations,
│   │       case constraints and economic parameters
│   │
│   ├── data_generator.py
│   │   └── Synthetic hourly demand, weather,
│   │       renewable generation, IEX prices and SoH
│   │
│   ├── forecaster.py
│   │   └── Rolling-origin forecasting and
│   │       P10/P50/P90 prediction
│   │
│   ├── milp_optimizer.py
│   │   └── scipy.optimize.milp (HiGHS)
│   │       for constrained energy scheduling
│   │
│   ├── mpc_controller.py
│   │   └── Receding-horizon Model Predictive Control
│   │
│   └── scenario_manager.py
│       └── Operational and market stress scenarios
│
├── app/
│   ├── server.py
│   │   └── FastAPI REST backend
│   │
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── tests/
│   └── test_system.py
│
├── main.py
├── requirements.txt
└── README.md
```

### Architecture Workflow

```text
Synthetic Data Generation
          |
          v
Demand & Renewable Forecasting
          |
          v
MILP Energy Optimization
          |
          v
Model Predictive Control
          |
          v
Energy Allocation & Scheduling
          |
          v
Scenario Stress Testing
          |
          v
FastAPI Backend
          |
          v
Interactive Control Tower Dashboard
```

---

## 4. Mathematical Optimization Formulation

EROH uses Mixed-Integer Linear Programming (MILP) to optimize energy allocation and procurement decisions over a 24-hour scheduling horizon.

Let:

- \(t\) represent the hourly time index.
- \(T=24\) represent the optimization horizon.
- \(P\) represent power variables in MW.
- \(E\) represent stored energy in MWh.
- \(R\) represent customer retail tariffs.

### 4.1 Objective Function

The optimization objective minimizes net operational cost while accounting for procurement, battery degradation, unserved demand, and revenue.

\[
\min \sum_{t=0}^{T-1}
\left[
C_{\mathrm{procure}}(t)
+C_{\mathrm{deg}}(t)
+C_{\mathrm{unserved}}(t)
-\mathrm{Revenue}(t)
\right]
\]

### 4.2 Energy Balance

The cluster energy balance accounts for renewable generation, grid procurement, battery discharge, charging demand, and market transactions.

\[
\begin{aligned}
&(1-L_{\mathrm{oa}})
(P_{\mathrm{solar},t}+P_{\mathrm{wind},t})\\
&+(1-L_{\mathrm{grid}})
(P_{\mathrm{discom},t}+P_{\mathrm{iex\_buy},t})\\
&+P_{\mathrm{bess\_dis},t}\\
&=P_{\mathrm{dc\_serv},t}
+P_{\mathrm{swap\_ch},t}
+P_{\mathrm{bess\_ch},t}
+P_{\mathrm{iex\_sell},t}
\end{aligned}
\]

### 4.3 Stationary Battery Energy Storage System (BESS)

Charging and discharging are mutually exclusive:

\[
u_{\mathrm{ch},t}+u_{\mathrm{dis},t}\leq 1
\]

\[
u_{\mathrm{ch},t},u_{\mathrm{dis},t}\in\{0,1\}
\]

Power limits:

\[
P_{\mathrm{bess\_ch},t}\leq
P_{\max}u_{\mathrm{ch},t}
\]

\[
P_{\mathrm{bess\_dis},t}\leq
P_{\max}u_{\mathrm{dis},t}
\]

State-of-charge transition:

\[
\begin{aligned}
E_{\mathrm{bess},t}
=&E_{\mathrm{bess},t-1}\\
&+\eta_{\mathrm{ch}}P_{\mathrm{bess\_ch},t-1}\Delta t\\
&-\frac{P_{\mathrm{bess\_dis},t-1}\Delta t}
{\eta_{\mathrm{dis}}}
\end{aligned}
\]

Operating limits:

\[
0.10E_{\mathrm{cap}}\leq E_{\mathrm{bess},t}
\leq 0.95E_{\mathrm{cap}}
\]

### 4.4 Battery Swapping Inventory

The ready-to-use battery inventory evolves based on swapping demand and newly recharged packs.

\[
N_{\mathrm{avail},t}
=
N_{\mathrm{avail},t-1}
-S_{t-1}
+
\left\lfloor
\frac{
P_{\mathrm{swap\_ch},t-1}
\eta_{\mathrm{pack}}\Delta t
}{
E_{\mathrm{pack}}
}
\right\rfloor
\]

Minimum inventory reserve:

\[
N_{\mathrm{avail},t}
\geq 0.05N_{\mathrm{total\_packs}}
\]

This ensures at least 5% of the total battery inventory remains reserved.

### 4.5 Dynamic Customer Pricing

Retail tariffs are constrained within a regulatory pricing corridor:

\[
0.80\bar R_{\mathrm{prev}}
\leq R_t\leq
1.20\bar R_{\mathrm{prev}}
\]

Demand responds to price changes through an elasticity-based relationship:

\[
D_{\mathrm{dc},t}(R_t)
=
D_{\mathrm{dc\_base},t}
\left[
1-\epsilon
\frac{R_t-\bar R_{\mathrm{prev}}}
{\bar R_{\mathrm{prev}}}
\right]
\]

DC charging capacity headroom:

\[
P_{\mathrm{dc\_serv},t}
\leq 0.95P_{\mathrm{dc\_installed}}
\]

This reserves at least 5% of installed DC capacity for priority and emergency fleets.

---

## 5. Parameter Taxonomy

The system distinguishes between case constraints and illustrative modeling assumptions.

### 5.1 Case Constraints

| Parameter | Value / Bound | Operational Role |
|---|---:|---|
| Total installed capacity | 1,200 MW | Network infrastructure |
| DC fast-charging share | 60% (720 MW) | Direct EV fast charging |
| Battery swapping share | 40% (480 MW) | Swapping infrastructure |
| Total storage volume | 600 MWh | BESS and swap-pack buffer |
| Renewable open-access target | ~35% | Renewable procurement |
| DC capacity reserve | ≥5% | Priority fleet headroom |
| Swap-pack reserve | ≥5% | Ready battery inventory |
| Retail tariff corridor | ±20% | Customer pricing protection |
| Service reliability target | ≥89.7% | Service-level objective |
| Energy loss ceiling | ≤8.0% | Efficiency objective |
| EBITDA margin target | >12.0% | Commercial viability |

### 5.2 Illustrative Modeling Assumptions

| Parameter | Assumed Value | Operational Role |
|---|---:|---|
| BESS round-trip efficiency | 88.4% | Battery conversion efficiency |
| Charging efficiency | 94% | Charging losses |
| Discharging efficiency | 94% | Discharging losses |
| Swap module size | 5 kWh | Standard battery module |
| BESS degradation cost | ₹0.42/kWh cycle | Battery wear cost |
| Solar PPA price | ₹3.65/kWh | Renewable procurement |
| Wind PPA price | ₹4.15/kWh | Renewable procurement |
| DISCOM grid tariff | ₹6.40–₹7.20/kWh | Grid procurement |
| ToD peak multiplier | 1.30x–1.40x | Peak tariff adjustment |
| IEX price ceiling assumption | ₹10.00/kWh | Market price limit |
| Price elasticity coefficient | -0.32 | Customer demand response |

*These values are illustrative modeling assumptions and should not be interpreted as verified real-time market quotations or regulatory approvals.*

---

## 6. Scenario Stress Testing

EROH includes four operational stress-test scenarios to evaluate the optimization framework under different operating conditions.

| Scenario | Description |
|---|---|
| Normal Weekday | Standard demand, solar generation, wind availability, and ToD tariffs |
| Monsoon Cloudburst | 80% daytime solar generation reduction with increased wind variability |
| Summer Heatwave | 46°C ambient temperature, 22% additional cooling load, and battery converter derating |
| IEX Price Spike | Market prices rise to the assumed ₹10/kWh ceiling during evening peak hours |

### Benchmark Results – Delhi NCR Hub

The following results are the prototype's reported benchmark outputs.

| Scenario | Service Reliability | Energy Loss | EBITDA Margin | Reserve Compliance | Daily Net EBITDA |
|---|---:|---:|---:|---:|---:|
| Normal Weekday | 100.0% | 6.8% | 68.1% | 100% | ₹23,360,671 |
| Monsoon Cloudburst | 100.0% | 6.4% | 67.3% | 100% | ₹23,455,961 |
| Summer Heatwave | 100.0% | 6.6% | 59.9% | 100% | ₹30,093,929 |
| IEX Price Spike | 100.0% | 6.1% | 57.3% | 100% | ₹21,802,011 |

### Target Compliance

| KPI | Target | Reported Benchmark |
|---|---:|---:|
| Service reliability | ≥89.7% | 100% |
| Energy loss rate | ≤8.0% | 6.1–6.8% |
| EBITDA margin | >12.0% | 57.3–68.1% |
| Reserve compliance | 100% | 100% |

**Important:** These are synthetic prototype simulation results, not measured results from a deployed commercial EV charging network. Real-world performance requires validation using operational data, actual tariffs, equipment specifications, and field trials.

---

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| API Backend | FastAPI |
| ASGI Server | Uvicorn |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Optimization Solver Interface | SciPy MILP (HiGHS) |
| Forecasting | Scikit-learn |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Chart.js |
| Deployment | Render |
| Version Control | Git & GitHub |

---

## 8. Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nivedita-gupta1411/EROH.git
cd EROH
```

### Step 2: Create a Virtual Environment

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Automated Tests

```bash
python -m unittest tests/test_system.py
```

### Step 5: Run the End-to-End CLI Prototype

```bash
python main.py
```

This executes the prototype workflow, including synthetic data generation, forecasting, MILP optimization, and scenario stress testing.

### Step 6: Launch the Web Application

```bash
python app/server.py
```

Open:

```text
http://127.0.0.1:8000
```

---

## 9. Live Deployment

The EROH web application is deployed on Render.

**Live Application:** https://eroh-3.onrender.com

**API Documentation:** https://eroh-3.onrender.com/docs

**API Parameters:** https://eroh-3.onrender.com/api/parameters

The deployed application provides access to the interactive control tower and its exposed backend APIs.

*Note: The free hosting instance may spin down after inactivity, causing a delay on the first request.*

---

## 10. Testing & Validation

The project includes an automated test suite:

```text
tests/test_system.py
```

Run it using:

```bash
python -m unittest tests/test_system.py
```

The test suite is intended to verify the mathematical formulations, system constraints, and optimization behavior implemented in the prototype.

---

## 11. Future Scope

Potential extensions to the prototype include:

- Integration with real-time EV charging station telemetry.
- Live weather and renewable generation APIs.
- Actual IEX market data integration.
- More advanced probabilistic demand forecasting.
- Battery State-of-Health (SoH) and degradation modeling.
- Multi-objective optimization for cost, emissions, and reliability.
- Production-grade authentication and role-based access.
- Cloud database integration and persistent operational monitoring.
- Deployment validation using real-world operational datasets.

---

## 12. Important Modeling Note

The project specification mentions 96,000 standard 5 kWh swap modules, which corresponds to 480 MWh of nominal pack energy. The storage allocation table separately specifies 240 MWh for swappable packs (360 MWh stationary BESS + 240 MWh swap packs = 600 MWh total).

These two figures need reconciliation before treating the storage allocation as a finalized system specification.

---

## 13. Project Disclaimer

EROH is an engineering and optimization prototype developed to demonstrate an integrated approach to EV energy management.

The data, tariff values, forecasts, and benchmark outputs are illustrative or synthetic unless explicitly identified otherwise. The prototype is not a certified grid-control platform, a live IEX trading system, or a substitute for regulatory, electrical, or commercial validation.

---

## Author

**Nivedita Gupta**

GitHub: [Nivedita-gupta1411](https://github.com/Nivedita-gupta1411)

Project Repository: [EROH](https://github.com/Nivedita-gupta1411/EROH)

**🚀 Explore the Live Demo:** https://eroh-3.onrender.com
