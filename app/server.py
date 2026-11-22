"""FastAPI backend server for the Regional EV Energy Optimization Control Tower."""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure project root is present in sys.path regardless of execution directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import (
    CLUSTER_CONFIGS, CaseConstraints, SimulationAssumptions, VALIDATION_DISCLAIMER
)
from core.scenario_manager import ScenarioManager, SCENARIO_CATALOG

app = FastAPI(
    title="Regional EV Energy Optimization Control Tower API",
    description="1,200 MW EV Network | 600 MWh Storage across 5 Indian Metro Clusters",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scenario_manager = ScenarioManager(seed=42)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/api/topology")
def get_topology():
    """Return cluster capacities, network totals, and system constraints."""
    clusters_data = {}
    for name, c in CLUSTER_CONFIGS.items():
        clusters_data[name] = {
            "name": c.name,
            "code": c.code,
            "total_capacity_mw": c.total_capacity_mw,
            "dc_capacity_mw": c.dc_capacity_mw,
            "swap_capacity_mw": c.swap_capacity_mw,
            "storage_capacity_mwh": c.storage_capacity_mwh,
            "stationary_bess_mwh": c.stationary_bess_mwh,
            "swap_pack_storage_mwh": c.swap_pack_storage_mwh,
            "total_swap_packs": c.total_swap_packs,
            "base_discom_tariff_inr_kwh": c.base_discom_tariff_inr_kwh,
            "peak_discom_multiplier": c.peak_discom_multiplier,
            "open_access_solar_mw": c.open_access_solar_mw,
            "open_access_wind_mw": c.open_access_wind_mw,
        }
    
    return {
        "network_total_mw": CaseConstraints.TOTAL_NETWORK_CAPACITY_MW,
        "dc_share_pct": CaseConstraints.DC_FAST_CHARGING_SHARE * 100.0,
        "swap_share_pct": CaseConstraints.BATTERY_SWAPPING_SHARE * 100.0,
        "total_storage_mwh": CaseConstraints.TOTAL_STORAGE_CAPACITY_MWH,
        "renewable_share_target_pct": CaseConstraints.RENEWABLE_SHARE_TARGET * 100.0,
        "clusters": clusters_data,
        "benchmarks": {
            "service_reliability_target": CaseConstraints.SERVICE_RELIABILITY_TARGET,
            "service_reliability_baseline": CaseConstraints.SERVICE_RELIABILITY_BASELINE,
            "energy_loss_rate_target": CaseConstraints.ENERGY_LOSS_RATE_TARGET,
            "energy_loss_rate_baseline": CaseConstraints.ENERGY_LOSS_RATE_BASELINE,
            "ebitda_margin_target": CaseConstraints.CLUSTER_EBITDA_MARGIN_TARGET,
            "uncommitted_reserve_pct": CaseConstraints.MIN_UNCOMMITTED_CAPACITY_RESERVE * 100.0,
            "dynamic_tariff_band_pct": 20.0,
        }
    }


@app.get("/api/scenarios")
def get_scenarios():
    """Return list of available scenario stress tests."""
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "weather": s.weather_condition,
            "grid": s.grid_condition,
            "challenge": s.key_challenge,
        }
        for s in SCENARIO_CATALOG.values()
    ]


@app.get("/api/run")
def run_simulation(cluster: str = "Delhi NCR", scenario: str = "normal_weekday"):
    """Execute on-the-fly MILP dispatch optimization for specified cluster and scenario."""
    if cluster not in CLUSTER_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster}' not found.")
    if scenario not in SCENARIO_CATALOG:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario}' not found.")
    
    result = scenario_manager.run_scenario(cluster_name=cluster, scenario_id=scenario)
    return JSONResponse(content=result)


@app.get("/api/parameters")
def get_parameter_distinction():
    """Return explicit taxonomy separating case constraints from illustrative simulation assumptions."""
    return {
        "disclaimer": VALIDATION_DISCLAIMER,
        "case_constraints": [
            {"parameter": "Total Network Installed Capacity", "param": "Total Network Installed Capacity", "value": "1,200 MW", "notes": "Mandated network size across 5 metro clusters", "rule": "Fixed regional capacity across Delhi, Mumbai, BLR, HYD, MAA"},
            {"parameter": "DC Fast-Charging Share", "param": "DC Fast-Charging Share", "value": "60% (720 MW)", "notes": "Mandated share for direct DC fast charging", "rule": "720 MW dedicated to commercial direct fast-charging"},
            {"parameter": "Battery Swapping Share", "param": "Battery Swapping Share", "value": "40% (480 MW)", "notes": "Mandated share for battery swapping stations", "rule": "480 MW dedicated to modular swapping bays (96k packs)"},
            {"parameter": "Total Network Storage", "param": "Total Network Storage", "value": "600 MWh", "notes": "Partitioned between stationary BESS and ready packs", "rule": "360 MWh stationary BESS + 240 MWh swap pack buffer"},
            {"parameter": "Renewable Open-Access Sourcing", "param": "Renewable Open-Access Sourcing", "value": "~35%", "notes": "Target long-term solar & wind PPA wheeling", "rule": "Inter-state open-access solar/wind PPA agreements"},
            {"parameter": "Priority Fleet Hard Capacity Reserve", "param": "Priority Fleet Hard Capacity Reserve", "value": ">= 5.0%", "notes": "Hard constraint: at least 5% DC capacity reserved", "rule": "P_dc_serv(t) <= 0.95 * P_dc_installed"},
            {"parameter": "Priority Fleet Swap-Pack Reserve", "param": "Priority Fleet Swap-Pack Reserve", "value": ">= 5.0%", "notes": "Hard constraint: at least 5% ready packs reserved", "rule": "N_avail(t) >= 0.05 * N_total_packs"},
            {"parameter": "Dynamic Tariff Corridor", "param": "Dynamic Tariff Corridor", "value": "±20% of prev day avg", "notes": "Upper & lower bounds on hourly customer retail pricing", "rule": "0.80 * R_prev <= R(t) <= 1.20 * R_prev"},
            {"parameter": "Service Reliability Target", "param": "Service Reliability Target", "value": ">= 89.7%", "notes": "Target SLA vs. 78.0% unoptimized baseline", "rule": "Total delivered energy / requested fleet load >= 89.7%"},
            {"parameter": "Energy Loss Rate Target", "param": "Energy Loss Rate Target", "value": "<= 8.0%", "notes": "Total losses (T&D, conversion, BESS) vs. 10.0% baseline", "rule": "Total system physical losses / energy input <= 8.0%"},
            {"parameter": "Cluster EBITDA Margin Target", "param": "Cluster EBITDA Margin Target", "value": "> 12.0%", "notes": "Commercial viability in Indian Rupees (INR)", "rule": "(Revenue - Procurement - Degradation - OPEX) / Revenue > 12.0%"},
            {"parameter": "Inter-Metro Physical Transfer", "param": "Inter-Metro Physical Transfer", "value": "DISALLOWED (Isolated)", "notes": "Independent regional nodal buses", "rule": "Distant metro clusters operate on separate state buses"},
        ],
        "illustrative_assumptions": [
            {"parameter": "BESS Roundtrip Efficiency", "param": "BESS Roundtrip Efficiency", "value": "88.4%", "notes": "94% charge efficiency * 94% discharge efficiency (LFP/NMC)", "rule": "Standard high-performance LFP stationary cells"},
            {"parameter": "Battery Swap Module Size", "param": "Battery Swap Module Size", "value": "5.0 kWh", "notes": "Standard Indian 2W/3W swappable pack form factor", "rule": "Commercial gig-worker modular standard pack"},
            {"parameter": "BESS Cell Degradation Cost", "param": "BESS Cell Degradation Cost", "value": "₹0.42 / kWh", "notes": "Amortized wear and capacity fade penalty per cycle", "rule": "Reflects replacement amortized cost over 4,000 cycles"},
            {"parameter": "Open-Access Solar PPA Price", "param": "Open-Access Solar PPA Price", "value": "₹3.65 / kWh", "notes": "Utility-scale solar tariff with interstate transmission", "rule": "Long-term solar power purchase contract rate"},
            {"parameter": "Open-Access Wind PPA Price", "param": "Open-Access Wind PPA Price", "value": "₹4.15 / kWh", "notes": "Interstate wind power purchase agreement rate", "rule": "State-level wind wheeling PPA contract rate"},
            {"parameter": "DISCOM Base Grid Tariffs", "param": "DISCOM Base Grid Tariffs", "value": "₹6.40 - ₹7.20 / kWh", "notes": "State-specific commercial ToD base rates (Delhi, Mumbai, etc.)", "rule": "Standard utility base commercial tariff"},
            {"parameter": "DISCOM Peak Multiplier", "param": "DISCOM Peak Multiplier", "value": "1.30x - 1.40x", "notes": "Time-of-day surcharge during 06-10 and 18-22 hrs", "rule": "ToD peak-hour multiplier applied to base tariff"},
            {"parameter": "IEX Regulatory Price Cap", "param": "IEX Regulatory Price Cap", "value": "₹10.00 / kWh", "notes": "CERC statutory ceiling for Day-Ahead spot market", "rule": "Statutory clearing price upper limit"},
            {"parameter": "Price Elasticity of Demand", "param": "Price Elasticity of Demand", "value": "-0.32", "notes": "Fleet volume change per 1% change in dynamic tariff", "rule": "D(R) = D_base * [1 - epsilon * (R - R_prev) / R_prev]"},
        ]
    }


# Serve static web frontend:
# 1. Mount at /static for any explicit /static/... references
# 2. Mount at root / with html=True AFTER all API routes so /, /style.css, /app.js, /index.html resolve with HTTP 200
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static_root")
else:
    @app.get("/")
    def serve_fallback():
        return {"message": "Regional EV Energy Optimization Control Tower API is active. Open /api/topology or /api/run."}


def start_server(port: int = 8000):
    uvicorn.run("app.server:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    start_server()
