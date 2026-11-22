"""Main runnable prototype for AI-Powered Regional EV Energy Optimization System.

Demonstrates:
1. Network capacity allocation across 5 Indian metro clusters (1,200 MW / 600 MWh)
2. Synthetic data generation with weather, demand, renewables, IEX prices, and SoH
3. Chronological rolling-origin backtesting with P10/P50/P90 quantile forecasting (WAPE & MAE)
4. Mixed-Integer Linear Programming (MILP) scheduler & MPC closed-loop simulation
5. 4 Scenario Stress Tests (Normal, Monsoon Cloudburst, Summer Heatwave, IEX Price Spike)
"""

import sys
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from core.config import CaseConstraints, SimulationAssumptions, CLUSTER_CONFIGS
from core.data_generator import SyntheticDataGenerator
from core.forecaster import ProbabilisticForecaster
from core.milp_optimizer import ClusterMILPOptimizer
from core.mpc_controller import MPCEngine
from core.scenario_manager import ScenarioManager, SCENARIO_CATALOG


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)


def display_network_topology():
    print_header("1. Network Topology & Capacity Allocation (1,200 MW / 600 MWh)")
    print(f"Case Mandate: 60% DC Fast-Charging (720 MW) | 40% Battery Swapping (480 MW)")
    print(f"Storage: 600 MWh Total | ~35% Open-Access Solar/Wind Power PPA Target\n")
    
    rows = []
    for name, c in CLUSTER_CONFIGS.items():
        rows.append({
            "Cluster": name,
            "Total (MW)": c.total_capacity_mw,
            "DC (MW)": c.dc_capacity_mw,
            "Swap (MW)": c.swap_capacity_mw,
            "Storage (MWh)": c.storage_capacity_mwh,
            "BESS (MWh)": c.stationary_bess_mwh,
            "Swap Packs": c.total_swap_packs,
            "Solar (MW)": c.open_access_solar_mw,
            "Wind (MW)": c.open_access_wind_mw,
            "DISCOM (₹/kWh)": f"₹{c.base_discom_tariff_inr_kwh:.2f}",
        })
    df_topo = pd.DataFrame(rows)
    print(df_topo.to_string(index=False))
    
    tot_mw = sum(c.total_capacity_mw for c in CLUSTER_CONFIGS.values())
    tot_dc = sum(c.dc_capacity_mw for c in CLUSTER_CONFIGS.values())
    tot_swap = sum(c.swap_capacity_mw for c in CLUSTER_CONFIGS.values())
    tot_mwh = sum(c.storage_capacity_mwh for c in CLUSTER_CONFIGS.values())
    print("-" * 80)
    print(f"NETWORK TOTAL: {tot_mw:.0f} MW Installed ({tot_dc:.0f} MW DC + {tot_swap:.0f} MW Swap) | {tot_mwh:.0f} MWh Storage")


def run_forecasting_backtest(df_delhi: pd.DataFrame):
    print_header("2. Chronological Rolling-Origin Backtesting (P10 / P50 / P90)")
    print("Strictly non-shuffled expanding window evaluation preventing future data leakage.")
    print("Evaluating WAPE (%), MAE, and Pinball Quantile Losses across 24-hour day-ahead horizons:\n")
    
    forecaster = ProbabilisticForecaster(quantiles=(0.10, 0.50, 0.90))
    targets = [
        ("dc_demand_mw", "DC Fast-Charging Demand (MW)"),
        ("solar_gen_mw", "Solar PV Open-Access Generation (MW)"),
        ("iex_price_inr_kwh", "IEX Day-Ahead Clearing Price (₹/kWh)"),
    ]
    
    results = []
    for target_col, desc in targets:
        metrics, _ = forecaster.rolling_origin_backtest(
            df=df_delhi,
            target_col=target_col,
            initial_train_hours=24 * 60,  # 60 days initial history
            step_hours=24 * 7,           # step forward 1 week at a time
            horizon_hours=24,
            max_origins=4,
        )
        results.append({
            "Target Feature": desc,
            "P50 WAPE (%)": f"{metrics.wape_p50:.2f}%",
            "P50 MAE": f"{metrics.mae_p50:.3f}",
            "Pinball P10": f"{metrics.pinball_p10:.3f}",
            "Pinball P50": f"{metrics.pinball_p50:.3f}",
            "Pinball P90": f"{metrics.pinball_p90:.3f}",
            "P10-P90 Coverage": f"{metrics.coverage_80:.1f}% (Nominal 80%)",
        })
    df_metrics = pd.DataFrame(results)
    print(df_metrics.to_string(index=False))


def run_scenarios_and_benchmarks():
    print_header("3. Scenario Stress Tests & Target KPI Verification")
    print(f"Target Benchmarks:")
    print(f"  • Service Reliability: >= {CaseConstraints.SERVICE_RELIABILITY_TARGET}% (vs. {CaseConstraints.SERVICE_RELIABILITY_BASELINE}% Baseline)")
    print(f"  • Energy Loss Rate:    <= {CaseConstraints.ENERGY_LOSS_RATE_TARGET}% (vs. {CaseConstraints.ENERGY_LOSS_RATE_BASELINE}% Baseline)")
    print(f"  • Cluster EBITDA:      > {CaseConstraints.CLUSTER_EBITDA_MARGIN_TARGET}% in INR")
    print(f"  • Hard Reserves:       >= 5.0% Uncommitted DC Capacity & Ready Swap-Packs\n")
    
    mgr = ScenarioManager(seed=42)
    clusters = ["Delhi NCR", "Mumbai MMR", "Bengaluru", "Hyderabad", "Chennai"]
    scenarios = ["normal_weekday", "monsoon_cloudburst", "summer_heatwave", "iex_price_spike"]

    print("Summary of Scenarios Executed Across Delhi NCR Hub (320 MW / 160 MWh):")
    summary_rows = []
    for sc_id in scenarios:
        res = mgr.run_scenario("Delhi NCR", sc_id)
        k = res["kpis"]
        fin = res["financials"]
        en = res["energy_summary"]
        
        summary_rows.append({
            "Scenario": res["scenario"]["name"],
            "Reliability (%)": f"{k['service_reliability']['actual']:.1f}% [{k['service_reliability']['status']}]",
            "Loss Rate (%)": f"{k['energy_loss_rate']['actual']:.1f}% [{k['energy_loss_rate']['status']}]",
            "EBITDA Margin": f"{fin['ebitda_margin_pct']:.1f}% [{k['ebitda_margin']['status']}]",
            "5% Reserve SLA": f"{k['uncommitted_reserve']['actual']:.0f}% [{k['uncommitted_reserve']['status']}]",
            "Renewable Mix": f"{en['renewable_share_pct']:.1f}%",
            "Daily EBITDA": f"₹{fin['ebitda_inr']:,.0f}",
        })
    df_sc = pd.DataFrame(summary_rows)
    print(df_sc.to_string(index=False))


def print_parameter_taxonomy():
    print_header("4. System Parameters: Case Constraints vs. Illustrative Assumptions")
    print("┌──────────────────────────────────┬──────────────────────┬────────────────────────────────────────────────────────┐")
    print("│ Parameter Name                   │ Value / Bound        │ Classification & Operational Role                      │")
    print("├──────────────────────────────────┼──────────────────────┼────────────────────────────────────────────────────────┤")
    print("│ Network Total Capacity           │ 1,200 MW             │ [CASE CONSTRAINT] Fixed installed regional fleet size  │")
    print("│ DC Fast-Charging Share           │ 60% (720 MW)         │ [CASE CONSTRAINT] Direct commercial charging split     │")
    print("│ Battery Swapping Share           │ 40% (480 MW)         │ [CASE CONSTRAINT] Swappable battery capacity split     │")
    print("│ Total Storage Capacity           │ 600 MWh              │ [CASE CONSTRAINT] Distributed BESS & swappable packs   │")
    print("│ Renewable Open-Access Share      │ ~35%                 │ [CASE CONSTRAINT] Long-term solar/wind PPA procurement │")
    print("│ Uncommitted Capacity Reserve     │ >= 5.0% (Hard)       │ [CASE CONSTRAINT] Emergency/priority fleet reservation │")
    print("│ Swap-Pack Inventory Reserve      │ >= 5.0% (Hard)       │ [CASE CONSTRAINT] Ready standby swappable pack buffer  │")
    print("│ Dynamic Retail Tariff Band       │ ±20% of prev day avg │ [CASE CONSTRAINT] Regulatory dynamic pricing corridor  │")
    print("│ Service Reliability Target       │ >= 89.7%             │ [CASE CONSTRAINT] Primary SLA target (vs. 78% baseline)│")
    print("│ Energy Loss Rate Target          │ <= 8.0%              │ [CASE CONSTRAINT] System efficiency (vs. 10% baseline) │")
    print("│ Cluster EBITDA Margin Target     │ > 12.0% (in INR)     │ [CASE CONSTRAINT] Regional cluster commercial viability│")
    print("│ Inter-Metro Physical Transfer    │ DISALLOWED (Isolated)│ [CASE CONSTRAINT] Independent regional nodal buses     │")
    print("│ BESS Converter Roundtrip Eff.    │ 88.4% (94% chg/dis)  │ [ILLUSTRATIVE ASSUMPTION] Standard LFP/NMC parameters │")
    print("│ Swap Module Capacity             │ 5.0 kWh per pack     │ [ILLUSTRATIVE ASSUMPTION] Standard 2W/3W commercial pack│")
    print("│ BESS Degradation Cost            │ ₹0.42 / kWh cycle    │ [ILLUSTRATIVE ASSUMPTION] Battery cell replacement wear│")
    print("│ Open-Access Solar PPA Price      │ ₹3.65 / kWh          │ [ILLUSTRATIVE ASSUMPTION] Long-term solar contract PPA │")
    print("│ Open-Access Wind PPA Price       │ ₹4.15 / kWh          │ [ILLUSTRATIVE ASSUMPTION] Wind wheeling contract PPA   │")
    print("│ Price Elasticity of Demand       │ -0.32                │ [ILLUSTRATIVE ASSUMPTION] Price responsiveness of fleet│")
    print("│ IEX Regulatory Price Ceiling     │ ₹10.00 / kWh         │ [ILLUSTRATIVE ASSUMPTION] CERC DAM spot price limit    │")
    print("└──────────────────────────────────┴──────────────────────┴────────────────────────────────────────────────────────┘")
    print("\n[NOTE]: Synthetic/illustrative simulation metrics require validation on real historical data and finance-approved accounts.")


def main():
    print("=" * 80)
    print(" REGIONAL EV ENERGY OPTIMIZATION & DISPATCH CONTROL TOWER ")
    print(" 1,200 MW EV Network | 600 MWh Storage | 5 Indian Metro Clusters")
    print("=" * 80)
    
    display_network_topology()
    
    print_header("Data Generation Engine Initialization")
    gen = SyntheticDataGenerator(seed=42)
    print("Generating 1-year (8,760 hours) chronological dataset for Delhi NCR...")
    df_delhi = gen.generate_hourly_dataset(cluster_name="Delhi NCR", periods=8760)
    print(f"Generated {len(df_delhi):,} hours of weather, demand, renewable, and IEX market features.")
    
    run_forecasting_backtest(df_delhi)
    run_scenarios_and_benchmarks()
    print_parameter_taxonomy()
    
    print("\n" + "=" * 80)
    print(" PROTOTYPE EXECUTION COMPLETE ")
    print(" [VALIDATION NOTICE]: All synthetic & illustrative metrics require validation")
    print(" on real historical fleet data, utility settlement bills & audited accounts.")
    print("=" * 80)


if __name__ == "__main__":
    main()
