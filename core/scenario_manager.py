"""Scenario generator and evaluation manager for the 5 Indian metro clusters.

Scenarios tested:
1. Normal Weekday (baseline operating conditions, standard renewable diurnal cycle)
2. Monsoon Cloudburst (sudden solar drop by ~80%, wind gusts, potential grid stress)
3. Summer Heatwave (extreme temperature 44-46°C, AC load surge, thermal battery derating)
4. IEX Price Spike (market clearing price spikes up to CERC cap ₹10.00/kWh during peak hours)
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from core.config import CLUSTER_CONFIGS, CaseConstraints, SimulationAssumptions, ClusterSpecs
from core.data_generator import SyntheticDataGenerator
from core.milp_optimizer import ClusterMILPOptimizer, ScheduleResult


@dataclass
class ScenarioDefinition:
    id: str
    name: str
    description: str
    weather_condition: str
    grid_condition: str
    key_challenge: str


SCENARIO_CATALOG: Dict[str, ScenarioDefinition] = {
    "normal_weekday": ScenarioDefinition(
        id="normal_weekday",
        name="Normal Weekday",
        description="Standard operating conditions with typical diurnal solar, moderate wind, and baseline grid tariffs.",
        weather_condition="Clear to partly cloudy, ambient temperature 28-34°C, normal irradiance.",
        grid_condition="Standard Time-of-Day DISCOM tariff, normal IEX day-ahead market clearing prices (₹3.5 - ₹6.5/kWh).",
        key_challenge="Optimize economic dispatch between open-access renewables and arbitrage BESS across morning/evening peaks."
    ),
    "monsoon_cloudburst": ScenarioDefinition(
        id="monsoon_cloudburst",
        name="Monsoon Cloudburst",
        description="Severe sudden cloudburst during midday causing ~80% solar generation collapse and heightened EV dwell times.",
        weather_condition="Heavy precipitation, 92% cloud cover, irradiance drops to <180 W/m², gusty winds.",
        grid_condition="DISCOM voltage swings, DISCOM wheeling active, moderate IEX prices.",
        key_challenge="Prevent unserved energy SLA violations when open-access solar plummets; dynamically throttle and dispatch BESS."
    ),
    "summer_heatwave": ScenarioDefinition(
        id="summer_heatwave",
        name="Summer Heatwave",
        description="Extreme ambient heat (up to 46°C) increasing EV cabin cooling load by 22% and derating battery roundtrip efficiency.",
        weather_condition="Extreme heatwave, peak temperature 45.8°C, high clear sky irradiance, calm wind.",
        grid_condition="High regional grid stress, elevated peak DISCOM tariffs, tight reserve margins.",
        key_challenge="Manage battery thermal degradation while supplying surge DC fast-charging demand within 5% uncommitted reserve."
    ),
    "iex_price_spike": ScenarioDefinition(
        id="iex_price_spike",
        name="IEX Price Spike",
        description="Regional thermal generation trip causing Day-Ahead Market prices to surge to regulatory ceiling (₹10.00/kWh).",
        weather_condition="Normal clear weather conditions.",
        grid_condition="IEX DAM prices hit CERC regulatory price cap (₹10.00/kWh) during evening ramp hours (17:00-22:00).",
        key_challenge="Minimize procurement costs by discharging stored solar energy and adjusting dynamic customer tariffs within ±20% band."
    ),
}


class ScenarioManager:
    """Manages synthetic scenario injection, MILP optimization, and KPI evaluation."""

    def __init__(self, seed: int = 42):
        self.data_gen = SyntheticDataGenerator(seed=seed)

    def generate_scenario_data(self, cluster_name: str, scenario_id: str) -> pd.DataFrame:
        """Create 24-hour deterministic scenario profile for a specific metro cluster."""
        cluster: ClusterSpecs = CLUSTER_CONFIGS[cluster_name]
        
        # Base 24-hour profile from generator
        base_df = self.data_gen.generate_hourly_dataset(
            start_date="2025-05-15" if scenario_id == "summer_heatwave" else "2025-07-20",
            periods=24,
            cluster_name=cluster_name
        )
        
        hours = base_df["hour"].values
        df = base_df.copy()

        if scenario_id == "normal_weekday":
            # Calibrated baseline
            pass

        elif scenario_id == "monsoon_cloudburst":
            # Cloudburst hits between 10:00 and 16:00
            monsoon_hours = (hours >= 10) & (hours <= 16)
            df.loc[monsoon_hours, "cloud_cover"] = 0.94
            df.loc[monsoon_hours, "irradiance_w_m2"] *= 0.18
            df.loc[monsoon_hours, "solar_gen_mw"] *= 0.18
            # Wind speed gust
            df.loc[monsoon_hours, "wind_speed_m_s"] = np.clip(df.loc[monsoon_hours, "wind_speed_m_s"] * 1.6, 2.0, 18.0)
            df.loc[monsoon_hours, "wind_gen_mw"] = np.clip(cluster.open_access_wind_mw * 0.75, 0.0, cluster.open_access_wind_mw)
            df["renewable_gen_mw"] = df["solar_gen_mw"] + df["wind_gen_mw"]

        elif scenario_id == "summer_heatwave":
            # Extreme afternoon temperatures (up to 46C)
            heat_wave_boost = 10.0 * np.exp(-((hours - 14.5) ** 2) / 12.0)
            df["ambient_temp_c"] = np.clip(36.0 + heat_wave_boost, 32.0, 46.2)
            # AC usage in EV fleets increases DC energy demand by ~22%
            df["dc_demand_mw"] = np.clip(df["dc_demand_mw"] * 1.22, 0.0, cluster.dc_capacity_mw * 0.94)
            df["total_demand_mw"] = df["dc_demand_mw"] + df["swap_demand_mwh"]

        elif scenario_id == "iex_price_spike":
            # Evening spike to regulatory ceiling ₹10.00/kWh (17:00 to 22:00)
            evening_spike = (hours >= 17) & (hours <= 22)
            df.loc[evening_spike, "iex_price_inr_kwh"] = SimulationAssumptions.IEX_REGULATORY_PRICE_CAP_INR_KWH
            df.loc[~evening_spike, "iex_price_inr_kwh"] = np.clip(df.loc[~evening_spike, "iex_price_inr_kwh"] * 1.15, 3.2, 7.5)

        return df

    def run_scenario(self, cluster_name: str, scenario_id: str) -> Dict[str, Any]:
        """Execute full MILP optimization for scenario and evaluate against targets."""
        cluster = CLUSTER_CONFIGS[cluster_name]
        scenario_def = SCENARIO_CATALOG[scenario_id]
        scenario_data = self.generate_scenario_data(cluster_name, scenario_id)
        
        optimizer = ClusterMILPOptimizer(cluster)
        is_heatwave = (scenario_id == "summer_heatwave")
        
        result: ScheduleResult = optimizer.optimize_24h(
            solar_forecast_mw=scenario_data["solar_gen_mw"].values,
            wind_forecast_mw=scenario_data["wind_gen_mw"].values,
            dc_demand_forecast_mw=scenario_data["dc_demand_mw"].values,
            swap_demand_forecast_count=scenario_data["swap_demand_count"].values,
            iex_price_forecast_inr_kwh=scenario_data["iex_price_inr_kwh"].values,
            prev_day_avg_tariff_inr_kwh=SimulationAssumptions.DEFAULT_PREV_DAY_AVG_TARIFF_INR_KWH,
            is_heatwave=is_heatwave,
        )

        # Build KPI Scorecard with Target vs. Baseline vs. Actual
        kpis = {
            "service_reliability": {
                "metric": "Service Reliability",
                "unit": "%",
                "actual": result.service_reliability_pct,
                "target": CaseConstraints.SERVICE_RELIABILITY_TARGET,
                "baseline": CaseConstraints.SERVICE_RELIABILITY_BASELINE,
                "status": "PASS" if result.service_reliability_pct >= CaseConstraints.SERVICE_RELIABILITY_TARGET else "WARN",
                "label_type": "Case Constraint (>= 89.7%)"
            },
            "energy_loss_rate": {
                "metric": "Energy Loss Rate",
                "unit": "%",
                "actual": result.energy_loss_rate_pct,
                "target": CaseConstraints.ENERGY_LOSS_RATE_TARGET,
                "baseline": CaseConstraints.ENERGY_LOSS_RATE_BASELINE,
                "status": "PASS" if result.energy_loss_rate_pct <= CaseConstraints.ENERGY_LOSS_RATE_TARGET else "WARN",
                "label_type": "Case Constraint (<= 8.0%)"
            },
            "ebitda_margin": {
                "metric": "Cluster EBITDA Margin",
                "unit": "%",
                "actual": result.ebitda_margin_pct,
                "target": CaseConstraints.CLUSTER_EBITDA_MARGIN_TARGET,
                "baseline": 8.5,
                "status": "PASS" if result.ebitda_margin_pct > CaseConstraints.CLUSTER_EBITDA_MARGIN_TARGET else "WARN",
                "label_type": "Case Constraint (> 12.0% in INR)"
            },
            "uncommitted_reserve": {
                "metric": "5% Uncommitted Reserve Compliance",
                "unit": "%",
                "actual": result.reserve_compliance_pct,
                "target": 100.0,
                "baseline": 85.0,
                "status": "PASS" if result.reserve_compliance_pct >= 99.0 else "WARN",
                "label_type": "Case Constraint (Hard >= 5% DC capacity & Swap packs)"
            }
        }

        # Energy Mix Breakdown
        tot_solar = float(np.sum(result.p_solar_mw))
        tot_wind = float(np.sum(result.p_wind_mw))
        tot_discom = float(np.sum(result.p_discom_mw))
        tot_iex = float(np.sum(result.p_iex_buy_mw))
        tot_gen = tot_solar + tot_wind + tot_discom + tot_iex
        renewable_share_pct = round(((tot_solar + tot_wind) / (tot_gen + 1e-6)) * 100.0, 1)

        summary = {
            "cluster": cluster.name,
            "scenario": {
                "id": scenario_def.id,
                "name": scenario_def.name,
                "description": scenario_def.description,
                "weather": scenario_def.weather_condition,
                "grid": scenario_def.grid_condition,
                "challenge": scenario_def.key_challenge,
            },
            "kpis": kpis,
            "financials": {
                "total_revenue_inr": result.total_revenue_inr,
                "procurement_cost_inr": result.total_procurement_cost_inr,
                "degradation_cost_inr": result.total_degradation_cost_inr,
                "ebitda_inr": result.ebitda_inr,
                "ebitda_margin_pct": result.ebitda_margin_pct,
            },
            "energy_summary": {
                "total_solar_mwh": round(tot_solar, 2),
                "total_wind_mwh": round(tot_wind, 2),
                "total_discom_mwh": round(tot_discom, 2),
                "total_iex_buy_mwh": round(tot_iex, 2),
                "total_energy_procured_mwh": round(tot_gen, 2),
                "renewable_share_pct": renewable_share_pct,
                "renewable_target_pct": CaseConstraints.RENEWABLE_SHARE_TARGET * 100.0,
            },
            "hourly_data": {
                "hour": list(range(24)),
                "p_solar": result.p_solar_mw.tolist(),
                "p_wind": result.p_wind_mw.tolist(),
                "p_discom": result.p_discom_mw.tolist(),
                "p_iex_buy": result.p_iex_buy_mw.tolist(),
                "p_iex_sell": result.p_iex_sell_mw.tolist(),
                "p_bess_ch": result.p_bess_ch_mw.tolist(),
                "p_bess_dis": result.p_bess_dis_mw.tolist(),
                "bess_soc_pct": result.bess_soc_pct.tolist(),
                "p_swap_ch": result.p_swap_ch_mw.tolist(),
                "swap_satisfied": result.swap_satisfied_count.tolist(),
                "swap_avail_packs": result.swap_avail_packs.tolist(),
                "p_dc_serv": result.p_dc_serv_mw.tolist(),
                "p_dc_demanded": scenario_data["dc_demand_mw"].tolist(),
                "dc_uncommitted_mw": result.p_dc_uncommitted_reserve_mw.tolist(),
                "tariff_inr_kwh": result.tariff_inr_kwh.tolist(),
                "iex_price_inr_kwh": scenario_data["iex_price_inr_kwh"].tolist(),
                "ambient_temp_c": scenario_data["ambient_temp_c"].tolist(),
            }
        }
        return summary
