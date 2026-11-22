"""Synthetic 5-year / hourly-style data generator for the Indian EV energy network.

Generates realistic hourly data across the 5 metro clusters:
- Weather: GHI irradiance, cloud cover, wind speed, ambient temperature
- EV Demand: DC fast charging load and Battery Swapping requests
- Renewables: Solar PV and Wind turbine actual generation
- Power Market: IEX Day-Ahead Market (DAM) hourly clearing prices
- Battery State: State of Charge (SoC) and State of Health (SoH) degradation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from core.config import CLUSTER_CONFIGS, ClusterSpecs, SimulationAssumptions


class SyntheticDataGenerator:
    """Generates multi-year hourly time series data capturing Indian regional climatic & grid dynamics."""
    
    METRO_CLIMATE = {
        "Delhi NCR": {"t_base": 25.0, "t_amp_season": 16.0, "t_amp_diurnal": 7.0, "monsoon_intensity": 0.55, "wind_mean": 4.2},
        "Mumbai MMR": {"t_base": 28.0, "t_amp_season": 4.0, "t_amp_diurnal": 4.5, "monsoon_intensity": 0.85, "wind_mean": 5.8},
        "Bengaluru": {"t_base": 23.5, "t_amp_season": 5.5, "t_amp_diurnal": 6.5, "monsoon_intensity": 0.60, "wind_mean": 6.5},
        "Hyderabad": {"t_base": 27.0, "t_amp_season": 8.0, "t_amp_diurnal": 7.0, "monsoon_intensity": 0.58, "wind_mean": 5.0},
        "Chennai": {"t_base": 29.0, "t_amp_season": 5.0, "t_amp_diurnal": 5.0, "monsoon_intensity": 0.70, "wind_mean": 6.2},
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_hourly_dataset(
        self,
        start_date: str = "2022-01-01",
        periods: int = 24 * 365 * 2,  # 2 years default (expandable to 5 years: 43800 hrs)
        cluster_name: str = "Delhi NCR",
    ) -> pd.DataFrame:
        """Generate comprehensive chronological hourly dataset for a given metro cluster."""
        cluster: ClusterSpecs = CLUSTER_CONFIGS[cluster_name]
        climate = self.METRO_CLIMATE.get(cluster_name, self.METRO_CLIMATE["Delhi NCR"])
        
        timestamps = pd.date_range(start=start_date, periods=periods, freq="h")
        n = len(timestamps)
        
        day_of_year = timestamps.dayofyear.values
        hour = timestamps.hour.values
        day_of_week = timestamps.dayofweek.values
        month = timestamps.month.values
        is_weekend = (day_of_week >= 5).astype(float)
        
        # 1. Weather Dynamics
        # Temperature: Seasonal sine wave + Diurnal wave + random fluctuations
        # Northern Indian peak summer in May/June (day ~140-170)
        temp_seasonal = climate["t_base"] + climate["t_amp_season"] * np.sin(2 * np.pi * (day_of_year - 105) / 365)
        temp_diurnal = climate["t_amp_diurnal"] * np.sin(2 * np.pi * (hour - 9) / 24)
        ambient_temp = np.clip(
            temp_seasonal + temp_diurnal + self.rng.normal(0, 1.8, n),
            5.0, 48.0
        )
        
        # Cloud Cover: Heavy during Southwest Monsoon (June to Sept, months 6-9) & NE monsoon for Chennai (Oct-Nov)
        if cluster_name == "Chennai":
            monsoon_mask = (month >= 9) & (month <= 11)
        else:
            monsoon_mask = (month >= 6) & (month <= 9)
            
        base_cloud = np.where(monsoon_mask, 0.65 * climate["monsoon_intensity"], 0.15)
        cloud_noise = self.rng.beta(2, 5, n) * 0.4
        cloud_cover = np.clip(base_cloud + cloud_noise + self.rng.normal(0, 0.08, n), 0.0, 1.0)
        
        # Solar Irradiance (GHI, W/m^2): Bell curve between 06:00 and 18:00 modulated by cloud cover
        solar_elevation = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12))
        solar_elevation = np.where((hour >= 6) & (hour <= 18), solar_elevation, 0.0)
        clear_sky_ghi = 980.0 * (solar_elevation ** 1.15)
        irradiance = clear_sky_ghi * (1.0 - 0.78 * (cloud_cover ** 1.3))
        irradiance = np.clip(irradiance + self.rng.normal(0, 15.0, n), 0.0, 1050.0)
        irradiance = np.where((hour < 6) | (hour > 18), 0.0, irradiance)
        
        # Wind Speed (m/s): Weibull-like distribution, higher in monsoon & afternoons
        diurnal_wind = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 12) / 24)
        monsoon_wind_boost = np.where(monsoon_mask, 1.35, 1.0)
        wind_speed = self.rng.weibull(2.1, n) * climate["wind_mean"] * diurnal_wind * monsoon_wind_boost
        wind_speed = np.clip(wind_speed, 0.5, 24.0)

        # 2. Renewable Generation (MW)
        # Solar PV: Open-access capacity * derated irradiance * thermal loss (-0.4%/degC above 25degC)
        thermal_derate = np.maximum(0.80, 1.0 - 0.004 * np.maximum(0.0, ambient_temp - 25.0))
        solar_gen = cluster.open_access_solar_mw * (irradiance / 1000.0) * thermal_derate * 0.88
        solar_gen = np.clip(solar_gen, 0.0, cluster.open_access_solar_mw)
        
        # Wind Generation: Power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s)
        wind_cf = np.zeros(n)
        mask_wind_ramp = (wind_speed >= 3.0) & (wind_speed < 12.0)
        mask_wind_rated = (wind_speed >= 12.0) & (wind_speed <= 25.0)
        wind_cf[mask_wind_ramp] = ((wind_speed[mask_wind_ramp] - 3.0) / (12.0 - 3.0)) ** 3
        wind_cf[mask_wind_rated] = 1.0
        wind_gen = cluster.open_access_wind_mw * wind_cf * 0.92  # 92% availability/wake efficiency
        
        # 3. EV Charging Demand
        # DC Fast Charging Demand (MW): Twin peaks (08:00-11:00 morning office & ride-hail, 17:00-21:00 evening return)
        dc_diurnal = (
            0.25
            + 0.40 * np.exp(-((hour - 9.5) ** 2) / 7.0)
            + 0.55 * np.exp(-((hour - 19.0) ** 2) / 8.0)
            + 0.15 * np.exp(-((hour - 14.0) ** 2) / 10.0)
        )
        weekend_damp = np.where(is_weekend == 1, 0.82, 1.0)
        dc_demand = cluster.dc_capacity_mw * dc_diurnal * weekend_damp * (1.0 + self.rng.normal(0, 0.06, n))
        dc_demand = np.clip(dc_demand, cluster.dc_capacity_mw * 0.08, cluster.dc_capacity_mw * 0.88)
        
        # Battery Swapping Demand (Swaps per hour): High morning commercial shift start + late evening shift return
        swap_diurnal = (
            0.20
            + 0.48 * np.exp(-((hour - 8.0) ** 2) / 6.0)
            + 0.52 * np.exp(-((hour - 18.5) ** 2) / 7.0)
            + 0.25 * np.exp(-((hour - 13.0) ** 2) / 8.0)
        )
        max_hourly_swaps = cluster.total_swap_packs * 0.28  # up to 28% of fleet swapped at peak hour
        swap_demand_count = np.round(max_hourly_swaps * swap_diurnal * (1.0 + self.rng.normal(0, 0.05, n))).astype(int)
        swap_demand_count = np.clip(swap_demand_count, int(cluster.total_swap_packs * 0.02), int(max_hourly_swaps))
        # Equivalent swap energy demand in MW (each pack 5 kWh = 0.005 MWh)
        swap_demand_mwh = swap_demand_count * (SimulationAssumptions.PACK_CAPACITY_KWH / 1000.0)
        
        # 4. IEX Day-Ahead Market (DAM) Clearing Prices (INR/kWh)
        # Low price during solar peak (duck curve effect: 11:00-15:00)
        # High price during morning ramp (06:00-09:00) and sharp evening peak (18:00-22:00)
        iex_base = 4.80
        iex_diurnal = (
            0.85
            + 0.45 * np.exp(-((hour - 7.5) ** 2) / 5.0)
            + 0.85 * np.exp(-((hour - 20.0) ** 2) / 6.0)
            - 0.35 * np.exp(-((hour - 13.0) ** 2) / 6.0) # solar depression
        )
        # Seasonal summer price rise (April-June)
        summer_factor = np.where((month >= 4) & (month <= 6), 1.25, 1.0)
        iex_price = iex_base * iex_diurnal * summer_factor + self.rng.normal(0, 0.45, n)
        
        # Occasional price spikes (2% probability) up to CERC regulatory cap (₹10/kWh)
        spike_mask = self.rng.uniform(0, 1, n) < 0.02
        iex_price = np.where(spike_mask, np.minimum(10.0, iex_price * self.rng.uniform(1.4, 2.0)), iex_price)
        iex_price = np.clip(iex_price, 2.20, SimulationAssumptions.IEX_REGULATORY_PRICE_CAP_INR_KWH)
        
        # 5. Battery Degradation State (Cumulative SoH & baseline SoC)
        # 5-year calendar + cycle fade down to ~82% SoH
        cumulative_hours = np.arange(n)
        calendar_fade = 0.035 * (cumulative_hours / (24 * 365))  # ~3.5% per year
        cycle_fade = 0.015 * (np.cumsum(dc_demand) / (cluster.dc_capacity_mw * 24 * 365))
        soh = np.clip(1.0 - calendar_fade - cycle_fade, 0.75, 1.0)
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "cluster": cluster_name,
            "ambient_temp_c": np.round(ambient_temp, 2),
            "cloud_cover": np.round(cloud_cover, 3),
            "irradiance_w_m2": np.round(irradiance, 1),
            "wind_speed_m_s": np.round(wind_speed, 2),
            "solar_gen_mw": np.round(solar_gen, 3),
            "wind_gen_mw": np.round(wind_gen, 3),
            "renewable_gen_mw": np.round(solar_gen + wind_gen, 3),
            "dc_demand_mw": np.round(dc_demand, 3),
            "swap_demand_count": swap_demand_count,
            "swap_demand_mwh": np.round(swap_demand_mwh, 3),
            "total_demand_mw": np.round(dc_demand + swap_demand_mwh, 3),
            "iex_price_inr_kwh": np.round(iex_price, 3),
            "bess_soh": np.round(soh, 4),
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend.astype(int),
        })
        return df

    def generate_all_clusters(self, start_date: str = "2022-01-01", periods: int = 24 * 365) -> Dict[str, pd.DataFrame]:
        """Generate time series for all 5 metro clusters."""
        return {
            name: self.generate_hourly_dataset(start_date=start_date, periods=periods, cluster_name=name)
            for name in CLUSTER_CONFIGS.keys()
        }
