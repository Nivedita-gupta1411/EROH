"""Configuration and case constraints for the 1,200 MW Indian EV Charging & Swap Network.

CRITICAL DISTINCTIONS:
1. MW (Power, instantaneous rate of energy transfer) vs. MWh (Energy, cumulative stored volume).
2. Stationary BESS (grid-scale stationary battery energy storage systems, continuous inverter control)
   vs. Swap-Pack Inventory (discrete modular 5.0 kWh battery packs, discrete queuing logistics).
3. Inter-Metro Transmission Boundaries: Distant metro clusters (e.g. Northern Region Delhi vs.
   Southern Region Chennai) are physically isolated on independent regional buses. No unrestricted
   copper-wire wheeling between distant clusters is assumed without inter-regional ATC limits and
   statutory state wheeling/cross-subsidy surcharges.
4. Case Study Hard Mandates (fixed targets, reserve bounds, dynamic pricing corridors)
   vs. Illustrative Simulation Assumptions (techno-economic parameters calibrated to Indian EV standards).

NOTE: All synthetic and illustrative simulation metrics require validation against real-world
historical fleet telematics, actual utility bills, and finance-approved accounts.
"""

from dataclasses import dataclass, field
from typing import Dict, List


VALIDATION_DISCLAIMER: str = (
    "Synthetic/illustrative simulation metrics require validation on real historical data "
    "and finance-approved accounts."
)


@dataclass(frozen=True)
class CaseConstraints:
    """Hard constraints and parameters explicitly mandated by problem specifications.
    These are NON-NEGOTIABLE case parameters.
    """
    
    # Network Capacity Specs (Instantaneous Power in MW)
    TOTAL_NETWORK_CAPACITY_MW: float = 1200.0  # 1,200 MW instantaneous connected capacity
    DC_FAST_CHARGING_SHARE: float = 0.60       # 60% = 720 MW direct vehicle fast-charging
    BATTERY_SWAPPING_SHARE: float = 0.40       # 40% = 480 MW swap station charging bays
    
    # Storage Volume Specs (Cumulative Stored Energy in MWh)
    # Distinct from MW capacity: represents 600 MWh total energy capacity
    TOTAL_STORAGE_CAPACITY_MWH: float = 600.0  # 600 MWh total network energy storage
    
    # Renewable Open-Access Share Mandate
    RENEWABLE_SHARE_TARGET: float = 0.35        # ~35% solar/wind open-access PPA target
    
    # Priority Fleet Hard Reserve Constraints (>= 5.0% Non-Negotiable)
    MIN_UNCOMMITTED_CAPACITY_RESERVE: float = 0.05  # At least 5% DC capacity uncommitted (MW)
    MIN_SWAP_PACK_INVENTORY_RESERVE: float = 0.05   # At least 5% Swap pack inventory reserved (packs)
    
    # Dynamic Pricing Bounds (within +/- 20% of previous day average)
    TARIFF_LOWER_BOUND_FACTOR: float = 0.80   # -20% lower boundary
    TARIFF_UPPER_BOUND_FACTOR: float = 1.20   # +20% upper boundary
    
    # Key Performance Indicator (KPI) Targets
    SERVICE_RELIABILITY_TARGET: float = 89.7   # Target >= 89.7% (vs. 78.0% baseline)
    SERVICE_RELIABILITY_BASELINE: float = 78.0
    
    ENERGY_LOSS_RATE_TARGET: float = 8.0       # Target <= 8.0% (vs. 10.0% baseline)
    ENERGY_LOSS_RATE_BASELINE: float = 10.0
    
    CLUSTER_EBITDA_MARGIN_TARGET: float = 12.0 # Target > 12.0% in INR
    
    # Inter-Metro Transmission Mandate
    ALLOW_UNRESTRICTED_INTER_METRO_TRANSFER: bool = False  # Physical isolation of regional buses


@dataclass
class ClusterSpecs:
    """Cluster-specific capacity and storage allocations."""
    name: str
    code: str
    total_capacity_mw: float
    dc_capacity_mw: float
    swap_capacity_mw: float
    storage_capacity_mwh: float
    stationary_bess_mwh: float
    swap_pack_storage_mwh: float
    total_swap_packs: int
    base_discom_tariff_inr_kwh: float
    peak_discom_multiplier: float
    open_access_solar_mw: float
    open_access_wind_mw: float


# Distribution of 1,200 MW & 600 MWh across the 5 Indian Metro Clusters
CLUSTER_CONFIGS: Dict[str, ClusterSpecs] = {
    "Delhi NCR": ClusterSpecs(
        name="Delhi NCR",
        code="DEL",
        total_capacity_mw=320.0,
        dc_capacity_mw=192.0,
        swap_capacity_mw=128.0,
        storage_capacity_mwh=160.0,
        stationary_bess_mwh=96.0,
        swap_pack_storage_mwh=64.0,
        total_swap_packs=12800,  # 5 kWh per pack
        base_discom_tariff_inr_kwh=6.80,
        peak_discom_multiplier=1.35,
        open_access_solar_mw=95.0,
        open_access_wind_mw=25.0,
    ),
    "Mumbai MMR": ClusterSpecs(
        name="Mumbai MMR",
        code="BOM",
        total_capacity_mw=280.0,
        dc_capacity_mw=168.0,
        swap_capacity_mw=112.0,
        storage_capacity_mwh=140.0,
        stationary_bess_mwh=84.0,
        swap_pack_storage_mwh=56.0,
        total_swap_packs=11200,
        base_discom_tariff_inr_kwh=7.20,
        peak_discom_multiplier=1.40,
        open_access_solar_mw=80.0,
        open_access_wind_mw=30.0,
    ),
    "Bengaluru": ClusterSpecs(
        name="Bengaluru",
        code="BLR",
        total_capacity_mw=240.0,
        dc_capacity_mw=144.0,
        swap_capacity_mw=96.0,
        storage_capacity_mwh=120.0,
        stationary_bess_mwh=72.0,
        swap_pack_storage_mwh=48.0,
        total_swap_packs=9600,
        base_discom_tariff_inr_kwh=6.50,
        peak_discom_multiplier=1.30,
        open_access_solar_mw=75.0,
        open_access_wind_mw=35.0,
    ),
    "Hyderabad": ClusterSpecs(
        name="Hyderabad",
        code="HYD",
        total_capacity_mw=200.0,
        dc_capacity_mw=120.0,
        swap_capacity_mw=80.0,
        storage_capacity_mwh=100.0,
        stationary_bess_mwh=60.0,
        swap_pack_storage_mwh=40.0,
        total_swap_packs=8000,
        base_discom_tariff_inr_kwh=6.40,
        peak_discom_multiplier=1.30,
        open_access_solar_mw=65.0,
        open_access_wind_mw=20.0,
    ),
    "Chennai": ClusterSpecs(
        name="Chennai",
        code="MAA",
        total_capacity_mw=160.0,
        dc_capacity_mw=96.0,
        swap_capacity_mw=64.0,
        storage_capacity_mwh=80.0,
        stationary_bess_mwh=48.0,
        swap_pack_storage_mwh=32.0,
        total_swap_packs=6400,
        base_discom_tariff_inr_kwh=6.60,
        peak_discom_multiplier=1.32,
        open_access_solar_mw=50.0,
        open_access_wind_mw=30.0,
    ),
}


@dataclass(frozen=True)
class SimulationAssumptions:
    """Illustrative operational parameters calibrated to Indian grid & EV market standards."""
    
    # Battery Technical Parameters
    PACK_CAPACITY_KWH: float = 5.0             # 5.0 kWh standard swappable module
    PACK_CHARGING_POWER_KW: float = 2.5        # Normal 0.5C charging to preserve SoH
    BESS_CHARGE_EFFICIENCY: float = 0.94       # Converter + Cell charge efficiency
    BESS_DISCHARGE_EFFICIENCY: float = 0.94    # Converter + Cell discharge efficiency
    BESS_ROUNDTRIP_EFFICIENCY: float = 0.8836  # ~88.4% roundtrip
    BESS_MIN_SOC: float = 0.10                 # Depth of discharge cushion
    BESS_MAX_SOC: float = 0.95                 # Overcharge protection
    BESS_C_RATE_MAX: float = 0.50              # Max charge/discharge rate (0.5C)
    BESS_DEGRADATION_COST_INR_PER_KWH: float = 0.42 # Battery wear replacement amortized
    
    # Transmission & Distribution Loss Assumptions
    DISCOM_GRID_LOSS_FACTOR: float = 0.030     # 3.0% substation step-down distribution loss
    OPEN_ACCESS_WHEELING_LOSS: float = 0.032   # 3.2% open-access state wheeling loss
    CHARGER_CONVERSION_LOSS: float = 0.025     # 2.5% SiC high-efficiency fast-charger loss
    
    # Financial & Market Assumptions (in INR)
    OPEN_ACCESS_SOLAR_PPA_INR_KWH: float = 3.65
    OPEN_ACCESS_WIND_PPA_INR_KWH: float = 4.15
    IEX_TRANSACTION_FEE_INR_KWH: float = 0.02
    IEX_REGULATORY_PRICE_CAP_INR_KWH: float = 10.00 # CERC Price Ceiling
    
    # Dynamic Pricing & Demand Elasticity
    DEFAULT_PREV_DAY_AVG_TARIFF_INR_KWH: float = 13.50 # Benchmark customer retail tariff
    PRICE_ELASTICITY_OF_DEMAND: float = -0.32          # Elastic response coefficient
    UNSERVED_ENERGY_PENALTY_INR_KWH: float = 35.00     # Shadow value of lost load
    RESERVE_VIOLATION_PENALTY_INR_KWH: float = 50.00   # Critical fleet service SLA penalty
