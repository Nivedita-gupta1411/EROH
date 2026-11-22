"""Automated unit test suite verifying mathematical formulations, constraints, and KPIs."""

import unittest
import numpy as np
from core.config import CaseConstraints, CLUSTER_CONFIGS, SimulationAssumptions
from core.data_generator import SyntheticDataGenerator
from core.forecaster import ProbabilisticForecaster
from core.milp_optimizer import ClusterMILPOptimizer
from core.mpc_controller import MPCEngine
from core.scenario_manager import ScenarioManager


class TestEVOptimizationSystem(unittest.TestCase):

    def setUp(self):
        self.cluster = CLUSTER_CONFIGS["Delhi NCR"]
        self.generator = SyntheticDataGenerator(seed=101)

    def test_network_capacity_consistency(self):
        """Verify 1,200 MW total capacity across 5 clusters with 60% DC / 40% Swap split."""
        tot_capacity = sum(c.total_capacity_mw for c in CLUSTER_CONFIGS.values())
        tot_dc = sum(c.dc_capacity_mw for c in CLUSTER_CONFIGS.values())
        tot_swap = sum(c.swap_capacity_mw for c in CLUSTER_CONFIGS.values())
        tot_storage = sum(c.storage_capacity_mwh for c in CLUSTER_CONFIGS.values())

        self.assertAlmostEqual(tot_capacity, CaseConstraints.TOTAL_NETWORK_CAPACITY_MW, places=1)
        self.assertAlmostEqual(tot_dc, CaseConstraints.TOTAL_NETWORK_CAPACITY_MW * 0.60, places=1)
        self.assertAlmostEqual(tot_swap, CaseConstraints.TOTAL_NETWORK_CAPACITY_MW * 0.40, places=1)
        self.assertAlmostEqual(tot_storage, CaseConstraints.TOTAL_STORAGE_CAPACITY_MWH, places=1)

    def test_synthetic_data_generator_ranges(self):
        """Verify physical validity of synthetic weather, demand, renewables, and market prices."""
        df = self.generator.generate_hourly_dataset(cluster_name="Delhi NCR", periods=24 * 30)
        self.assertEqual(len(df), 720)
        
        # Physical bounds
        self.assertTrue((df["irradiance_w_m2"] >= 0.0).all())
        self.assertTrue((df["cloud_cover"] >= 0.0).all() and (df["cloud_cover"] <= 1.0).all())
        self.assertTrue((df["dc_demand_mw"] >= 0.0).all())
        self.assertTrue((df["iex_price_inr_kwh"] >= 2.0).all())
        self.assertTrue((df["iex_price_inr_kwh"] <= SimulationAssumptions.IEX_REGULATORY_PRICE_CAP_INR_KWH).all())
        self.assertTrue((df["bess_soh"] >= 0.70).all() and (df["bess_soh"] <= 1.0).all())

    def test_milp_hard_reserve_and_tariff_constraints(self):
        """Verify hard >= 5% reserves and +/- 20% tariff bounding in MILP solution."""
        df = self.generator.generate_hourly_dataset(cluster_name="Delhi NCR", periods=24)
        optimizer = ClusterMILPOptimizer(self.cluster)
        
        prev_r = 13.50
        res = optimizer.optimize_24h(
            solar_forecast_mw=df["solar_gen_mw"].values,
            wind_forecast_mw=df["wind_gen_mw"].values,
            dc_demand_forecast_mw=df["dc_demand_mw"].values,
            swap_demand_forecast_count=df["swap_demand_count"].values,
            iex_price_forecast_inr_kwh=df["iex_price_inr_kwh"].values,
            prev_day_avg_tariff_inr_kwh=prev_r,
        )

        self.assertTrue(res.success)
        
        # 1. Hard 5% DC capacity reserve check
        min_dc_reserve = CaseConstraints.MIN_UNCOMMITTED_CAPACITY_RESERVE * self.cluster.dc_capacity_mw
        self.assertTrue(np.all(res.p_dc_uncommitted_reserve_mw >= (min_dc_reserve - 1e-4)))

        # 2. Hard 5% Swap pack reserve check
        min_swap_reserve = int(CaseConstraints.MIN_SWAP_PACK_INVENTORY_RESERVE * self.cluster.total_swap_packs)
        self.assertTrue(np.all(res.swap_avail_packs >= (min_swap_reserve - 1)))

        # 3. Dynamic Tariff bounded within +/- 20%
        self.assertTrue(np.all(res.tariff_inr_kwh >= (0.80 * prev_r - 1e-2)))
        self.assertTrue(np.all(res.tariff_inr_kwh <= (1.20 * prev_r + 1e-2)))

        # 4. Simultaneous charge/discharge mutual exclusion
        simultaneous = (res.p_bess_ch_mw > 0.05) & (res.p_bess_dis_mw > 0.05)
        self.assertFalse(np.any(simultaneous))

        # 5. Bus Energy Balance Verification
        loss_oa = SimulationAssumptions.OPEN_ACCESS_WHEELING_LOSS
        loss_grid = SimulationAssumptions.DISCOM_GRID_LOSS_FACTOR
        for t in range(24):
            input_power = (1.0 - loss_oa) * (res.p_solar_mw[t] + res.p_wind_mw[t]) + \
                          (1.0 - loss_grid) * (res.p_discom_mw[t] + res.p_iex_buy_mw[t]) + \
                          res.p_bess_dis_mw[t]
            output_power = res.p_dc_serv_mw[t] + res.p_swap_ch_mw[t] + res.p_bess_ch_mw[t] + res.p_iex_sell_mw[t]
            self.assertAlmostEqual(input_power, output_power, places=1)

    def test_scenarios_execution(self):
        """Verify that all 4 scenarios execute cleanly and compute target metrics."""
        mgr = ScenarioManager(seed=42)
        scenarios = ["normal_weekday", "monsoon_cloudburst", "summer_heatwave", "iex_price_spike"]
        
        for sc_id in scenarios:
            res = mgr.run_scenario("Delhi NCR", sc_id)
            self.assertIn("kpis", res)
            self.assertIn("financials", res)
            self.assertGreater(res["kpis"]["service_reliability"]["actual"], 80.0)
            self.assertLess(res["kpis"]["energy_loss_rate"]["actual"], 9.0)
            self.assertGreater(res["financials"]["ebitda_margin_pct"], 10.0)


if __name__ == "__main__":
    unittest.main()
