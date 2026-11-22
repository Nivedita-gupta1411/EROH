"""Chronological rolling-origin backtesting and probabilistic quantile forecasting (P10/P50/P90).

Evaluates load, renewable generation, and IEX price forecasts using WAPE, MAE, and Pinball Loss.
Strictly non-shuffled, expanding/rolling-origin evaluation prevents future data leakage.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class ForecastMetrics:
    """Evaluation metrics for quantile predictions."""
    target_name: str
    mae_p50: float
    wape_p50: float
    pinball_p10: float
    pinball_p50: float
    pinball_p90: float
    coverage_80: float  # Percentage of true observations falling inside P10-P90 band
    sample_count: int


class ProbabilisticForecaster:
    """Rolling-origin probabilistic forecaster utilizing HistGradientBoosting with quantile loss."""

    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)):
        self.quantiles = quantiles
        self.feature_cols = [
            "hour", "day_of_week", "month", "is_weekend",
            "ambient_temp_c", "cloud_cover", "irradiance_w_m2", "wind_speed_m_s",
            "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24"
        ]

    def _engineer_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create non-leaking lag and rolling features chronologically."""
        data = df.copy().sort_values("timestamp").reset_index(drop=True)
        
        # Lag features (strictly historical: at least 24 hours prior)
        data["lag_24"] = data[target_col].shift(24)
        data["lag_168"] = data[target_col].shift(168)  # 1 week lag
        data["rolling_mean_24"] = data[target_col].shift(24).rolling(window=24, min_periods=1).mean()
        data["rolling_std_24"] = data[target_col].shift(24).rolling(window=24, min_periods=1).std().fillna(0)
        
        # Backfill initial window warm-up cleanly
        data["lag_24"] = data["lag_24"].bfill()
        data["lag_168"] = data["lag_168"].bfill()
        data["rolling_mean_24"] = data["rolling_mean_24"].bfill()
        data["rolling_std_24"] = data["rolling_std_24"].bfill()
        
        return data

    @staticmethod
    def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Weighted Absolute Percentage Error (%) = sum(|y - y_hat|) / sum(y) * 100."""
        denom = np.sum(np.abs(y_true))
        if denom < 1e-6:
            return 0.0
        return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
        """Pinball loss for quantile q."""
        diff = y_true - y_pred
        return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))

    def rolling_origin_backtest(
        self,
        df: pd.DataFrame,
        target_col: str,
        initial_train_hours: int = 24 * 90,  # 90 days warm-up
        step_hours: int = 24 * 7,            # step forward 1 week at a time
        horizon_hours: int = 24,             # 24-hour day-ahead horizon
        max_origins: int = 6,                # number of rolling origins
    ) -> Tuple[ForecastMetrics, pd.DataFrame]:
        """Execute chronological rolling-origin backtesting without shuffling or data leakage."""
        data = self._engineer_features(df, target_col)
        
        all_forecasts = []
        n_total = len(data)
        origins = []
        current_origin = initial_train_hours
        while current_origin + horizon_hours <= n_total and len(origins) < max_origins:
            origins.append(current_origin)
            current_origin += step_hours
            
        for origin in origins:
            train_data = data.iloc[:origin]
            test_data = data.iloc[origin : origin + horizon_hours]
            
            X_train = train_data[self.feature_cols]
            y_train = train_data[target_col].values
            X_test = test_data[self.feature_cols]
            y_test = test_data[target_col].values
            
            origin_preds = {"timestamp": test_data["timestamp"].values, "actual": y_test, "origin": origin}
            
            # Train quantile models
            for q in self.quantiles:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=60,
                    learning_rate=0.08,
                    random_state=42,
                    min_samples_leaf=20,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                # Enforce physical non-negativity where applicable
                if "mw" in target_col or "count" in target_col or "price" in target_col:
                    preds = np.maximum(0.0, preds)
                q_key = f"p{int(q * 100)}"
                origin_preds[q_key] = preds
                
            all_forecasts.append(pd.DataFrame(origin_preds))
            
        results_df = pd.concat(all_forecasts, ignore_index=True)
        
        # Enforce quantile monotonicity: P10 <= P50 <= P90
        results_df["p10"] = np.minimum(results_df["p10"], results_df["p50"])
        results_df["p90"] = np.maximum(results_df["p90"], results_df["p50"])
        
        y_act = results_df["actual"].values
        y_p50 = results_df["p50"].values
        y_p10 = results_df["p10"].values
        y_p90 = results_df["p90"].values
        
        mae = self.calculate_mae(y_act, y_p50)
        wape = self.calculate_wape(y_act, y_p50)
        pinball_10 = self.pinball_loss(y_act, y_p10, 0.10)
        pinball_50 = self.pinball_loss(y_act, y_p50, 0.50)
        pinball_90 = self.pinball_loss(y_act, y_p90, 0.90)
        coverage = float(np.mean((y_act >= y_p10) & (y_act <= y_p90)) * 100.0)
        
        metrics = ForecastMetrics(
            target_name=target_col,
            mae_p50=round(mae, 3),
            wape_p50=round(wape, 2),
            pinball_p10=round(pinball_10, 3),
            pinball_p50=round(pinball_50, 3),
            pinball_p90=round(pinball_90, 3),
            coverage_80=round(coverage, 2),
            sample_count=len(results_df)
        )
        
        return metrics, results_df

    def train_and_forecast_24h(
        self,
        historical_df: pd.DataFrame,
        target_col: str,
        future_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate 24-hour ahead P10/P50/P90 probabilistic forecast for online MPC scheduling."""
        combined = pd.concat([historical_df, future_features_df], ignore_index=True)
        engineered = self._engineer_features(combined, target_col)
        
        n_hist = len(historical_df)
        train_data = engineered.iloc[:n_hist]
        future_data = engineered.iloc[n_hist:]
        
        X_train = train_data[self.feature_cols]
        y_train = train_data[target_col].values
        X_future = future_data[self.feature_cols]
        
        preds_dict = {"timestamp": future_features_df["timestamp"].values}
        for q in self.quantiles:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=50,
                learning_rate=0.08,
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_future)
            if "mw" in target_col or "count" in target_col or "price" in target_col:
                pred = np.maximum(0.0, pred)
            preds_dict[f"p{int(q * 100)}"] = pred
            
        res = pd.DataFrame(preds_dict)
        res["p10"] = np.minimum(res["p10"], res["p50"])
        res["p90"] = np.maximum(res["p90"], res["p50"])
        return res
