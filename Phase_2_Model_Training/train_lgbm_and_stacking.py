import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "cpi_headline_yoy"
PERSISTENCE = ["cpi_headline_yoy_lag1", "cpi_headline_yoy_lag12"]
MACRO = ["USA_CPI", "iip_general_yoy", "m3_money_supply_yoy", "rbi_repo_rate_yoy_diff"]
COMMODITY = [
    "wpi_primary_food_articles_yoy", "wpi_manufactured_food_yoy", "wpi_wheat_yoy",
    "wpi_rice_yoy", "wpi_maize_yoy", "wpi_gram_chana_yoy", "wpi_arhar_tur_yoy",
    "wpi_moong_yoy", "wpi_urad_yoy", "wpi_masur_yoy", "wpi_tomato_yoy",
    "wpi_onion_yoy", "wpi_potato_yoy", "wpi_brinjal_yoy", "wpi_cabbage_yoy",
    "wpi_garlic_yoy", "wpi_mustard_oil_yoy", "wpi_soybean_oil_yoy",
    "wpi_groundnut_oil_yoy", "wpi_milk_yoy", "wpi_egg_yoy", "wpi_sugar_yoy",
    "crude_oil_brent_usd_yoy", "palm_oil_usd_yoy", "global_fertilizer_index_yoy"
]
CLIMATE_RAW = [
    "rainfall_total_mm", "rainfall_rate_mm_per_day", "temp_mean_celsius",
    "relative_humidity_2m_pct", "surface_soil_wetness_index", "wind_speed_10m_ms", "solar_irradiance_kwm2"
]
CLIMATE = CLIMATE_RAW + [f"{c}_anom" for c in CLIMATE_RAW]

CONFIGS = {
    "Full": CLIMATE + MACRO + COMMODITY + PERSISTENCE,
    "Climate_only": CLIMATE + PERSISTENCE,
    "Macro_only": MACRO + PERSISTENCE,
    "Commodity_only": COMMODITY + PERSISTENCE
}


# TimeSeriesStackingRegressor Class for persistent serialization and clone compatibility
class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Time-Series Stacked Ensemble combining ElasticNet, XGBoost, and LightGBM
    using Out-Of-Fold Chronological Validation predictions to train a non-negative Ridge Meta-Learner.
    Inherits from BaseEstimator and RegressorMixin for compatibility with scikit-learn clone and SHAP.
    """
    def __init__(self, val_ratio=0.25):
        self.val_ratio = val_ratio
        self.m_en = Pipeline([("scaler", StandardScaler()), ("en", ElasticNet(alpha=0.1, l1_ratio=1.0, random_state=42))])
        self.m_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42)
        self.m_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42, n_jobs=1, verbose=-1)
        self.meta = Ridge(alpha=1.0, positive=True)
        self.feature_names_in_ = None

    def get_params(self, deep=True):
        return {"val_ratio": self.val_ratio}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
        
        split_idx = int(len(X_df) * (1.0 - self.val_ratio))
        X_subtr, y_subtr = X_df.iloc[:split_idx], y_ser.iloc[:split_idx]
        X_val, y_val = X_df.iloc[split_idx:], y_ser.iloc[split_idx:]
        
        # Fit base models on sub-train
        self.m_en.fit(X_subtr, y_subtr)
        self.m_xgb.fit(X_subtr, y_subtr)
        self.m_lgb.fit(X_subtr, y_subtr)
        
        # Predict on validation partition
        val_preds = np.column_stack([
            self.m_en.predict(X_subtr.iloc[:0]) if len(X_val) == 0 else self.m_en.predict(X_val),
            self.m_xgb.predict(X_subtr.iloc[:0]) if len(X_val) == 0 else self.m_xgb.predict(X_val),
            self.m_lgb.predict(X_subtr.iloc[:0]) if len(X_val) == 0 else self.m_lgb.predict(X_val)
        ])
        self.meta.fit(val_preds, y_val)
        
        # Refit base models on full training data
        self.m_en.fit(X_df, y_ser)
        self.m_xgb.fit(X_df, y_ser)
        self.m_lgb.fit(X_df, y_ser)
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame) and getattr(self, "feature_names_in_", None) is not None:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        elif isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X)
        
        base_preds = np.column_stack([
            self.m_en.predict(X_df),
            self.m_xgb.predict(X_df),
            self.m_lgb.predict(X_df)
        ])
        return self.meta.predict(base_preds)


if __name__ == "__main__":
    print("==================================================")
    print("PHASE 2 (ML STREAMLINING): STACKING ENSEMBLE TRAINING")
    print("==================================================")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))

    # Train Stacking Regressor across 4 configurations
    for config_name in ["Full", "Commodity_only", "Macro_only", "Climate_only"]:
        cols = [c for c in CONFIGS[config_name] if c in train_df.columns]
        X_tr, y_tr = train_df[cols], train_df[TARGET]
        X_te, y_te = test_df[cols], test_df[TARGET]
        
        stacker = TimeSeriesStackingRegressor()
        stacker.fit(X_tr, y_tr)
        pred_te = stacker.predict(X_te)
        rmse = root_mean_squared_error(y_te, pred_te)
        r2 = r2_score(y_te, pred_te)
        print(f"Stacking Ensemble [{config_name}] -> Test RMSE: {rmse:.4f} | R2: {r2:.4f} (Weights: {stacker.meta.coef_})")
        
        save_path = os.path.join(MODEL_DIR, f"Stacking_{config_name}.joblib")
        joblib.dump(stacker, save_path)

    print("Stacking models successfully fitted and saved to models/")

