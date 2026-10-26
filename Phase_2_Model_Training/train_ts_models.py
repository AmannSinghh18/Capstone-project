import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Phase_2_Model_Training.ts_wrappers import (
    RegularizedARDLWrapper,
    BlockDFMWrapper,
    VARXWrapper,
    BatesGrangerCombinationWrapper
)

warnings.filterwarnings("ignore")

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

MODEL_FACTORIES = {
    "Bates_Granger": lambda: BatesGrangerCombinationWrapper(),
    "DFM": lambda: BlockDFMWrapper(factors_per_block=2, alpha=2.0),
    "Ridge_ARDL": lambda: RegularizedARDLWrapper(q_lags=2, alpha=0.2, l1_ratio=0.2),
    "VARX": lambda: VARXWrapper(alpha=1.0)
}

if __name__ == "__main__":
    print("==================================================")
    print("PHASE 2 (TS): ESTIMATING UPGRADED 4-MODEL TS SUITE")
    print("==================================================")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
    
    for m_name, m_func in MODEL_FACTORIES.items():
        for config_name, feature_list in CONFIGS.items():
            cols = [c for c in feature_list if c in train_df.columns]
            X_tr, y_tr = train_df[cols], train_df[TARGET]
            X_te, y_te = test_df[cols], test_df[TARGET]
            
            model = m_func()
            model.fit(X_tr, y_tr)
            pred_te = model.predict(X_te)
            rmse = root_mean_squared_error(y_te, pred_te)
            r2 = r2_score(y_te, pred_te)
            
            save_path = os.path.join(MODEL_DIR, f"{m_name}_{config_name}.joblib")
            joblib.dump(model, save_path)
            print(f"[{m_name}] [{config_name}] -> Test RMSE: {rmse:.4f} | R2: {r2:.4f}")
            
    print("\nAll 4 upgraded TS models successfully estimated and saved to models/")
