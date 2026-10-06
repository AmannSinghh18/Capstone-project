import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd

# Add Project root to sys.path to allow imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Phase_2_TS_Models.ts_wrappers import SARIMAXWrapper, AutoARIMAWrapper, VARXWrapper, RidgeARXWrapper

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

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))

print("==================================================")
print("PHASE 2 (TS): TIME SERIES MODELS ESTIMATION & FITTING")
print("==================================================")

MODEL_CLASSES = {
    "SARIMAX": SARIMAXWrapper,
    "AutoARIMA": AutoARIMAWrapper,
    "VARX": VARXWrapper,
    "Ridge_ARX": RidgeARXWrapper
}

# Fit and save all 16 TS Model-Config combinations
for model_name, wrapper_cls in MODEL_CLASSES.items():
    for config_name, feature_list in CONFIGS.items():
        cols = [c for c in feature_list if c in train_df.columns]
        X_tr = train_df[cols]
        y_tr = train_df[TARGET]
        
        print(f"Fitting TS Model: {model_name} | Config: {config_name} (Features: {len(cols)})")
        model = wrapper_cls()
        model.fit(X_tr, y_tr)
        
        save_path = os.path.join(MODEL_DIR, f"{model_name}_{config_name}.joblib")
        joblib.dump(model, save_path)
        print(f"Saved checkpoint: {save_path}")

print("==================================================")
print("Phase 2 TS Model Training Complete.")
print("==================================================")
