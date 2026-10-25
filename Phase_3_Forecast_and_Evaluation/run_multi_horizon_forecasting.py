import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

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
    "Commodity_only": COMMODITY + PERSISTENCE,
    "Macro_only": MACRO + PERSISTENCE,
    "Climate_only": CLIMATE + PERSISTENCE
}

# Load complete processed dataset
RAW_PATH = os.path.join(PROJECT_ROOT, "processed_capstone_dataset_2001_2025.csv")
if not os.path.isfile(RAW_PATH):
    RAW_PATH = os.path.join(PROJECT_ROOT, "processed_capstone_dataset_2001_2025 (1).csv")

df = pd.read_csv(RAW_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Build anomalies
split0 = int(len(df) * 0.80)
month = df["Date"].dt.month
for c in CLIMATE_RAW:
    stats = df.iloc[:split0].groupby(month.iloc[:split0])[c].agg(["mean", "std"])
    df[f"{c}_anom"] = (df[c] - month.map(stats["mean"])) / month.map(stats["std"]).replace(0, pd.NA)
    df[f"{c}_anom"] = df[f"{c}_anom"].fillna(0.0)

# Persistence lags
df["cpi_headline_yoy_lag1"] = df[TARGET].shift(1)
df["cpi_headline_yoy_lag12"] = df[TARGET].shift(12)
df = df.dropna(subset=PERSISTENCE).reset_index(drop=True)

HORIZONS = [1, 3, 6, 12]
results = []

print("==================================================")
print("PHASE 3 (ADVANCED): MULTI-HORIZON FORECASTING (h=1, 3, 6, 12)")
print("==================================================")

for h in HORIZONS:
    # Direct target generation: y_{t+h}
    df_h = df.copy()
    df_h[f"target_h{h}"] = df_h[TARGET].shift(-h)
    df_h = df_h.dropna(subset=[f"target_h{h}"]).reset_index(drop=True)
    
    split_idx = int(len(df_h) * 0.80)
    train_h = df_h.iloc[:split_idx]
    test_h = df_h.iloc[split_idx:]
    y_tr_h = train_h[f"target_h{h}"]
    y_te_h = test_h[f"target_h{h}"]
    
    for config_name, feature_list in CONFIGS.items():
        cols = [c for c in feature_list if c in train_h.columns]
        X_tr = train_h[cols]
        X_te = test_h[cols]
        
        # 1. ElasticNet
        pipe_en = Pipeline([("scaler", StandardScaler()), ("en", ElasticNet(alpha=0.1, l1_ratio=0.8, random_state=42))])
        pipe_en.fit(X_tr, y_tr_h)
        pred_en = pipe_en.predict(X_te)
        
        results.append({
            "Horizon_Months": h,
            "Model": "ElasticNet",
            "Config": config_name,
            "RMSE": root_mean_squared_error(y_te_h, pred_en),
            "MAE": mean_absolute_error(y_te_h, pred_en),
            "R2": r2_score(y_te_h, pred_en)
        })
        
        # 2. Ridge ARX
        pipe_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0, random_state=42))])
        pipe_ridge.fit(X_tr, y_tr_h)
        pred_ridge = pipe_ridge.predict(X_te)
        
        results.append({
            "Horizon_Months": h,
            "Model": "Ridge_ARX",
            "Config": config_name,
            "RMSE": root_mean_squared_error(y_te_h, pred_ridge),
            "MAE": mean_absolute_error(y_te_h, pred_ridge),
            "R2": r2_score(y_te_h, pred_ridge)
        })
        
        # 3. XGBoost
        xgb_m = xgb.XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42)
        xgb_m.fit(X_tr, y_tr_h)
        pred_xgb = xgb_m.predict(X_te)
        
        results.append({
            "Horizon_Months": h,
            "Model": "XGBoost",
            "Config": config_name,
            "RMSE": root_mean_squared_error(y_te_h, pred_xgb),
            "MAE": mean_absolute_error(y_te_h, pred_xgb),
            "R2": r2_score(y_te_h, pred_xgb)
        })

res_df = pd.DataFrame(results)
out_path = os.path.join(OUT_DIR, "phase3_multi_horizon_accuracy.csv")
res_df.to_csv(out_path, index=False)

print(f"Multi-horizon accuracy results exported to: {out_path}")
# Display sample summary
summary = res_df.groupby(["Horizon_Months", "Config"])["RMSE"].mean().unstack()
print("\nMean Out-of-Sample RMSE across Horizons and Configurations:")
print(summary.round(4))
