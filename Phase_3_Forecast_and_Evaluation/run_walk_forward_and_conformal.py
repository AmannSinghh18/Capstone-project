import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
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
FULL_FEATURES = CLIMATE + MACRO + COMMODITY + PERSISTENCE

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
full_df = pd.concat([train_df, test_df], ignore_index=True)

print("==================================================")
print("PHASE 3 (ADVANCED): WALK-FORWARD BACKTEST & CONFORMAL PREDICTION")
print("==================================================")

# 1. Walk-Forward Expanding Window Backtesting
# Starting evaluation window after 160 observations (~year 2015), step 12 months
test_start = int(len(full_df) * 0.65)
step_size = 12
backtest_results = []

MODELS = {
    "ElasticNet": lambda: Pipeline([("scaler", StandardScaler()), ("en", ElasticNet(alpha=0.1, l1_ratio=0.8, random_state=42))]),
    "Ridge_ARX": lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0, random_state=42))]),
    "XGBoost": lambda: xgb.XGBRegressor(n_estimators=80, learning_rate=0.03, max_depth=3, random_state=42),
    "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=80, learning_rate=0.03, max_depth=3, random_state=42, n_jobs=1, verbose=-1)
}

cols = [c for c in FULL_FEATURES if c in full_df.columns]

for t in range(test_start, len(full_df) - 6, step_size):
    t_end = min(t + step_size, len(full_df))
    tr_slice = full_df.iloc[:t]
    te_slice = full_df.iloc[t:t_end]
    
    X_tr, y_tr = tr_slice[cols], tr_slice[TARGET]
    X_te, y_te = te_slice[cols], te_slice[TARGET]
    
    for m_name, m_func in MODELS.items():
        m = m_func()
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        
        backtest_results.append({
            "Window_Start_Idx": t,
            "Window_End_Idx": t_end,
            "Observations_Tested": len(y_te),
            "Model": m_name,
            "RMSE": root_mean_squared_error(y_te, preds),
            "MAE": mean_absolute_error(y_te, preds),
            "R2": r2_score(y_te, preds)
        })

backtest_df = pd.DataFrame(backtest_results)
wf_path = os.path.join(OUT_DIR, "phase3_walk_forward_backtest_accuracy.csv")
backtest_df.to_csv(wf_path, index=False)
print(f"Walk-forward backtest table saved to: {wf_path}")
print("Mean Backtest RMSE across Windows:")
print(backtest_df.groupby("Model")[["RMSE", "MAE", "R2"]].mean().round(4))


# 2. Conformal Prediction Intervals (90% & 95% Coverage)
# Fit base models on 80% train, calibrate on 20% validation split of train, test on out-of-sample test
val_split = int(len(train_df) * 0.75)
calib_tr = train_df.iloc[:val_split]
calib_val = train_df.iloc[val_split:]

X_c_tr, y_c_tr = calib_tr[cols], calib_tr[TARGET]
X_c_val, y_c_val = calib_val[cols], calib_val[TARGET]
X_test, y_test = test_df[cols], test_df[TARGET]

conformal_records = []

for m_name, m_func in MODELS.items():
    model = m_func()
    model.fit(X_c_tr, y_c_tr)
    
    # Calibration residuals (non-conformity scores)
    val_preds = model.predict(X_c_val)
    residuals = np.abs(y_c_val.values - val_preds)
    
    # Conformal quantiles with finite-sample adjustment (1 + 1/n)
    n_cal = len(residuals)
    q90_level = min(1.0, np.ceil((n_cal + 1) * 0.90) / n_cal)
    q95_level = min(1.0, np.ceil((n_cal + 1) * 0.95) / n_cal)
    
    q90 = np.quantile(residuals, q90_level)
    q95 = np.quantile(residuals, q95_level)
    
    # Test set predictions & intervals
    test_preds = model.predict(X_test)
    lower_90 = test_preds - q90
    upper_90 = test_preds + q90
    lower_95 = test_preds - q95
    upper_95 = test_preds + q95
    
    cov_90 = np.mean((y_test.values >= lower_90) & (y_test.values <= upper_90)) * 100
    cov_95 = np.mean((y_test.values >= lower_95) & (y_test.values <= upper_95)) * 100
    avg_width_90 = 2 * q90
    avg_width_95 = 2 * q95
    
    conformal_records.append({
        "Model": m_name,
        "Target_Coverage_90%": 90.0,
        "Empirical_Coverage_90%": float(cov_90),
        "Avg_Interval_Width_90%": float(avg_width_90),
        "Target_Coverage_95%": 95.0,
        "Empirical_Coverage_95%": float(cov_95),
        "Avg_Interval_Width_95%": float(avg_width_95)
    })

conf_df = pd.DataFrame(conformal_records)
conf_path = os.path.join(OUT_DIR, "phase3_conformal_prediction_intervals.csv")
conf_df.to_csv(conf_path, index=False)
print(f"\nConformal prediction table saved to: {conf_path}")
print(conf_df.to_string(index=False))
print("Phase 3 Advanced Evaluation complete.")
