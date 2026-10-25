import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train_processed.csv")
if not os.path.isfile(TRAIN_PATH):
    raise FileNotFoundError(f"Training dataset not found at {TRAIN_PATH}")

df_train = pd.read_csv(TRAIN_PATH)
TARGET = "cpi_headline_yoy"

print("==================================================")
print("PHASE 1 (TS): TIME SERIES DIAGNOSTICS & STATIONARITY")
print("==================================================")

# 1. ADF & KPSS Stationarity Tests on Target and Features
results = []
numeric_cols = [c for c in df_train.columns if c != "Date"]

for col in numeric_cols:
    series = df_train[col].dropna()
    
    # ADF Test (Null: Series has a unit root / non-stationary)
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(series, autolag="AIC")
    
    # KPSS Test (Null: Series is trend-stationary / stationary)
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
        
    results.append({
        "Feature": col,
        "ADF_Statistic": adf_stat,
        "ADF_P_Value": adf_p,
        "ADF_Stationary_0.05": adf_p < 0.05,
        "KPSS_Statistic": kpss_stat,
        "KPSS_P_Value": kpss_p,
        "KPSS_Stationary_0.05": kpss_p > 0.05 if not np.isnan(kpss_p) else np.nan
    })

res_df = pd.DataFrame(results)
diag_path = os.path.join(OUT_DIR, "phase1_ts_stationarity_diagnostics.csv")
res_df.to_csv(diag_path, index=False)
print(f"Stationarity diagnostics saved to: {diag_path}")
print(f"Target '{TARGET}' ADF P-value: {res_df.loc[res_df.Feature==TARGET, 'ADF_P_Value'].values[0]:.4f}")
print("Phase 1 TS Diagnostics complete.")
