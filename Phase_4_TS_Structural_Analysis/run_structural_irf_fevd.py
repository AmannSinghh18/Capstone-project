import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))

print("==================================================")
print("PHASE 4 (TS STRUCTURAL): IMPULSE RESPONSES (IRF) & FEVD")
print("==================================================")

# Structural Cholesky ordering from most exogenous to policy response to target:
# 1. Rainfall (Climate exogenous)
# 2. Crude Oil (Global commodity shock)
# 3. Primary Food WPI (Domestic agricultural price shock)
# 4. RBI Repo Rate (Central bank policy instrument)
# 5. Headline CPI YoY (Inflation target)
VAR_VARS = [
    "rainfall_total_mm",
    "crude_oil_brent_usd_yoy",
    "wpi_primary_food_articles_yoy",
    "rbi_repo_rate_yoy_diff",
    "cpi_headline_yoy"
]

df_var = train_df[VAR_VARS].dropna()

# Fit VAR model
var_model = VAR(df_var)
var_res = var_model.fit(maxlags=2, ic="aic")
print(f"Fitted VAR({var_res.k_ar}) with {len(df_var)} observations.")

# 1. Impulse Response Functions (12 periods)
HORIZON = 12
irf = var_res.irf(HORIZON)
cpi_idx = VAR_VARS.index("cpi_headline_yoy")

irf_records = []
for h in range(HORIZON + 1):
    for j, shock_var in enumerate(VAR_VARS):
        response_val = irf.orth_irfs[h, cpi_idx, j]
        stderr_val = irf.stderr()[h, cpi_idx, j] if hasattr(irf, "stderr") else np.nan
        irf_records.append({
            "Horizon_Step": h,
            "Shock_Variable": shock_var,
            "Response_Target": "cpi_headline_yoy",
            "Orthogonal_Impulse_Response": float(response_val),
            "Std_Error": float(stderr_val) if not np.isnan(stderr_val) else 0.0,
            "CI95_Lower": float(response_val - 1.96 * stderr_val) if not np.isnan(stderr_val) else float(response_val),
            "CI95_Upper": float(response_val + 1.96 * stderr_val) if not np.isnan(stderr_val) else float(response_val),
        })

irf_df = pd.DataFrame(irf_records)
irf_path = os.path.join(OUT_DIR, "phase4_ts_irf_table.csv")
irf_df.to_csv(irf_path, index=False)
print(f"IRF table saved to: {irf_path}")

# 2. Forecast Error Variance Decomposition (FEVD)
fevd = var_res.fevd(HORIZON)
fevd_decomp = fevd.decomp[cpi_idx] # shape (HORIZON, num_vars)

fevd_records = []
eval_horizons = [1, 3, 6, 12]
for h in eval_horizons:
    step_idx = h - 1
    shares = fevd_decomp[step_idx] * 100
    for j, shock_var in enumerate(VAR_VARS):
        fevd_records.append({
            "Horizon_Month": h,
            "Shock_Variable": shock_var,
            "Target": "cpi_headline_yoy",
            "Variance_Contribution_%": float(shares[j])
        })

fevd_df = pd.DataFrame(fevd_records)
fevd_path = os.path.join(OUT_DIR, "phase4_ts_fevd_table.csv")
fevd_df.to_csv(fevd_path, index=False)
print(f"FEVD table saved to: {fevd_path}")

# Display FEVD summary
fevd_summary = fevd_df.pivot(index="Shock_Variable", columns="Horizon_Month", values="Variance_Contribution_%")
print("\nForecast Error Variance Decomposition (% contribution to Headline CPI variance):")
print(fevd_summary.round(2))
print("Phase 4 Structural Analysis complete.")
