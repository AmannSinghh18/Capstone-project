import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from scipy.stats import kendalltau

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from Phase_2_TS_Models.ts_wrappers import SARIMAXWrapper, AutoARIMAWrapper, VARXWrapper, RidgeARXWrapper

warnings.filterwarnings("ignore")

DATA = os.path.join(PROJECT, "data")
MODEL_DIR = os.path.join(PROJECT, "models")
OUT = os.path.join(PROJECT, "outputs", "tables")
os.makedirs(OUT, exist_ok=True)

TARGET = "cpi_headline_yoy"
P = ["cpi_headline_yoy_lag1", "cpi_headline_yoy_lag12"]
MACRO = ["USA_CPI", "iip_general_yoy", "m3_money_supply_yoy", "rbi_repo_rate_yoy_diff"]
COMMODITY = [
    "wpi_primary_food_articles_yoy", "wpi_manufactured_food_yoy", "wpi_wheat_yoy", "wpi_rice_yoy", "wpi_maize_yoy",
    "wpi_gram_chana_yoy", "wpi_arhar_tur_yoy", "wpi_moong_yoy", "wpi_urad_yoy", "wpi_masur_yoy", "wpi_tomato_yoy",
    "wpi_onion_yoy", "wpi_potato_yoy", "wpi_brinjal_yoy", "wpi_cabbage_yoy", "wpi_garlic_yoy", "wpi_mustard_oil_yoy",
    "wpi_soybean_oil_yoy", "wpi_groundnut_oil_yoy", "wpi_milk_yoy", "wpi_egg_yoy", "wpi_sugar_yoy",
    "crude_oil_brent_usd_yoy", "palm_oil_usd_yoy", "global_fertilizer_index_yoy"
]
RAW = ["rainfall_total_mm", "rainfall_rate_mm_per_day", "temp_mean_celsius", "relative_humidity_2m_pct", "surface_soil_wetness_index", "wind_speed_10m_ms", "solar_irradiance_kwm2"]
CLIMATE = RAW + [f"{x}_anom" for x in RAW]

CONFIGS = {
    "Full": CLIMATE + MACRO + COMMODITY + P,
    "Climate_only": CLIMATE + P,
    "Macro_only": MACRO + P,
    "Commodity_only": COMMODITY + P
}
GROUPS = {"Climate": CLIMATE, "Macroeconomic": MACRO, "Commodity": COMMODITY}
MODEL_NAMES = ["SARIMAX", "AutoARIMA", "VARX", "Ridge_ARX"]

N_PERM = 100
rng = np.random.default_rng(42)

def extract_coefs(model, cols):
    if hasattr(model, "pipeline"):
        pipe = model.pipeline
        reg = pipe.named_steps[pipe.steps[-1][0]]
        if hasattr(reg, "coef_"):
            return np.abs(reg.coef_)
    elif hasattr(model, "model_res_") and model.model_res_ is not None:
        if hasattr(model.model_res_, "params"):
            params = model.model_res_.params
            if len(params) >= len(cols):
                return np.abs(params[-len(cols):])
    return np.ones(len(cols))

def group_shares(coefs, cols):
    raw = {g: float(np.abs(coefs[[i for i, c in enumerate(cols) if c in fs]]).mean())
           for g, fs in GROUPS.items() if any(c in fs for c in cols)}
    total = sum(raw.values())
    return {g: 100 * v / total if total else 0 for g, v in raw.items()}

train = pd.read_csv(os.path.join(DATA, "train_processed.csv"))
test = pd.read_csv(os.path.join(DATA, "test_processed.csv"))
y = test[TARGET].to_numpy()

nat, var_rows, gpfi, stability = [], [], [], []

print("==================================================")
print("PHASE 4 (TS): ECONOMETRIC FEATURE ATTRIBUTION & STABILITY")
print("==================================================")

for mn in MODEL_NAMES:
    for cn, requested in CONFIGS.items():
        path = os.path.join(MODEL_DIR, f"{mn}_{cn}.joblib")
        if not os.path.isfile(path):
            continue
        model = joblib.load(path)
        cols = [c for c in requested if c in test.columns]
        Xtr, Xte = train[cols], test[cols]
        
        coefs = extract_coefs(model, cols)
        for f, v in zip(cols, coefs):
            cat = "Persistence" if f in P else next((g for g, fs in GROUPS.items() if f in fs), "Other")
            var_rows.append({"Model": mn, "Config": cn, "Feature": f, "Category": cat, "Mean_abs_coef": float(v)})
            
        shares = group_shares(coefs, cols)
        nat.append({"Model": mn, "Config": cn, **shares})
        
        # GPFI evaluated on Full config
        if cn == "Full":
            pred_base = model.predict(Xte)
            if hasattr(pred_base, "values"):
                pred_base = pred_base.values
            base_rmse = root_mean_squared_error(y, pred_base)
            
            for cat, fs in GROUPS.items():
                vc = [c for c in fs if c in cols]
                if not vc:
                    continue
                deltas = []
                for _ in range(N_PERM):
                    Xp = Xte.copy()
                    order = rng.permutation(len(Xp))
                    Xp[vc] = Xp[vc].iloc[order].to_numpy()
                    p_pred = model.predict(Xp)
                    if hasattr(p_pred, "values"):
                        p_pred = p_pred.values
                    deltas.append(root_mean_squared_error(y, p_pred) - base_rmse)
                    
                delta = np.asarray(deltas)
                p_val = (1 + np.sum(delta <= 0)) / (N_PERM + 1)
                gpfi.append({
                    "Model": mn, "Category": cat, "GPFI_mean_delta_RMSE": delta.mean(),
                    "GPFI_std": delta.std(ddof=1), "P_Value": p_val,
                    "CI95_L": np.quantile(delta, 0.025), "CI95_U": np.quantile(delta, 0.975)
                })

# BH-FDR correction for Econometric GPFI
gdf = pd.DataFrame(gpfi)
if not gdf.empty:
    p = gdf["P_Value"].to_numpy()
    order = np.argsort(p)
    adj = np.empty(len(p))
    adj[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    gdf["BH_FDR_P_Value"] = np.minimum(adj, 1)
    gdf["Significant_0.05"] = gdf["BH_FDR_P_Value"] < 0.05

gdf.to_csv(os.path.join(OUT, "phase4_econometric_gpfi.csv"), index=False)

sdf = pd.DataFrame(var_rows)
sub = sdf[sdf.Category != "Persistence"].copy()
sub = sub.sort_values(["Model", "Config", "Mean_abs_coef"], ascending=[True, True, False])
sub["Rank_Overall"] = sub.groupby(["Model", "Config"]).cumcount() + 1
sub["Rank_Cat"] = sub.groupby(["Model", "Config", "Category"]).cumcount() + 1
sub["Share_PCT"] = sub["Mean_abs_coef"] / sub.groupby(["Model", "Config"])["Mean_abs_coef"].transform("sum") * 100

sub.to_csv(os.path.join(OUT, "phase4_econometric_all_variable_rankings.csv"), index=False)
sub[sub.Rank_Overall <= 3].to_csv(os.path.join(OUT, "phase4_econometric_top3_variables_by_model_config.csv"), index=False)

nat_df = pd.DataFrame(nat)
nat_df.to_csv(os.path.join(OUT, "phase4_econometric_native_attribution.csv"), index=False)
nat_df[nat_df.Config == "Full"].to_csv(os.path.join(OUT, "phase4_econometric_attribution.csv"), index=False)

print("Saved phase4_econometric_native_attribution.csv, phase4_econometric_gpfi.csv, and variable rankings.")
print("Phase 4 TS complete.")
