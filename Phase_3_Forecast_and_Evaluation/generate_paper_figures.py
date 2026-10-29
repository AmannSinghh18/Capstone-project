import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# Define base directory paths
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Phase_2_Model_Training.train_lgbm_and_stacking import TimeSeriesStackingRegressor
from Phase_2_Model_Training.ts_wrappers import (
    RegularizedARDLWrapper,
    BlockDFMWrapper,
    DFMWrapper,
    BatesGrangerCombinationWrapper,
    VARXWrapper,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)

# Feature group specifications
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
CLIMATE_RAW = [
    "rainfall_total_mm", "rainfall_rate_mm_per_day", "temp_mean_celsius", "relative_humidity_2m_pct",
    "surface_soil_wetness_index", "wind_speed_10m_ms", "solar_irradiance_kwm2"
]
CLIMATE = CLIMATE_RAW + [f"{x}_anom" for x in CLIMATE_RAW]

CONFIGS = {
    "Full": CLIMATE + MACRO + COMMODITY + P,
    "Climate_only": CLIMATE + P,
    "Macro_only": MACRO + P,
    "Commodity_only": COMMODITY + P
}

ML_MODELS = ["Stacking", "ElasticNet", "XGBoost", "SVR"]
TS_MODELS = ["Bates_Granger", "DFM", "Ridge_ARDL", "VARX"]

# Visual aesthetics setup for research paper publication
plt.rcParams.update({
    "font.size": 11,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "lines.linewidth": 1.9,
})

# Load datasets
raw_full = pd.read_csv(os.path.join(PROJECT_ROOT, "processed_capstone_dataset_2001_2025.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
dates = pd.to_datetime(raw_full["Date"].iloc[-len(test_df):].values)
y_actual = test_df[TARGET].to_numpy()

# Generate predictions for ML models (Full config)
ml_preds = {}
for m_name in ML_MODELS:
    path = os.path.join(MODEL_DIR, f"{m_name}_Full.joblib")
    m = joblib.load(path)
    cols = [c for c in CONFIGS["Full"] if c in test_df.columns]
    pred = m.predict(test_df[cols])
    if hasattr(pred, "values"):
        pred = pred.values
    ml_preds[m_name] = np.asarray(pred).flatten()

# Generate predictions for TS models (Full config)
ts_preds = {}
for m_name in TS_MODELS:
    path = os.path.join(MODEL_DIR, f"{m_name}_Full.joblib")
    m = joblib.load(path)
    cols = [c for c in CONFIGS["Full"] if c in test_df.columns]
    pred = m.predict(test_df[cols])
    if hasattr(pred, "values"):
        pred = pred.values
    ts_preds[m_name] = np.asarray(pred).flatten()


# -------------------------------------------------------------
# FIGURE 1: ML Model Loss Trajectory (Absolute & Cumulative)
# -------------------------------------------------------------
print("Generating Figure 1: ML Loss Trajectory...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

ml_colors = {
    "Stacking": "#1f77b4",     # Bold Blue (Champion)
    "ElasticNet": "#2ca02c",   # Forest Green
    "XGBoost": "#ff7f0e",      # Amber Orange
    "SVR": "#d62728"           # Crimson Red
}
ml_labels = {
    "Stacking": "Stacking Ensemble (Champion, RMSE 0.634)",
    "ElasticNet": "ElasticNet Regularized (RMSE 0.668)",
    "XGBoost": "XGBoost Gradient Boosted Trees (RMSE 0.816)",
    "SVR": "Support Vector Regression (RMSE 0.906)"
}

for m_name in ML_MODELS:
    err = np.abs(y_actual - ml_preds[m_name])
    cum_se = np.cumsum((y_actual - ml_preds[m_name]) ** 2)
    lw = 2.4 if m_name == "Stacking" else 1.8
    ax1.plot(dates, err, label=ml_labels[m_name], color=ml_colors[m_name], linewidth=lw, alpha=0.9)
    ax2.plot(dates, cum_se, label=ml_labels[m_name], color=ml_colors[m_name], linewidth=lw, alpha=0.9)

ax1.set_title("(a) Out-of-Sample Monthly Absolute Forecast Loss $|y_t - \hat{y}_t|$: Machine Learning Suite", fontweight="bold", loc="left")
ax1.set_ylabel("Absolute Loss (% pts)")
ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")
ax1.set_ylim(bottom=0)

ax2.set_title("(b) Cumulative Squared Error Trajectory $\sum_{\\tau=1}^t (y_\\tau - \hat{y}_\\tau)^2$", fontweight="bold", loc="left")
ax2.set_ylabel("Cumulative Loss (SSE)")
ax2.set_xlabel("Out-of-Sample Evaluation Date (2021 - 2025)")
ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_xlim(dates[0], dates[-1])

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_loss_trajectory_ml.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# FIGURE 2: TS Model Loss Trajectory (Absolute & Cumulative)
# -------------------------------------------------------------
print("Generating Figure 2: TS Loss Trajectory...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

ts_colors = {
    "Bates_Granger": "#005f73",  # Deep Teal (Champion)
    "Ridge_ARDL": "#9b2226",     # Rich Burgundy
    "DFM": "#ee9b00",            # Ochre Gold
    "VARX": "#6c757d"            # Steel Gray
}
ts_labels = {
    "Bates_Granger": "Bates-Granger Optimal Combination (Champion, RMSE 0.802)",
    "Ridge_ARDL": "Regularized ARDL(p, q) (RMSE 0.813)",
    "DFM": "Block-Structured Dynamic Factor Model (RMSE 0.840)",
    "VARX": "Vector Autoregression (VARX) (RMSE 0.922)"
}

for m_name in TS_MODELS:
    err = np.abs(y_actual - ts_preds[m_name])
    cum_se = np.cumsum((y_actual - ts_preds[m_name]) ** 2)
    lw = 2.4 if m_name == "Bates_Granger" else 1.8
    ax1.plot(dates, err, label=ts_labels[m_name], color=ts_colors[m_name], linewidth=lw, alpha=0.9)
    ax2.plot(dates, cum_se, label=ts_labels[m_name], color=ts_colors[m_name], linewidth=lw, alpha=0.9)

ax1.set_title("(a) Out-of-Sample Monthly Absolute Forecast Loss $|y_t - \hat{y}_t|$: Econometric TS Suite", fontweight="bold", loc="left")
ax1.set_ylabel("Absolute Loss (% pts)")
ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")
ax1.set_ylim(bottom=0)

ax2.set_title("(b) Cumulative Squared Error Trajectory $\sum_{\\tau=1}^t (y_\\tau - \hat{y}_\\tau)^2$", fontweight="bold", loc="left")
ax2.set_ylabel("Cumulative Loss (SSE)")
ax2.set_xlabel("Out-of-Sample Evaluation Date (2021 - 2025)")
ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_xlim(dates[0], dates[-1])

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_loss_trajectory_ts.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# FIGURE 3: Actual vs. Forecast Inflation Trajectory (Dual Paradigm)
# -------------------------------------------------------------
print("Generating Figure 3: Actual vs Predicted Inflation...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)

# Panel 1: ML Models vs Actual
ax1.axhspan(2.0, 6.0, color="#f0f3f4", alpha=0.8, label="RBI Inflation Tolerance Band (4% $\pm$ 2%)")
ax1.axhline(4.0, color="#7f8c8d", linestyle=":", linewidth=1.2, label="RBI Target Midpoint (4.0%)")
ax1.plot(dates, y_actual, color="black", linewidth=2.8, label="Actual CPI Headline YoY Inflation")

for m_name in ML_MODELS:
    style = "-" if m_name == "Stacking" else "--"
    alpha = 1.0 if m_name == "Stacking" else 0.75
    lw = 2.4 if m_name == "Stacking" else 1.6
    ax1.plot(dates, ml_preds[m_name], label=m_name, color=ml_colors[m_name], linestyle=style, linewidth=lw, alpha=alpha)

ax1.set_title("(a) Machine Learning Paradigm: Out-of-Sample Forecast Tracking (2021 - 2025)", fontweight="bold", loc="left")
ax1.set_ylabel("CPI YoY Inflation (%)")
ax1.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", edgecolor="#ccc")

# Panel 2: TS Models vs Actual
ax2.axhspan(2.0, 6.0, color="#f0f3f4", alpha=0.8, label="RBI Inflation Tolerance Band (4% $\pm$ 2%)")
ax2.axhline(4.0, color="#7f8c8d", linestyle=":", linewidth=1.2, label="RBI Target Midpoint (4.0%)")
ax2.plot(dates, y_actual, color="black", linewidth=2.8, label="Actual CPI Headline YoY Inflation")

for m_name in TS_MODELS:
    style = "-" if m_name == "Bates_Granger" else "--"
    alpha = 1.0 if m_name == "Bates_Granger" else 0.75
    lw = 2.4 if m_name == "Bates_Granger" else 1.6
    ax2.plot(dates, ts_preds[m_name], label=m_name.replace("_", " "), color=ts_colors[m_name], linestyle=style, linewidth=lw, alpha=alpha)

ax2.set_title("(b) Time Series Econometric Paradigm: Out-of-Sample Forecast Tracking (2021 - 2025)", fontweight="bold", loc="left")
ax2.set_ylabel("CPI YoY Inflation (%)")
ax2.set_xlabel("Evaluation Date (Monthly, 2021 - 2025)")
ax2.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", edgecolor="#ccc")
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_xlim(dates[0], dates[-1])

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_forecast_vs_actual_dual_paradigm.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# FIGURE 4: Comparative RMSE across Paradigms and Configurations
# -------------------------------------------------------------
print("Generating Figure 4: Model RMSE Comparison...")
ml_acc = pd.read_csv(os.path.join(TABLE_DIR, "phase3_ml_accuracy_table.csv"))
ts_acc = pd.read_csv(os.path.join(TABLE_DIR, "phase3_ts_accuracy_table.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

configs_order = ["Full", "Commodity_only", "Macro_only", "Climate_only"]
configs_labels = ["Full\n(45 Feats)", "Commodity\n(25 Feats)", "Macro\n(4 Feats)", "Climate\n(14 Feats)"]
x = np.arange(len(configs_order))
width = 0.19

# ML Panel
for i, m_name in enumerate(ML_MODELS):
    sub = ml_acc[ml_acc["Model"] == m_name].set_index("Config")
    vals = [sub.loc[c, "RMSE"] for c in configs_order]
    offset = (i - 1.5) * width
    bars = ax1.bar(x + offset, vals, width, label=m_name, color=ml_colors[m_name], alpha=0.88, edgecolor="black", linewidth=0.6)
    # Add text labels on top of champion bars
    if m_name == "Stacking":
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax1.set_title("(a) Machine Learning Paradigm", fontweight="bold")
ax1.set_ylabel("Out-of-Sample RMSE (% pts)")
ax1.set_xticks(x)
ax1.set_xticklabels(configs_labels)
ax1.set_ylim(0, 1.35)
ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")

# TS Panel
for i, m_name in enumerate(TS_MODELS):
    sub = ts_acc[ts_acc["Model"] == m_name].set_index("Config")
    vals = [sub.loc[c, "RMSE"] for c in configs_order]
    offset = (i - 1.5) * width
    bars = ax2.bar(x + offset, vals, width, label=m_name.replace("_", " "), color=ts_colors[m_name], alpha=0.88, edgecolor="black", linewidth=0.6)
    if m_name == "Bates_Granger":
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax2.set_title("(b) Time Series Econometric Paradigm", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(configs_labels)
ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")

fig.suptitle("Out-of-Sample Root Mean Squared Error (RMSE) across Configurations", fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_model_accuracy_comparison.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# FIGURE 5: GPFI Causal Degradation Comparison (Triangulation)
# -------------------------------------------------------------
print("Generating Figure 5: GPFI Feature Importance...")
ml_gpfi = pd.read_csv(os.path.join(TABLE_DIR, "phase4_ml_gpfi.csv"))
ts_gpfi = pd.read_csv(os.path.join(TABLE_DIR, "phase4_econometric_gpfi.csv"))

cats = ["Commodity", "Macroeconomic", "Climate"]
cats_display = ["Commodity Basket\n(Food & Energy)", "Macroeconomic\n(Liquidity & Policy)", "Agro-Climate\n(Weather & Anom.)"]
x_c = np.arange(len(cats))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

# ML GPFI
for i, m_name in enumerate(ML_MODELS):
    sub = ml_gpfi[ml_gpfi["Model"] == m_name].set_index("Category")
    vals = [sub.loc[c, "GPFI_mean_delta_RMSE"] for c in cats]
    stds = [sub.loc[c, "GPFI_std"] for c in cats]
    offset = (i - 1.5) * width
    ax1.bar(x_c + offset, vals, width, yerr=stds, capsize=3, label=m_name, color=ml_colors[m_name], alpha=0.88, edgecolor="black", linewidth=0.6)

ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax1.set_title("(a) ML Models: Accuracy Loss (GPFI $\Delta$RMSE)", fontweight="bold")
ax1.set_ylabel("Error Degradation $\Delta$RMSE (% pts)")
ax1.set_xticks(x_c)
ax1.set_xticklabels(cats_display)
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc")

# TS GPFI
for i, m_name in enumerate(TS_MODELS):
    sub = ts_gpfi[ts_gpfi["Model"] == m_name].set_index("Category")
    vals = [sub.loc[c, "GPFI_mean_delta_RMSE"] for c in cats]
    stds = [sub.loc[c, "GPFI_std"] for c in cats]
    offset = (i - 1.5) * width
    ax2.bar(x_c + offset, vals, width, yerr=stds, capsize=3, label=m_name.replace("_", " "), color=ts_colors[m_name], alpha=0.88, edgecolor="black", linewidth=0.6)

ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax2.set_title("(b) Econometric TS Models: Accuracy Loss (GPFI $\Delta$RMSE)", fontweight="bold")
ax2.set_xticks(x_c)
ax2.set_xticklabels(cats_display)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc")

fig.suptitle("Grouped Permutation Feature Importance: Predictive Causal Necessity ($\Delta$RMSE Loss)", fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_gpfi_causal_necessity.png"), dpi=300)
plt.close()

print("==================================================")
print(f"All 5 publication figures successfully saved to: {FIG_DIR}")
print("1. fig1_loss_trajectory_ml.png")
print("2. fig2_loss_trajectory_ts.png")
print("3. fig3_forecast_vs_actual_dual_paradigm.png")
print("4. fig4_model_accuracy_comparison.png")
print("5. fig5_gpfi_causal_necessity.png")
print("==================================================")
