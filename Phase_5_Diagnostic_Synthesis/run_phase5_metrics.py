import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TABLE_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")

ACC = os.path.join(TABLE_DIR, "phase3_ml_accuracy_table.csv")
DM = os.path.join(TABLE_DIR, "phase3_ml_dm_significance_table.csv")
NAT = os.path.join(TABLE_DIR, "phase4_ml_native_attribution.csv")
GPFI = os.path.join(TABLE_DIR, "phase4_ml_gpfi.csv")
STAB = os.path.join(TABLE_DIR, "phase4_ml_attribution_stability.csv")

required = [ACC, NAT, GPFI]
missing = [p for p in required if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(
        "Run Phase 3 and Phase 4 first. Missing: " +
        ", ".join(os.path.basename(p) for p in missing)
    )

acc = pd.read_csv(ACC)
nat = pd.read_csv(NAT)
gpfi = pd.read_csv(GPFI)
stab = pd.read_csv(STAB) if os.path.isfile(STAB) else pd.DataFrame()
dm = pd.read_csv(DM) if os.path.isfile(DM) else pd.DataFrame()

CATS = ["Climate", "Macroeconomic", "Commodity"]
CAT_CONFIG = {
    "Climate": "Climate_only",
    "Macroeconomic": "Macro_only",
    "Commodity": "Commodity_only",
}

# Phase 5.1: within-ML-paradigm consensus.
full = nat[nat["Config"] == "Full"].copy()
full["Top_Category"] = full[CATS].idxmax(axis=1)

consensus = []
for cat in CATS:
    vals = full[cat]
    consensus.append({
        "Category": cat,
        "Mean_SHAP_Share_%": vals.mean(),
        "Top_Model_Count": int((full["Top_Category"] == cat).sum()),
        "Top_Model_Share_%": 100 * (full["Top_Category"] == cat).mean(),
        "Mean_Rank": full[CATS].rank(axis=1, ascending=False)[cat].mean(),
    })
consensus = pd.DataFrame(consensus)

# Phase 5.2: diagnostic triangulation.
# Test 1 = native grouped SHAP.
# Test 2 = category-only forecasting accuracy (lower RMSE is better).
# Test 3 = GPFI accuracy reduction (positive, significant loss is strongest evidence).
tri = []
full_acc = acc[acc["Config"] == "Full"]
full_rmse = full_acc.groupby("Model")["RMSE"].first()

for cat in CATS:
    conf = CAT_CONFIG[cat]
    ca = acc[acc["Config"] == conf].copy()
    ca["Full_RMSE"] = ca["Model"].map(full_rmse)
    ca["RMSE_Difference_vs_Full"] = ca["RMSE"] - ca["Full_RMSE"]

    native = full[cat]
    gp = gpfi[gpfi["Category"] == cat].copy()

    tri.append({
        "Category": cat,
        "Mean_SHAP_Share_%": native.mean(),
        "Native_Rank": int(consensus.loc[consensus.Category == cat, "Mean_SHAP_Share_%"].rank(
            ascending=False, method="min"
        ).iloc[0]),
        "Category_only_Mean_RMSE": ca["RMSE"].mean(),
        "Category_only_RMSE_Rank": int(
            acc[acc["Config"].isin([CAT_CONFIG[c] for c in CATS])]
            .groupby("Config")["RMSE"].mean()
            .rank(ascending=True, method="min")[conf]
        ),
        "Mean_RMSE_vs_Full": ca["RMSE_Difference_vs_Full"].mean(),
        "Mean_GPFI_RMSE_Loss": gp["GPFI_mean_delta_RMSE"].mean(),
        "GPFI_Significant_Model_Count": int(gp["Significant_0.05"].sum())
            if "Significant_0.05" in gp else 0,
        "GPFI_Model_Count": len(gp),
    })

tri = pd.DataFrame(tri)

# Explicit diagnostic labels based only on the three methodology tests.
# Core: high native attribution + strong category-only accuracy + significant positive GPFI.
# Redundant: strong category-only accuracy but weak/non-significant GPFI.
# Non-Linear Synergy: relatively low native SHAP but strong/significant GPFI.
for df in (consensus, tri):
    pass

native_rank = consensus.set_index("Category")["Mean_SHAP_Share_%"].rank(ascending=False, method="min")
acc_rank = tri.set_index("Category")["Category_only_Mean_RMSE"].rank(ascending=True, method="min")

labels = []
for _, r in tri.iterrows():
    cat = r["Category"]
    high_native = native_rank[cat] == 1
    strong_accuracy = acc_rank[cat] <= 2
    strong_gpfi = (
        r["Mean_GPFI_RMSE_Loss"] > 0 and
        r["GPFI_Significant_Model_Count"] >= 2
    )
    weak_gpfi = r["Mean_GPFI_RMSE_Loss"] <= 0 or r["GPFI_Significant_Model_Count"] < 2
    low_native = native_rank[cat] >= 2

    if high_native and strong_accuracy and strong_gpfi:
        label = "Core Driver"
    elif strong_accuracy and weak_gpfi:
        label = "Redundant Driver"
    elif low_native and strong_gpfi:
        label = "Non-Linear Synergy"
    else:
        label = "Mixed / Inconclusive"
    labels.append(label)

tri["Diagnostic_Classification"] = labels

# Optional stability support; it is reported, not used to override the three-test labels.
if not stab.empty:
    s = stab.groupby("Category").agg(
        Mean_CVG=("CVG", "mean"),
        Mean_Kendall_Tau=("Mean_Kendall_Tau", "mean")
    ).reset_index()
    tri = tri.merge(s, on="Category", how="left")

# Phase 5.3: cross-paradigm comparison if econometric consensus tables exist.
# This does not fabricate econometric results; it only consumes available tables.
econ_candidates = [
    "phase5_econometric_consensus.csv",
    "phase4_econometric_attribution.csv",
    "phase4_econometric_native_attribution.csv",
]
econ = None
for name in econ_candidates:
    path = os.path.join(TABLE_DIR, name)
    if os.path.isfile(path):
        econ = pd.read_csv(path)
        break

if econ is not None and "Category" in econ.columns:
    ml_rank = consensus[["Category", "Mean_SHAP_Share_%"]].copy()
    ml_rank["ML_Rank"] = ml_rank["Mean_SHAP_Share_%"].rank(ascending=False, method="min")
    econ_cols = [c for c in econ.columns if c.lower() in
                 {"mean_share_%", "share_%", "mean_attribution_%", "rank"}]
    if econ_cols:
        ec = econ[["Category", econ_cols[0]]].copy()
        ec["Econometric_Rank"] = ec[econ_cols[0]].rank(ascending=False, method="min")
        cross = ml_rank.merge(ec[["Category", "Econometric_Rank"]], on="Category", how="left")
    else:
        cross = ml_rank
else:
    cross = consensus[["Category", "Mean_SHAP_Share_%"]].copy()
    cross["Econometric_Rank"] = np.nan
    cross["Note"] = "No econometric Phase 4 attribution table found; ML-only synthesis retained."

# Save outputs.
consensus.to_csv(os.path.join(TABLE_DIR, "phase5_ml_within_paradigm_consensus.csv"), index=False)
tri.to_csv(os.path.join(TABLE_DIR, "phase5_ml_diagnostic_triangulation.csv"), index=False)
cross.to_csv(os.path.join(TABLE_DIR, "phase5_ml_cross_paradigm_convergence.csv"), index=False)

print("Phase 5 complete.")
print("Saved: within-paradigm consensus, diagnostic triangulation, and cross-paradigm convergence tables.")
