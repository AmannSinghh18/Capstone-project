import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TABLE_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")

ACC = os.path.join(TABLE_DIR, "phase3_ts_accuracy_table.csv")
DM = os.path.join(TABLE_DIR, "phase3_ts_dm_significance_table.csv")
NAT = os.path.join(TABLE_DIR, "phase4_econometric_native_attribution.csv")
GPFI = os.path.join(TABLE_DIR, "phase4_econometric_gpfi.csv")

required = [ACC, NAT, GPFI]
missing = [p for p in required if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(
        "Run Phase 3 TS and Phase 4 TS first. Missing: " +
        ", ".join(os.path.basename(p) for p in missing)
    )

acc = pd.read_csv(ACC)
nat = pd.read_csv(NAT)
gpfi = pd.read_csv(GPFI)
dm = pd.read_csv(DM) if os.path.isfile(DM) else pd.DataFrame()

CATS = ["Climate", "Macroeconomic", "Commodity"]
CAT_CONFIG = {
    "Climate": "Climate_only",
    "Macroeconomic": "Macro_only",
    "Commodity": "Commodity_only",
}

# Phase 5.1 (TS): within-TS-paradigm consensus
full = nat[nat["Config"] == "Full"].copy()
full["Top_Category"] = full[CATS].idxmax(axis=1)

consensus = []
for cat in CATS:
    vals = full[cat]
    consensus.append({
        "Category": cat,
        "Mean_Econometric_Share_%": vals.mean(),
        "Top_Model_Count": int((full["Top_Category"] == cat).sum()),
        "Top_Model_Share_%": 100 * (full["Top_Category"] == cat).mean(),
        "Mean_Rank": full[CATS].rank(axis=1, ascending=False)[cat].mean(),
    })
consensus = pd.DataFrame(consensus)

# Phase 5.2 (TS): diagnostic triangulation for TS paradigm
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
        "Mean_Econometric_Share_%": native.mean(),
        "Native_Rank": int(consensus.loc[consensus.Category == cat, "Mean_Econometric_Share_%"].rank(
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

native_rank = consensus.set_index("Category")["Mean_Econometric_Share_%"].rank(ascending=False, method="min")
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

    if high_native and strong_accuracy and strong_gpfi:
        label = "Core Driver"
    elif strong_accuracy and weak_gpfi:
        label = "Redundant / Linear Baseline"
    elif not high_native and strong_gpfi:
        label = "Synergistic Driver"
    else:
        label = "Mixed / Inconclusive"
    labels.append(label)

tri["Diagnostic_Classification"] = labels

# Save TS outputs
consensus.to_csv(os.path.join(TABLE_DIR, "phase5_ts_within_paradigm_consensus.csv"), index=False)
tri.to_csv(os.path.join(TABLE_DIR, "phase5_ts_diagnostic_triangulation.csv"), index=False)

print("Phase 5 TS complete.")
print("Saved: phase5_ts_within_paradigm_consensus.csv and phase5_ts_diagnostic_triangulation.csv")
