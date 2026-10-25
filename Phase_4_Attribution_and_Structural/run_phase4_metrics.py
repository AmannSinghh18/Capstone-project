import os, joblib, warnings
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import root_mean_squared_error
from sklearn.base import clone
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")
BASE=os.path.dirname(os.path.abspath(__file__))
PROJECT=os.path.dirname(BASE)
DATA=os.path.join(PROJECT,"data"); MODEL_DIR=os.path.join(PROJECT,"models")
OUT=os.path.join(PROJECT,"outputs","tables"); os.makedirs(OUT,exist_ok=True)

TARGET="cpi_headline_yoy"
P=["cpi_headline_yoy_lag1","cpi_headline_yoy_lag12"]
MACRO=["USA_CPI","iip_general_yoy","m3_money_supply_yoy","rbi_repo_rate_yoy_diff"]
COMMODITY=["wpi_primary_food_articles_yoy","wpi_manufactured_food_yoy","wpi_wheat_yoy","wpi_rice_yoy","wpi_maize_yoy","wpi_gram_chana_yoy","wpi_arhar_tur_yoy","wpi_moong_yoy","wpi_urad_yoy","wpi_masur_yoy","wpi_tomato_yoy","wpi_onion_yoy","wpi_potato_yoy","wpi_brinjal_yoy","wpi_cabbage_yoy","wpi_garlic_yoy","wpi_mustard_oil_yoy","wpi_soybean_oil_yoy","wpi_groundnut_oil_yoy","wpi_milk_yoy","wpi_egg_yoy","wpi_sugar_yoy","crude_oil_brent_usd_yoy","palm_oil_usd_yoy","global_fertilizer_index_yoy"]
RAW=["rainfall_total_mm","rainfall_rate_mm_per_day","temp_mean_celsius","relative_humidity_2m_pct","surface_soil_wetness_index","wind_speed_10m_ms","solar_irradiance_kwm2"]
CLIMATE=RAW+[f"{x}_anom" for x in RAW]
CONFIGS={"Full":CLIMATE+MACRO+COMMODITY+P,"Climate_only":CLIMATE+P,"Macro_only":MACRO+P,"Commodity_only":COMMODITY+P}
GROUPS={"Climate":CLIMATE,"Macroeconomic":MACRO,"Commodity":COMMODITY}
MODEL_NAMES=["RandomForest","XGBoost","ElasticNet","SVR"]
N_PERM=100
rng=np.random.default_rng(42)

def shap_values(model, name, Xtr, Xte):
    pipe=hasattr(model,"named_steps"); reg=model.named_steps[model.steps[-1][0]] if pipe else model
    if name in ("RandomForest","XGBoost"):
        vals=shap.TreeExplainer(reg)(Xte).values
    elif name=="ElasticNet":
        scaler=model.named_steps["scaler"]
        vals=shap.LinearExplainer(reg,scaler.transform(Xtr))(scaler.transform(Xte)).values
    else:
        scaler=model.named_steps["scaler"]; bg=shap.kmeans(scaler.transform(Xtr),min(20,len(Xtr)))
        vals=shap.KernelExplainer(reg.predict,bg).shap_values(scaler.transform(Xte),nsamples=100,l1_reg="num_features(10)")
    if isinstance(vals,list): vals=vals[0]
    return np.asarray(vals)

def group_shares(vals, cols):
    raw={g:float(np.abs(vals[:,[i for i,c in enumerate(cols) if c in fs]]).mean())
         for g,fs in GROUPS.items() if any(c in fs for c in cols)}
    total=sum(raw.values())
    return {g:100*v/total if total else 0 for g,v in raw.items()}

train=pd.read_csv(os.path.join(DATA,"train_processed.csv"))
test=pd.read_csv(os.path.join(DATA,"test_processed.csv")); y=test[TARGET].to_numpy()
nat=[]; var_rows=[]; gpfi=[]; stability=[]

for mn in MODEL_NAMES:
    for cn, requested in CONFIGS.items():
        path=os.path.join(MODEL_DIR,f"{mn}_{cn}.joblib")
        if not os.path.isfile(path): continue
        model=joblib.load(path); cols=[c for c in requested if c in test.columns]
        Xtr,Xte=train[cols],test[cols]
        vals=shap_values(model,mn,Xtr,Xte)
        absmean=np.abs(vals).mean(axis=0)
        for f,v in zip(cols,absmean):
            cat="Persistence" if f in P else next((g for g,fs in GROUPS.items() if f in fs),"Other")
            var_rows.append({"Model":mn,"Config":cn,"Feature":f,"Category":cat,"Mean_abs_SHAP":float(v)})
        shares=group_shares(vals,cols)
        nat.append({"Model":mn,"Config":cn,**shares})

        # GPFI is evaluated on the Full configuration only.
        if cn=="Full":
            base=root_mean_squared_error(y,model.predict(Xte))
            for cat,fs in GROUPS.items():
                vc=[c for c in fs if c in cols]
                if not vc: continue
                deltas=[]
                for _ in range(N_PERM):
                    Xp=Xte.copy(); order=rng.permutation(len(Xp))
                    Xp[vc]=Xp[vc].iloc[order].to_numpy()
                    deltas.append(root_mean_squared_error(y,model.predict(Xp))-base)
                delta=np.asarray(deltas)
                p=(1+np.sum(delta<=0))/(N_PERM+1)
                gpfi.append({"Model":mn,"Category":cat,"GPFI_mean_delta_RMSE":delta.mean(),
                             "GPFI_std":delta.std(ddof=1),"P_Value":p,
                             "CI95_L":np.quantile(delta,.025),"CI95_U":np.quantile(delta,.975)})

# BH-FDR for the GPFI test family.
gdf=pd.DataFrame(gpfi)
if not gdf.empty:
    p=gdf["P_Value"].to_numpy(); order=np.argsort(p); adj=np.empty(len(p)); adj[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    gdf["BH_FDR_P_Value"]=np.minimum(adj,1); gdf["Significant_0.05"]=gdf["BH_FDR_P_Value"]<.05
gdf.to_csv(os.path.join(OUT,"phase4_ml_gpfi.csv"),index=False)

sdf=pd.DataFrame(var_rows); sub=sdf[sdf.Category!="Persistence"].copy()
sub=sub.sort_values(["Model","Config","Mean_abs_SHAP"],ascending=[True,True,False])
sub["Rank_Overall"]=sub.groupby(["Model","Config"]).cumcount()+1
sub["Rank_Cat"]=sub.groupby(["Model","Config","Category"]).cumcount()+1
sub["Share_PCT"]=sub["Mean_abs_SHAP"]/sub.groupby(["Model","Config"])["Mean_abs_SHAP"].transform("sum")*100
sub.to_csv(os.path.join(OUT,"phase4_ml_all_variable_rankings.csv"),index=False)
sub.sort_values(["Model","Config","Category","Rank_Cat"]).to_csv(os.path.join(OUT,"phase4_ml_all_variable_rankings_by_category.csv"),index=False)
sub[sub.Rank_Overall<=3].to_csv(os.path.join(OUT,"phase4_ml_top3_variables_by_model_config.csv"),index=False)
pd.DataFrame(nat).to_csv(os.path.join(OUT,"phase4_ml_native_attribution.csv"),index=False)

# Compact expanding-window stability: fixed saved specification, refit within each chronological window.
# This tests whether group-SHAP rankings are stable without retuning hyperparameters.
for mn in MODEL_NAMES:
    path=os.path.join(MODEL_DIR,f"{mn}_Full.joblib")
    if not os.path.isfile(path): continue
    base_model=joblib.load(path); cols=CONFIGS["Full"]
    cols=[c for c in cols if c in train.columns]
    X=train[cols]; yt=train[TARGET].to_numpy()
    splits=[(tr,va) for tr,va in __import__("sklearn.model_selection",fromlist=["TimeSeriesSplit"]).TimeSeriesSplit(n_splits=5).split(X)]
    ranks=[]; shares=[]
    for tr,va in splits:
        fm=clone(base_model); fm.fit(X.iloc[tr],yt[tr])
        v=shap_values(fm,mn,X.iloc[tr],X.iloc[va]); s=group_shares(v,cols)
        shares.append(s); ranks.append(pd.Series(s).rank(ascending=False))
    if shares:
        overall=pd.Series(group_shares(shap_values(base_model,mn,train[cols],test[cols]),cols))
        target_rank=overall.rank(ascending=False)
        tau=np.nanmean([kendalltau(pd.Series(r).reindex(target_rank.index), target_rank).statistic for r in ranks])
        for cat in GROUPS:
            vals=[s.get(cat,np.nan) for s in shares]
            cv=np.std(vals,ddof=1)/np.mean(vals) if np.mean(vals)!=0 else np.nan
            stability.append({"Model":mn,"Config":"Full","Category":cat,"CVG":cv,"Mean_Kendall_Tau":tau})
pd.DataFrame(stability).to_csv(os.path.join(OUT,"phase4_ml_attribution_stability.csv"),index=False)

print("Phase 4 complete.")
