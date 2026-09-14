import os, joblib, warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT = os.path.join(BASE, "outputs", "tables")
os.makedirs(OUT, exist_ok=True)

TARGET = "cpi_headline_yoy"
P = ["cpi_headline_yoy_lag1", "cpi_headline_yoy_lag12"]
MACRO = ["USA_CPI","iip_general_yoy","m3_money_supply_yoy","rbi_repo_rate_yoy_diff"]
COMMODITY = ["wpi_primary_food_articles_yoy","wpi_manufactured_food_yoy","wpi_wheat_yoy","wpi_rice_yoy","wpi_maize_yoy","wpi_gram_chana_yoy","wpi_arhar_tur_yoy","wpi_moong_yoy","wpi_urad_yoy","wpi_masur_yoy","wpi_tomato_yoy","wpi_onion_yoy","wpi_potato_yoy","wpi_brinjal_yoy","wpi_cabbage_yoy","wpi_garlic_yoy","wpi_mustard_oil_yoy","wpi_soybean_oil_yoy","wpi_groundnut_oil_yoy","wpi_milk_yoy","wpi_egg_yoy","wpi_sugar_yoy","crude_oil_brent_usd_yoy","palm_oil_usd_yoy","global_fertilizer_index_yoy"]
CLIMATE_RAW = ["rainfall_total_mm","rainfall_rate_mm_per_day","temp_mean_celsius","relative_humidity_2m_pct","surface_soil_wetness_index","wind_speed_10m_ms","solar_irradiance_kwm2"]
CLIMATE = CLIMATE_RAW + [f"{x}_anom" for x in CLIMATE_RAW]
CONFIGS = {"Full":CLIMATE+MACRO+COMMODITY+P,"Climate_only":CLIMATE+P,"Macro_only":MACRO+P,"Commodity_only":COMMODITY+P}
MODEL_NAMES = ["RandomForest","XGBoost","ElasticNet","SVR"]

def cols_for(config, df):
    return [c for c in CONFIGS[config] if c in df.columns]

def hln_dm(y, p1, p2, h=1):
    d = (y-p1)**2 - (y-p2)**2
    T = len(d)
    if T < 2: return np.nan
    dm = d.mean() / np.sqrt(max(np.var(d, ddof=1)/T, 1e-12))
    factor = np.sqrt((T+1-2*h+h*(h-1)/T)/T)
    return float(2*st.t.sf(abs(dm*factor), df=T-1))

def bh(pvals, q=.05):
    p = np.asarray(pvals, dtype=float); out = np.full(len(p), np.nan)
    ok = np.isfinite(p); idx = np.where(ok)[0]
    if not len(idx): return out
    order = idx[np.argsort(p[idx])]; m=len(order)
    adj=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
    out[order]=np.minimum(adj,1.0); return out

train = pd.read_csv(os.path.join(DATA,"train_processed.csv"))
test = pd.read_csv(os.path.join(DATA,"test_processed.csv"))
y = test[TARGET].to_numpy()
preds, acc = {}, []

for model_name in MODEL_NAMES:
    preds[model_name] = {}
    for config in CONFIGS:
        path=os.path.join(MODEL_DIR,f"{model_name}_{config}.joblib")
        if not os.path.isfile(path): continue
        model=joblib.load(path); cols=cols_for(config, test)
        pred=model.predict(test[cols]); preds[model_name][config]=pred
        acc.append({"Model":model_name,"Config":config,"RMSE":root_mean_squared_error(y,pred),
                    "MAE":mean_absolute_error(y,pred),
                    "MAPE":100*mean_absolute_percentage_error(y,pred),"R2":r2_score(y,pred)})
acc_df=pd.DataFrame(acc)
acc_df.to_csv(os.path.join(OUT,"phase3_ml_accuracy_table.csv"),index=False)

# All six configuration comparisons within each ML model.
rows=[]
for m in MODEL_NAMES:
    if m not in preds: continue
    available=[c for c in CONFIGS if c in preds[m]]
    for i,a in enumerate(available):
        for b in available[i+1:]:
            rows.append({"Model":m,"Comparison":f"{a} vs {b}",
                         "P_Value_HLN":hln_dm(y,preds[m][a],preds[m][b])})
dm=pd.DataFrame(rows)
if not dm.empty:
    dm["BH_FDR_P_Value"]=bh(dm["P_Value_HLN"].to_numpy())
    dm["Significant_0.05"]=dm["BH_FDR_P_Value"]<.05
dm.to_csv(os.path.join(OUT,"phase3_ml_dm_significance_table.csv"),index=False)
print("Phase 3 complete.")
