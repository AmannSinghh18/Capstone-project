import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "processed_capstone_dataset_2001_2025.csv"),
    os.path.join(PROJECT_ROOT, "processed_capstone_dataset_2001_2025 (1).csv"),
]
RAW_PATH = next((p for p in RAW_CANDIDATES if os.path.isfile(p)), None)
if RAW_PATH is None:
    raise FileNotFoundError(
        "Could not find processed_capstone_dataset_2001_2025.csv in Capstone_Project/"
    )

# 1. Feature Definitions
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

print("Loading raw data...")
df = pd.read_csv(RAW_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Calculating climate anomalies and persistence lags...")
# Calculate Anomalies (Fit on first 80% only)
split0 = int(len(df) * 0.80)
month = df["Date"].dt.month
for c in CLIMATE_RAW:
    stats = df.iloc[:split0].groupby(month.iloc[:split0])[c].agg(["mean", "std"])
    df[f"{c}_anom"] = (df[c] - month.map(stats["mean"])) / month.map(stats["std"]).replace(0, pd.NA)
    df[f"{c}_anom"] = df[f"{c}_anom"].fillna(0.0)

# Persistence (Lags)
df["cpi_headline_yoy_lag1"] = df[TARGET].shift(1)
df["cpi_headline_yoy_lag12"] = df[TARGET].shift(12)
df = df.dropna(subset=PERSISTENCE).reset_index(drop=True)

# Filter Only Substantive Columns + Target
FINAL_COLUMNS = CLIMATE + MACRO + COMMODITY + PERSISTENCE + [TARGET]
df_final = df[FINAL_COLUMNS]

print("Splitting and saving to data/ ...")
split_idx = int(len(df_final) * 0.80)
train_df = df_final.iloc[:split_idx]
test_df = df_final.iloc[split_idx:]

os.makedirs(DATA_DIR, exist_ok=True)
train_df.to_csv(os.path.join(DATA_DIR, "train_processed.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test_processed.csv"), index=False)
print("Saved train_processed.csv and test_processed.csv successfully.")