"""
Consolidated Capstone Dataset Merging Pipeline
----------------------------------------------
File Location: preprocessing/combine_all_datasets.py
Input Directory: ../data/
Output File: ../master_capstone_dataset_2000_2026.csv (Project Root Folder)

This script reads all individual data files from the data/ folder, merges them
sequentially on 'Date', and exports the combined master dataset to the root folder.
"""

import os
import pandas as pd

def build_consolidated_master_dataset(data_folder='../data/'):
    print(f"Starting Data Consolidation Pipeline reading from: {data_folder}")
    
    # Fallback check if files are inside a zip subfolder in data/
    if not os.path.exists(os.path.join(data_folder, 'imf_india_cpi_headline_2000_2026.csv')):
        alt_path = os.path.join(data_folder, 'extracted_capstone_zip')
        if os.path.exists(alt_path):
            data_folder = alt_path

    # -------------------------------------------------------------------------
    # 1. Load Primary Target Variable (Headline CPI)
    # -------------------------------------------------------------------------
    cpi_path = os.path.join(data_folder, 'imf_india_cpi_headline_2000_2026.csv')
    df_cpi = pd.read_csv(cpi_path)
    df_cpi['Date'] = pd.to_datetime(df_cpi['Date'])
    print(f"[1/8] Loaded IMF Headline CPI: {df_cpi.shape}")

    # -------------------------------------------------------------------------
    # 2. Load Global US Inflation Control
    # -------------------------------------------------------------------------
    us_path = os.path.join(data_folder, 'CPI.csv')
    df_us = pd.read_csv(us_path)
    df_us['Date'] = pd.to_datetime(df_us['Date'])
    df_us = df_us[['Date', 'USA_CPI']]
    print(f"[2/8] Loaded US CPI Control: {df_us.shape}")

    # -------------------------------------------------------------------------
    # 3. Load NASA POWER / IMD Climate Indicators
    # -------------------------------------------------------------------------
    climate_path = os.path.join(data_folder, 'india_climate_historical_hybrid_2000_2025.csv')
    df_clim = pd.read_csv(climate_path)
    # Parse date using dayfirst=True to handle DD-MM-YYYY format accurately
    df_clim['Date'] = pd.to_datetime(df_clim['Date'], dayfirst=True)
    print(f"[3/8] Loaded Climate Grid: {df_clim.shape}")

    # -------------------------------------------------------------------------
    # 4. Load WPI Wholesale Food Commodities (Levels & YoY)
    # -------------------------------------------------------------------------
    wpi_path = os.path.join(data_folder, 'wpi_food_basket_and_ceda_cross_validation_2000_2026.csv')
    df_wpi = pd.read_csv(wpi_path)
    df_wpi['Date'] = pd.to_datetime(df_wpi['Date'])
    print(f"[4/8] Loaded WPI Food Commodities: {df_wpi.shape}")

    # -------------------------------------------------------------------------
    # 5. Load Global Macro & Energy Shocks
    # -------------------------------------------------------------------------
    global_path = os.path.join(data_folder, 'global_macro_essential_features_2000_2025.csv')
    df_global = pd.read_csv(global_path)
    df_global['Date'] = pd.to_datetime(df_global['Date'])
    print(f"[5/8] Loaded Global Commodities: {df_global.shape}")

    # -------------------------------------------------------------------------
    # 6. Load Industrial Production Index (IIP General)
    # -------------------------------------------------------------------------
    iip_path = os.path.join(data_folder, 'iip_general_2000_2025_spliced.csv')
    df_iip = pd.read_csv(iip_path)
    df_iip['Date'] = pd.to_datetime(df_iip['Date'])
    print(f"[6/8] Loaded IIP Industrial Activity: {df_iip.shape}")

    # -------------------------------------------------------------------------
    # 7. Load Broad Money Supply (M3)
    # -------------------------------------------------------------------------
    m3_path = os.path.join(data_folder, 'MABMM301INM189N_2000_2026_complete.csv')
    df_m3 = pd.read_csv(m3_path)
    df_m3['Date'] = pd.to_datetime(df_m3['observation_date'])
    df_m3 = df_m3[['Date', 'MABMM301INM189N', 'm3_money_supply_yoy']]
    print(f"[7/8] Loaded Money Supply M3: {df_m3.shape}")

    # -------------------------------------------------------------------------
    # 8. Load RBI Repo Rate (Policy Stance)
    # -------------------------------------------------------------------------
    repo_path = os.path.join(data_folder, 'rbi_repo_rate_2000_2026.csv')
    df_repo = pd.read_csv(repo_path)
    df_repo['Date'] = pd.to_datetime(df_repo['Date'])
    print(f"[8/8] Loaded RBI Repo Rate: {df_repo.shape}")

    # -------------------------------------------------------------------------
    # 9. Sequential Left-Merge onto Primary Headline CPI Timeline
    # -------------------------------------------------------------------------
    df_master = df_cpi.copy()
    datasets = [df_us, df_clim, df_wpi, df_global, df_iip, df_m3, df_repo]

    for sub_df in datasets:
        df_master = pd.merge(df_master, sub_df, on='Date', how='left')

    # Sort sequentially by date
    df_master = df_master.sort_values('Date').reset_index(drop=True)

    # Convert date to clean ISO string format
    df_master['Date'] = df_master['Date'].dt.strftime('%Y-%m-%d')

    # -------------------------------------------------------------------------
    # 10. Export Master Dataset to Root Directory (../master_capstone_dataset_2000_2026.csv)
    # -------------------------------------------------------------------------
    output_path = '../master_capstone_dataset_2000_2026.csv'
    df_master.to_csv(output_path, index=False)
    print(f"\nSUCCESS: Combined Master CSV exported to root directory: '{output_path}'")
    print(f"Total Rows: {len(df_master)} | Total Columns: {len(df_master.columns)}")

if __name__ == '__main__':
    build_consolidated_master_dataset()