"""
Consolidated Capstone Dataset Merging Pipeline
----------------------------------------------
File Location: preprocessing/combine_all_datasets.py
Input Directory: Capstone-project/data/
Output File: Capstone-project/master_capstone_dataset_2000_2025.csv
"""

import os
import pandas as pd

def build_consolidated_master_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))               # .../preprocessing
    project_root = os.path.dirname(script_dir)                            # .../Capstone-project
    data_folder = os.path.join(project_root, 'data')                       # .../Capstone-project/data

    print(f"Project Root: {project_root}")
    print(f"Reading CSVs from: {data_folder}")

    if not os.path.exists(os.path.join(data_folder, 'imf_india_cpi_headline_2000_2026.csv')):
        alt_path = os.path.join(data_folder, 'extracted_capstone_zip')
        if os.path.exists(alt_path):
            data_folder = alt_path

    # 1. Headline CPI Target
    df_cpi = pd.read_csv(os.path.join(data_folder, 'imf_india_cpi_headline_2000_2026.csv'))
    df_cpi['Date'] = pd.to_datetime(df_cpi['Date'])

    # 2. US CPI Control
    df_us = pd.read_csv(os.path.join(data_folder, 'CPI.csv'))
    df_us['Date'] = pd.to_datetime(df_us['Date'])
    df_us = df_us[['Date', 'USA_CPI']]

    # 3. Climate Grid (Parse DD-MM-YYYY format)
    df_clim = pd.read_csv(os.path.join(data_folder, 'india_climate_historical_hybrid_2000_2025.csv'))
    df_clim['Date'] = pd.to_datetime(df_clim['Date'], dayfirst=True)
    df_clim = df_clim.drop(columns=['temp_max_celsius', 'temp_min_celsius'], errors='ignore')

    # 4. WPI Commodities
    df_wpi = pd.read_csv(os.path.join(data_folder, 'wpi_food_basket_and_ceda_cross_validation_2000_2026.csv'))
    df_wpi['Date'] = pd.to_datetime(df_wpi['Date'])

    # 5. Global Commodities (Parse DD-MM-YYYY format using dayfirst=True)
    global_path = os.path.join(data_folder, 'global_macro_essential_features_2000_2025.csv')
    if not os.path.exists(global_path):
        global_path = 'global_macro_essential_features_2000_2025.csv'

    df_global = pd.read_csv(global_path)
    df_global['Date'] = pd.to_datetime(df_global['Date'], dayfirst=True)

    # 6. IIP Activity
    df_iip = pd.read_csv(os.path.join(data_folder, 'iip_general_2000_2025_spliced.csv'))
    df_iip['Date'] = pd.to_datetime(df_iip['Date'])

    # 7. Money Supply M3
    df_m3 = pd.read_csv(os.path.join(data_folder, 'MABMM301INM189N_2000_2026_complete.csv'))
    df_m3['Date'] = pd.to_datetime(df_m3['observation_date'])
    df_m3 = df_m3[['Date', 'MABMM301INM189N', 'm3_money_supply_yoy']]

    # 8. RBI Repo Rate
    df_repo = pd.read_csv(os.path.join(data_folder, 'rbi_repo_rate_2000_2026.csv'))
    df_repo['Date'] = pd.to_datetime(df_repo['Date'])

    # Sequential Left-Merge onto Primary Headline CPI Timeline
    df_master = df_cpi.copy()
    datasets = [df_us, df_clim, df_wpi, df_global, df_iip, df_m3, df_repo]

    for sub_df in datasets:
        df_master = pd.merge(df_master, sub_df, on='Date', how='left')

    df_master = df_master.sort_values('Date').reset_index(drop=True)

    # Truncate at Dec 2025
    df_master = df_master[df_master['Date'] <= '2025-12-01'].copy()
    if 'USA_CPI' in df_master.columns:
        df_master['USA_CPI'] = df_master['USA_CPI'].ffill()

    df_master['Date'] = df_master['Date'].dt.strftime('%Y-%m-%d')

    output_path = os.path.join(project_root, 'master_capstone_dataset_2000_2025.csv')
    df_master.to_csv(output_path, index=False)

    print(f"SUCCESS: Consolidated Master Dataset exported to: {output_path}")
    print(f"Time Horizon: {df_master['Date'].min()} to {df_master['Date'].max()}")
    print(f"Total Rows: {len(df_master)} | Total Columns: {len(df_master.columns)}")

if __name__ == '__main__':
    build_consolidated_master_dataset()