# Capstone Empirical Machine Learning & Time Series Econometric Investigation of India's Headline CPI Inflation

An end-to-end, dual-paradigm empirical framework for forecasting and diagnosing the drivers of India's **Headline Consumer Price Index (CPI) Year-over-Year (YoY) Inflation**.

This project implements a **5-Phase Modular Pipeline** across two distinct computational paradigms:
1. **Machine Learning (ML) Paradigm** (ElasticNet, Random Forest, SVR, XGBoost with SHAP & GPFI attributions)
2. **Time Series (TS) Econometric Paradigm** (SARIMAX, AutoARIMA, VARX, Ridge-ARX with Coefficient & GPFI attributions)

---

## 📁 Repository Directory Structure

```
Capstone-project/
├── README.md                                 # Complete Project Overview & Instructions
├── master_capstone_dataset_2000_2026.csv      # Consolidated raw time-series dataset (2000-2026)
├── processed_capstone_dataset_2001_2025.csv   # Primary processed dataset (2001-2025)
│
├── Phase_1_Data_Prep/                        # ML Data Integration & Feature Engineering
│   ├── generate_datasets.py                  # Anomaly z-scores & 80/20 train/test split generator
│   ├── master.ipynb                          # Notebook consolidating 2000-2026 dataset
│   └── preprocessing.ipynb                   # Notebook consolidating 2001-2025 dataset
│
├── Phase_1_TS_Diagnostics/                   # Time Series Diagnostics & Stationarity
│   └── run_phase1_ts_diagnostics.py          # Stationarity ADF and KPSS tests for all 45 features
│
├── Phase_2_ML_Models/                        # Machine Learning Model Estimation & Hyperparameter CV
│   ├── ElasticNet.ipynb                      # ElasticNet grid search & checkpoint fitting
│   ├── RandomForest.ipynb                    # Random Forest grid search & checkpoint fitting
│   ├── SVR.ipynb                             # SVR grid search & checkpoint fitting
│   └── XGBoost.ipynb                         # XGBoost grid search & checkpoint fitting
│
├── Phase_2_TS_Models/                        # Time Series Model Estimation & Wrappers
│   ├── __init__.py                           # Package initialization
│   ├── ts_wrappers.py                        # Scikit-learn compliant wrappers for SARIMAX, AutoARIMA, VARX, Ridge-ARX
│   └── run_phase2_ts_models.py               # Fits all 16 TS model-config combinations
│
├── phase_3_ML_Accuracy.py                    # Phase 3 ML: Test set accuracy (RMSE, MAE, MAPE, R²) & HLN DM tests
├── phase_3_TS_Accuracy.py                    # Phase 3 TS: Test set accuracy (RMSE, MAE, MAPE, R²) & HLN DM tests
│
├── Phase_4_Feature_Attribution/              # Feature Attribution & Stability Analysis
│   ├── run_phase4_metrics.py                 # ML SHAP values, GPFI permutation tests & stability metrics
│   └── run_phase4_ts_metrics.py              # TS Econometric coefficient shares & GPFI metrics
│
├── Phase_5_Diagnostic_Synthesis/             # Multi-Paradigm Triangulation & Cross-Convergence
│   ├── run_phase5_metrics.py                 # ML diagnostic triangulation & cross-paradigm synthesis
│   └── run_phase5_ts_metrics.py              # TS diagnostic triangulation & consensus
│
├── data/                                     # Train/Test Processed Datasets
│   ├── train_processed.csv                   # 80% Chronological training set
│   └── test_processed.csv                    # 20% Chronological testing set
│
├── models/                                   # Checkpoints for 32 Serialized Models (.joblib)
├── outputs/tables/                           # Exported CSV Result Tables
└── preprocessing/                            # Raw data sources (CPI, IIP, WPI, Climate, Global Macro)
```

---

## 📊 Dataset & Feature Specifications

- **Target Variable**: `cpi_headline_yoy` (India Consumer Price Index YoY Inflation Rate)
- **Feature Categories**:
  - **Commodity & Food Basket (25 features)**: Primary food articles, wheat, rice, pulses, vegetables (tomato, onion, potato), edible oils, crude oil (Brent), palm oil, global fertilizer index.
  - **Macroeconomic (4 features)**: US CPI, India IIP (Index of Industrial Production), M3 Money Supply, RBI Repo Rate YoY diff.
  - **Climate & Weather (14 features)**: 7 raw metrics (rainfall, temperature, humidity, wetness, wind, solar) + 7 seasonal Z-score anomaly metrics.
  - **Persistence (2 features)**: CPI YoY Lag 1 and Lag 12.
- **Train/Test Split**: 80% Train, 20% Out-of-sample Test (chronological time-series split).

---

## 🔬 Dual-Paradigm Methodology & Results

### 1. Machine Learning (ML) Paradigm
- **Models**: ElasticNet, Random Forest, Support Vector Regression (SVR), XGBoost.
- **Top Out-of-Sample Model**: `ElasticNet` (`Full` config) achieving **RMSE: 0.6679**, **MAE: 0.5367**, **$R^2$: 0.8427**.
- **Diagnostic Classification**:
  - **Commodity**: **Core Driver** (69.61% SHAP share, Rank 1 standalone RMSE 0.8011).
  - **Macroeconomic**: **Non-Linear Synergy** (21.49% SHAP share, Rank 2 standalone RMSE 0.8595).
  - **Climate**: **Mixed / Inconclusive** (8.90% SHAP share, Rank 3 standalone RMSE 1.0653).

### 2. Time Series (TS) Econometric Paradigm
- **Models**: SARIMAX, AutoARIMA, VARX, Ridge-ARX.
- **Top Out-of-Sample Model**: `Ridge_ARX` (`Macro_only` config) achieving **RMSE: 0.7864**, **MAE: 0.5713**, **$R^2$: 0.7819**.
- **Diagnostic Classification**:
  - **Macroeconomic**: **Synergistic Driver** (30.54% Econometric share, Rank 1 standalone RMSE 0.8033).
  - **Commodity**: **Synergistic Driver** (28.37% Econometric share, Rank 2 standalone RMSE 1.0032).
  - **Climate**: **Mixed / Inconclusive** (41.09% Econometric share, Rank 3 standalone RMSE 1.2630).

### 3. Cross-Paradigm Convergence Summary

| Domain | ML SHAP Share (%) | ML Rank | Econometric Share (%) | Econometric Rank | Cross-Paradigm Convergence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Macroeconomic** | **21.49%** | **2** | **30.54%** | **2** | **CONVERGED (Rank #2 in both)** |
| **Commodity** | **69.61%** | **1** | **28.37%** | **3** | Non-Linear Lead in ML |
| **Climate** | **8.90%** | **3** | **41.09%** | **1** | Linear Baseline in TS |

---

## 🚀 Execution Guide

You can run any phase of the pipeline independently via terminal commands:

### Data Preparation & Stationarity Diagnostics
```bash
python Phase_1_Data_Prep/generate_datasets.py
python Phase_1_TS_Diagnostics/run_phase1_ts_diagnostics.py
```

### Model Fitting
```bash
# Time Series Models
python Phase_2_TS_Models/run_phase2_ts_models.py
```

### Out-of-Sample Accuracy Evaluation
```bash
# ML Paradigm Accuracy
python phase_3_ML_Accuracy.py

# Time Series Paradigm Accuracy
python phase_3_TS_Accuracy.py
```

### Feature Attribution & Permutation Importance
```bash
# ML Feature Attribution (SHAP & GPFI)
python Phase_4_Feature_Attribution/run_phase4_metrics.py

# Time Series Feature Attribution (Econometric Coefficients & GPFI)
python Phase_4_Feature_Attribution/run_phase4_ts_metrics.py
```

### Multi-Paradigm Diagnostic Triangulation & Convergence
```bash
# ML Triangulation & Cross-Paradigm Convergence
python Phase_5_Diagnostic_Synthesis/run_phase5_metrics.py

# Time Series Triangulation
python Phase_5_Diagnostic_Synthesis/run_phase5_ts_metrics.py
```

---

## 📈 Output Tables Inventory

All results are automatically saved in `outputs/tables/`:
- `phase1_ts_stationarity_diagnostics.csv`
- `phase3_ml_accuracy_table.csv` / `phase3_ml_dm_significance_table.csv`
- `phase3_ts_accuracy_table.csv` / `phase3_ts_dm_significance_table.csv`
- `phase4_ml_gpfi.csv` / `phase4_ml_native_attribution.csv` / `phase4_ml_all_variable_rankings.csv`
- `phase4_econometric_gpfi.csv` / `phase4_econometric_native_attribution.csv` / `phase4_econometric_all_variable_rankings.csv`
- `phase5_ml_within_paradigm_consensus.csv` / `phase5_ml_diagnostic_triangulation.csv`
- `phase5_ts_within_paradigm_consensus.csv` / `phase5_ts_diagnostic_triangulation.csv`
- `phase5_ml_cross_paradigm_convergence.csv`
