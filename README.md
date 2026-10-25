# Capstone Empirical Machine Learning & Time Series Econometric Investigation of India's Headline CPI Inflation

An end-to-end, dual-paradigm empirical framework for forecasting and diagnosing the drivers of India's **Headline Consumer Price Index (CPI) Year-over-Year (YoY) Inflation**.

This project implements a **5-Phase Modular Architecture** across two distinct computational paradigms:
1. **Machine Learning (ML) Paradigm** (ElasticNet, Random Forest, SVR, XGBoost, LightGBM, and Time-Series Stacking Ensemble with SHAP & GPFI attributions)
2. **Time Series (TS) Econometric Paradigm** (SARIMAX, AutoARIMA, VARX, Ridge-ARX, and Dynamic Factor Models with SVAR Impulse Responses & FEVD)

---

## 📁 Repository Directory Structure

```
Capstone-project/
├── README.md                                 # Complete Project Overview & Methodology Documentation
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
├── Phase_2_ML_Models/                        # Machine Learning Model Estimation & Ensembling
│   ├── ElasticNet.ipynb                      # ElasticNet grid search & checkpoint fitting
│   ├── RandomForest.ipynb                    # Random Forest grid search & checkpoint fitting
│   ├── SVR.ipynb                             # SVR grid search & checkpoint fitting
│   ├── XGBoost.ipynb                         # XGBoost grid search & checkpoint fitting
│   └── train_lgbm_and_stacking.py            # LightGBM & Time-Series Stacking Ensemble models
│
├── Phase_2_TS_Models/                        # Time Series Model Estimation & Wrappers
│   ├── __init__.py                           # Package initialization
│   ├── ts_wrappers.py                        # Scikit-learn compliant wrappers for SARIMAX, AutoARIMA, VARX, Ridge-ARX, DFM
│   ├── run_phase2_ts_models.py               # Fits 16 baseline TS model-config combinations
│   └── run_phase2_dfm_models.py              # Fits Dynamic Factor Models (FAVAR) across 4 configs
│
├── phase_3_ML_Accuracy.py                    # Phase 3 ML: Test set accuracy (RMSE, MAE, MAPE, R²) & HLN DM tests
├── phase_3_TS_Accuracy.py                    # Phase 3 TS: Test set accuracy (RMSE, MAE, MAPE, R²) & HLN DM tests
│
├── Phase_3_MultiHorizon/                     # Multi-Horizon Direct Forecasting (h=1, 3, 6, 12 months)
│   └── run_multi_horizon_forecasting.py      # Evaluates term-structure transmission across feature blocks
│
├── Phase_3_Evaluation_Advanced/              # Robustness & Uncertainty Quantification
│   └── run_walk_forward_and_conformal.py     # Rolling walk-forward backtesting & 90%/95% Conformal Prediction Intervals
│
├── Phase_4_Feature_Attribution/              # Feature Attribution & Stability Analysis
│   ├── run_phase4_metrics.py                 # ML SHAP values, GPFI permutation tests & stability metrics
│   └── run_phase4_ts_metrics.py              # TS Econometric coefficient shares & GPFI metrics
│
├── Phase_4_TS_Structural_Analysis/           # SVAR Structural Shock Transmission
│   └── run_structural_irf_fevd.py            # Orthogonalized Impulse Responses (IRF) & Variance Decomposition (FEVD)
│
├── Phase_5_Diagnostic_Synthesis/             # Multi-Paradigm Triangulation & Cross-Convergence
│   ├── run_phase5_metrics.py                 # ML diagnostic triangulation & cross-paradigm synthesis
│   └── run_phase5_ts_metrics.py              # TS diagnostic triangulation & consensus
│
├── data/                                     # Train/Test Processed Datasets
│   ├── train_processed.csv                   # 80% Chronological training set (2001-2020)
│   └── test_processed.csv                    # 20% Chronological testing set (2021-2025)
│
├── models/                                   # Checkpoints for 44 Serialized Models (.joblib)
├── outputs/tables/                           # Exported CSV Result Tables
└── preprocessing/                            # Raw data sources (CPI, IIP, WPI, Climate, Global Macro)
```

---

## 📊 Dataset & Feature Specifications

- **Target Variable**: `cpi_headline_yoy` (India Consumer Price Index YoY Inflation Rate)
- **Feature Categories (45 substantive indicators)**:
  - **Commodity & Food Basket (25 features)**: Primary food articles, wheat, rice, maize, pulses (gram, tur, moong, urad, masur), vegetables (tomato, onion, potato, brinjal, cabbage, garlic), edible oils (mustard, soybean, groundnut), dairy, egg, sugar, Brent crude oil, palm oil, global fertilizer index.
  - **Macroeconomic (4 features)**: US CPI YoY, India IIP (Index of Industrial Production), M3 Money Supply YoY, RBI Repo Rate YoY diff.
  - **Climate & Weather (14 features)**: 7 raw metrics (rainfall, temperature, humidity, surface soil wetness, wind speed, solar irradiance) + 7 seasonal Z-score anomaly metrics.
  - **Persistence (2 features)**: CPI YoY Lag 1 and Lag 12.
- **Train/Test Split**: Chronological 80% Train, 20% Out-of-sample Test.

---

## 🔬 Dual-Paradigm Methodology & Key Findings

### 1. Machine Learning (ML) Paradigm
- **Models**: ElasticNet, Random Forest, SVR, XGBoost, LightGBM, and Stacking Ensemble (Super Learner).
- **Top Out-of-Sample Model**: **TimeSeriesStackingRegressor (Full config)** achieving **RMSE: 0.6343**, **MAE: 0.4674**, **MAPE: 13.61%**, **$R^2$: 0.8581**, outperforming the baseline ElasticNet (RMSE: 0.6679, $R^2$: 0.8427).
- **Diagnostic Classification**:
  - **Commodity**: **Core Driver** (69.61% SHAP share, Rank 1 standalone RMSE 0.8011, positive GPFI loss significant in 4/4 models).
  - **Macroeconomic**: **Non-Linear Synergy** (21.49% SHAP share, Rank 2 standalone RMSE 0.8595, positive GPFI loss in 2/4 models).
  - **Climate**: **Mixed / Inconclusive** (8.90% SHAP share, Rank 3 standalone RMSE 1.0653).

### 2. Time Series (TS) Econometric Paradigm
- **Models**: SARIMAX, AutoARIMA, VARX, Ridge-ARX, and Dynamic Factor Models (FAVAR).
- **Top Out-of-Sample Model**: **DFM (Macro_only config)** achieving **RMSE: 0.7853**, **$R^2$: 0.7825**, and **Ridge_ARX (Macro_only)** achieving **RMSE: 0.7864**, **$R^2$: 0.7819**.
- **Diagnostic Classification**:
  - **Macroeconomic**: **Synergistic Driver** (30.54% Econometric share, Rank 1 standalone RMSE 0.8033).
  - **Commodity**: **Synergistic Driver** (28.37% Econometric share, Rank 2 standalone RMSE 1.0032).
  - **Climate**: **Mixed / Inconclusive** (41.09% Econometric share, Rank 3 standalone RMSE 1.2630).

### 3. Multi-Horizon Forecasting ($h = 1, 3, 6, 12$ Months)
- **Short-Horizon Dominance ($h = 1$)**: Commodity features achieve the lowest RMSE (**1.1778**), driving immediate food-price spikes.
- **Medium/Long-Horizon Dominance ($h = 3, 6, 12$)**: Macroeconomic features dominate:
  - $h = 3$: Macro_only RMSE **1.3709** vs. Commodity_only 1.5569.
  - $h = 6$: Macro_only RMSE **1.6498** vs. Commodity_only 1.8691.
  - $h = 12$: Macro_only RMSE **1.9978** vs. Commodity_only 2.4059.

### 4. Structural Econometric Decomposition (SVAR IRF & FEVD)
- **Forecast Error Variance Decomposition (FEVD)** shows that domestic Primary Food articles account for **40.21%** of CPI variance at 1 month, peaking at **59.32%** at 3 months, and remaining above **43%** through 12 months.
- Direct climate shocks (Rainfall) account for under 0.25% direct variance, confirming that weather variations are largely transmitted downstream through food articles.

### 5. Uncertainty Quantification (Conformal Prediction)
- **Nominal 90% Target Coverage**:
  - ElasticNet: **94.8%** empirical coverage (Interval Width: $\pm 1.51\%$).
  - LightGBM: **91.4%** empirical coverage (Interval Width: $\pm 1.80\%$).
  - XGBoost: **89.7%** empirical coverage (Interval Width: $\pm 1.86\%$).

---

## 🚀 Execution Guide

Run any phase or advanced module via standard terminal commands:

### Phase 1: Data Preparation & Diagnostics
```bash
python Phase_1_Data_Prep/generate_datasets.py
python Phase_1_TS_Diagnostics/run_phase1_ts_diagnostics.py
```

### Phase 2: Model Estimation
```bash
# Baseline TS Models
python Phase_2_TS_Models/run_phase2_ts_models.py

# Advanced Dynamic Factor Models (FAVAR)
python Phase_2_TS_Models/run_phase2_dfm_models.py

# Advanced ML Models (LightGBM & Stacking Ensemble)
python Phase_2_ML_Models/train_lgbm_and_stacking.py
```

### Phase 3: Out-of-Sample Accuracy & Multi-Horizon Evaluation
```bash
# Standard ML Test Accuracy (6 Model Families)
python phase_3_ML_Accuracy.py

# Standard TS Test Accuracy (5 Model Families)
python phase_3_TS_Accuracy.py

# Direct Multi-Horizon Forecasting (h=1, 3, 6, 12 months)
python Phase_3_MultiHorizon/run_multi_horizon_forecasting.py

# Rolling Walk-Forward Backtesting & Conformal Prediction Intervals
python Phase_3_Evaluation_Advanced/run_walk_forward_and_conformal.py
```

### Phase 4: Feature Attribution & Structural SVAR Analysis
```bash
# ML Feature Attribution (SHAP, Permutation GPFI, Stability)
python Phase_4_Feature_Attribution/run_phase4_metrics.py

# Time Series Feature Attribution (Coefficients, GPFI)
python Phase_4_Feature_Attribution/run_phase4_ts_metrics.py

# Structural SVAR Impulse Responses (IRF) & Variance Decomposition (FEVD)
python Phase_4_TS_Structural_Analysis/run_structural_irf_fevd.py
```

### Phase 5: Diagnostic Triangulation & Cross-Paradigm Convergence
```bash
# ML Triangulation & Cross-Paradigm Convergence
python Phase_5_Diagnostic_Synthesis/run_phase5_metrics.py

# Time Series Triangulation & Consensus
python Phase_5_Diagnostic_Synthesis/run_phase5_ts_metrics.py
```

---

## 📈 Output Tables Inventory

All analytical tables are saved in `outputs/tables/`:

| Output Table | Description |
| :--- | :--- |
| `phase1_ts_stationarity_diagnostics.csv` | ADF and KPSS stationarity test statistics & p-values for all 45 features. |
| `phase3_ml_accuracy_table.csv` | Out-of-sample RMSE, MAE, MAPE, and $R^2$ across 6 ML model families. |
| `phase3_ml_dm_significance_table.csv` | Harvey-Leybourne-Newbold Diebold-Mariano tests with BH-FDR p-values (ML). |
| `phase3_ts_accuracy_table.csv` | Out-of-sample RMSE, MAE, MAPE, and $R^2$ across 5 TS model families. |
| `phase3_ts_dm_significance_table.csv` | Diebold-Mariano configuration comparisons with BH-FDR correction (TS). |
| `phase3_multi_horizon_accuracy.csv` | Out-of-sample accuracy across $h \in \{1, 3, 6, 12\}$ months. |
| `phase3_walk_forward_backtest_accuracy.csv`| Rolling expanding-window backtesting performance from 2016 to 2025. |
| `phase3_conformal_prediction_intervals.csv`| Empirical coverage rates and interval widths for 90% and 95% confidence bands. |
| `phase4_ml_gpfi.csv` / `phase4_ml_native_attribution.csv` | Grouped Permutation Feature Importance & SHAP shares across ML models. |
| `phase4_econometric_gpfi.csv` / `phase4_econometric_native_attribution.csv` | Econometric coefficient shares and permutation importance across TS models. |
| `phase4_ts_irf_table.csv` | 12-month SVAR Orthogonalized Impulse Response Functions with 95% error bands. |
| `phase4_ts_fevd_table.csv` | Forecast Error Variance Decomposition across 1, 3, 6, and 12-month horizons. |
| `phase5_ml_within_paradigm_consensus.csv` | Within-paradigm driver ranking consensus for ML models. |
| `phase5_ml_diagnostic_triangulation.csv` | Core/Synergistic diagnostic categorization for ML paradigm. |
| `phase5_ts_within_paradigm_consensus.csv` | Within-paradigm driver ranking consensus for TS models. |
| `phase5_ts_diagnostic_triangulation.csv` | Core/Synergistic diagnostic categorization for TS paradigm. |
| `phase5_ml_cross_paradigm_convergence.csv`| Cross-paradigm rank convergence between Machine Learning and Econometrics. |
