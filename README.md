# Capstone Empirical Machine Learning & Time Series Econometric Investigation of India's Headline CPI Inflation

An end-to-end, dual-paradigm empirical framework for forecasting and diagnosing the drivers of India's **Headline Consumer Price Index (CPI) Year-over-Year (YoY) Inflation**.

This project implements a clean **5-Phase Unified Modular Architecture** integrating two distinct computational paradigms:
1. **Machine Learning (ML) Paradigm** (TimeSeriesStackingRegressor, ElasticNet, XGBoost, and SVR with SHAP & GPFI attributions)
2. **Time Series (TS) Econometric Paradigm** (Bates-Granger Optimal Forecast Combination, Block-Structured Dynamic Factor Models, Regularized ARDL(p, q), and VARX with SVAR Impulse Responses & FEVD)

---

## 📁 Repository Directory Structure

```
Capstone-project/
├── README.md                                 # Complete Project Overview & Methodology Documentation
├── master_capstone_dataset_2000_2026.csv      # Consolidated raw time-series dataset (2000-2026)
├── processed_capstone_dataset_2001_2025.csv   # Primary processed dataset (2001-2025)
│
├── Phase_1_Data_and_Diagnostics/             # Phase 1: Data Integration & Stationarity Diagnostics
│   ├── generate_datasets.py                  # Anomaly z-scores & 80/20 train/test split generator
│   ├── run_phase1_ts_diagnostics.py          # Stationarity ADF and KPSS tests for all 45 features
│   ├── master.ipynb                          # Notebook consolidating 2000-2026 dataset
│   └── preprocessing.ipynb                   # Notebook consolidating 2001-2025 dataset
│
├── Phase_2_Model_Training/                   # Phase 2: ML & Econometric Model Estimation
│   ├── ElasticNet.ipynb                      # ElasticNet grid search & checkpoint fitting
│   ├── RandomForest.ipynb                    # Random Forest grid search & checkpoint fitting
│   ├── SVR.ipynb                             # SVR grid search & checkpoint fitting
│   ├── XGBoost.ipynb                         # XGBoost grid search & checkpoint fitting
│   ├── train_lgbm_and_stacking.py            # LightGBM & Time-Series Stacking Ensemble models
│   ├── ts_wrappers.py                        # Scikit-learn compliant wrappers (BatesGrangerCombination, BlockDFM, RegularizedARDL, VARX)
│   ├── train_ts_models.py                    # Trains & serializes the streamlined 4 TS model architectures across 4 configs
│   ├── run_phase2_ts_models.py               # Fits baseline TS model-config combinations
│   └── run_phase2_dfm_models.py              # Fits Dynamic Factor Models (FAVAR) across 4 configs
│
├── Phase_3_Forecast_and_Evaluation/          # Phase 3: Accuracy, Multi-Horizon & Robustness Evaluation
│   ├── run_phase3_ml_accuracy.py             # ML test set accuracy (RMSE, MAE, MAPE, R²) & HLN DM tests
│   ├── run_phase3_ts_accuracy.py             # TS test set accuracy (RMSE, MAE, MAPE, R²) & HLN DM tests
│   ├── generate_paper_figures.py             # Publication figure generator (Loss curves, forecasts, GPFI)
│   ├── run_multi_horizon_forecasting.py      # Multi-horizon direct forecasting (h=1, 3, 6, 12 months)
│   └── run_walk_forward_and_conformal.py     # Rolling walk-forward backtesting & 90%/95% Conformal Prediction intervals
│
├── Phase_4_Attribution_and_Structural/       # Phase 4: Feature Attribution & SVAR Structural Analysis
│   ├── run_phase4_metrics.py                 # ML SHAP values, GPFI permutation tests & stability metrics
│   ├── run_phase4_ts_metrics.py              # TS Econometric coefficient shares & GPFI metrics
│   └── run_structural_irf_fevd.py            # SVAR Orthogonalized Impulse Responses (IRF) & Variance Decomposition (FEVD)
│
├── Phase_5_Diagnostic_Synthesis/             # Phase 5: Multi-Paradigm Triangulation & Cross-Convergence
│   ├── run_phase5_metrics.py                 # ML diagnostic triangulation & cross-paradigm synthesis
│   └── run_phase5_ts_metrics.py              # TS diagnostic triangulation & consensus
│
├── data/                                     # Train/Test Processed Datasets
│   ├── train_processed.csv                   # 80% Chronological training set (2001-2020)
│   └── test_processed.csv                    # 20% Chronological testing set (2021-2025)
│
├── models/                                   # Checkpoints for 32 Serialized Models (.joblib) (16 ML + 16 TS)
├── outputs/                                  # Exported Analytical Tables & Figures
│   ├── tables/                               # Consolidated tables across all phases
│   └── figures/                              # High-resolution (300 DPI) publication figures
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
- **Streamlined 4-Model Core**: **TimeSeriesStackingRegressor** (Ensemble), **ElasticNet**, **XGBoost**, and **SVR**.
- **Top Out-of-Sample Model**: **TimeSeriesStackingRegressor (Full config)** achieving **RMSE: 0.6343**, **MAE: 0.4674**, **MAPE: 13.61%**, **$R^2$: 0.8581**, outperforming the baseline ElasticNet (RMSE: 0.6679, $R^2$: 0.8427).
- **Diagnostic Classification**:
  - **Commodity**: **Core Driver** (68.43% SHAP share, Rank 1 standalone RMSE 0.7604, positive GPFI loss significant in 4/4 models).
  - **Macroeconomic**: **Redundant Driver** (24.34% SHAP share, Rank 2 standalone RMSE 0.8295, positive GPFI loss in 1/4 models).
  - **Climate**: **Mixed / Inconclusive** (7.23% SHAP share, Rank 3 standalone RMSE 1.0117).

### 2. Time Series (TS) Econometric Paradigm
- **Streamlined 4-Model Core**:
  - **Bates-Granger (1969) Optimal Econometric Combination**: Constrained optimization solving for variance-minimizing forecast weights across econometric base models.
  - **Block-Structured Dynamic Factor Models (Block-DFM / FAVAR)**: Structural decomposition extracting dynamic latent factors across Macro, Food Commodity, and Climate blocks.
  - **Regularized ARDL($p, q$)**: High-order Autoregressive Distributed Lag model capturing multi-month delayed pass-through shocks ($q \in \{1, 2\}$) under ElasticNet regularization.
  - **VARX**: Vector Autoregression with exogenous covariate vectors.
- **Definitive TS Champion**: **Bates-Granger Optimal Combination** achieving **RMSE: 0.7853**, **$R^2$: 0.7825** on `Macro_only`, and dominating the `Full` specification with **RMSE: 0.8023** ($R^2$: 0.7730).
- **Delayed Pass-Through Gains**: **Regularized ARDL($p, q$)** cuts `Full` config RMSE from 0.8857 down to **0.8131** ($R^2$: 0.7668) by capturing distributed multi-month agricultural/commodity shocks.
- **Structural Causal Diagnostic**: **SVAR** (Structural Vector Autoregression) retained in Phase 4 estimating Orthogonalized Impulse Response Functions (IRF) and Forecast Error Variance Decomposition (FEVD).
- **Diagnostic Classification**:
  - **Macroeconomic**: **Rank 1 Native Driver** (49.33% Econometric share, Rank 1 standalone RMSE 0.8130).
  - **Commodity**: **Rank 2 Native Driver** (32.69% Econometric share, Rank 2 standalone RMSE 0.8554, with major delayed pass-through).
  - **Climate**: **Rank 3 Driver** (17.98% Econometric share, Rank 3 standalone RMSE 0.9690).

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

Run any phase via standard terminal commands:

### Phase 1: Data Preparation & Diagnostics
```bash
python Phase_1_Data_and_Diagnostics/generate_datasets.py
python Phase_1_Data_and_Diagnostics/run_phase1_ts_diagnostics.py
```

### Phase 2: Model Estimation
```bash
# Time Series Models (Bates-Granger, Block-DFM, Regularized ARDL, VARX)
python Phase_2_Model_Training/train_ts_models.py

# Advanced ML Models (Stacking Ensemble, ElasticNet, XGBoost, SVR)
python Phase_2_Model_Training/train_lgbm_and_stacking.py
```

### Phase 3: Out-of-Sample Accuracy & Multi-Horizon Evaluation
```bash
# ML Test Set Accuracy (4 Streamlined Model Families)
python Phase_3_Forecast_and_Evaluation/run_phase3_ml_accuracy.py

# TS Test Set Accuracy (4 Streamlined Model Families)
python Phase_3_Forecast_and_Evaluation/run_phase3_ts_accuracy.py

# Direct Multi-Horizon Forecasting (h=1, 3, 6, 12 months)
python Phase_3_Forecast_and_Evaluation/run_multi_horizon_forecasting.py

# Rolling Walk-Forward Backtesting & Conformal Prediction Intervals
python Phase_3_Forecast_and_Evaluation/run_walk_forward_and_conformal.py
```

### Phase 4: Feature Attribution & Structural SVAR Analysis
```bash
# ML Feature Attribution (SHAP, Permutation GPFI, Stability)
python Phase_4_Attribution_and_Structural/run_phase4_metrics.py

# Time Series Feature Attribution (Coefficients, GPFI)
python Phase_4_Attribution_and_Structural/run_phase4_ts_metrics.py

# Structural SVAR Impulse Responses (IRF) & Variance Decomposition (FEVD)
python Phase_4_Attribution_and_Structural/run_structural_irf_fevd.py
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
| `phase3_ml_accuracy_table.csv` | Out-of-sample RMSE, MAE, MAPE, and $R^2$ across 4 ML model families. |
| `phase3_ml_dm_significance_table.csv` | Harvey-Leybourne-Newbold Diebold-Mariano tests with BH-FDR p-values (ML). |
| `phase3_ts_accuracy_table.csv` | Out-of-sample RMSE, MAE, MAPE, and $R^2$ across 4 TS model families. |
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
