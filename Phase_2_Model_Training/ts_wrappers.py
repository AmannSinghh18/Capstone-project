import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

P = ["cpi_headline_yoy_lag1", "cpi_headline_yoy_lag12"]
MACRO = ["USA_CPI", "iip_general_yoy", "m3_money_supply_yoy", "rbi_repo_rate_yoy_diff"]
COMMODITY = [
    "wpi_primary_food_articles_yoy", "wpi_manufactured_food_yoy", "wpi_wheat_yoy", "wpi_rice_yoy", "wpi_maize_yoy",
    "wpi_gram_chana_yoy", "wpi_arhar_tur_yoy", "wpi_moong_yoy", "wpi_urad_yoy", "wpi_masur_yoy", "wpi_tomato_yoy",
    "wpi_onion_yoy", "wpi_potato_yoy", "wpi_brinjal_yoy", "wpi_cabbage_yoy", "wpi_garlic_yoy", "wpi_mustard_oil_yoy",
    "wpi_soybean_oil_yoy", "wpi_groundnut_oil_yoy", "wpi_milk_yoy", "wpi_egg_yoy", "wpi_sugar_yoy",
    "crude_oil_brent_usd_yoy", "palm_oil_usd_yoy", "global_fertilizer_index_yoy"
]
CLIMATE_RAW = [
    "rainfall_total_mm", "rainfall_rate_mm_per_day", "temp_mean_celsius", "relative_humidity_2m_pct",
    "surface_soil_wetness_index", "wind_speed_10m_ms", "solar_irradiance_kwm2"
]
CLIMATE = CLIMATE_RAW + [f"{x}_anom" for x in CLIMATE_RAW]

GROUPS = {
    "Macro": MACRO,
    "Commodity": COMMODITY,
    "Climate": CLIMATE
}


class RegularizedARDLWrapper(BaseEstimator, RegressorMixin):
    """
    High-Order Regularized Autoregressive Distributed Lag (ARDL(p, q)) Model.
    Captures multi-month delayed pass-through shocks from agricultural and commodity markets
    using distributed lags (q=1, 2) regularized via ElasticNet shrinkage.
    """
    def __init__(self, q_lags=2, alpha=0.5, l1_ratio=0.1):
        self.q_lags = q_lags
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=42)
        self.feature_names_in_ = None

    def get_params(self, deep=True):
        return {"q_lags": self.q_lags, "alpha": self.alpha, "l1_ratio": self.l1_ratio}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _build_lags(self, X_df):
        cols = list(X_df.columns)
        lagged_dfs = [X_df.copy()]
        exog_cols = [c for c in cols if not c.startswith("cpi_headline_yoy_lag")]
        for lag in range(1, self.q_lags + 1):
            sh = X_df[exog_cols].shift(lag)
            sh.columns = [f"{c}_lag{lag}" for c in exog_cols]
            lagged_dfs.append(sh)
        X_out = pd.concat(lagged_dfs, axis=1)
        return X_out.bfill().fillna(0.0)

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        X_aug = self._build_lags(X_df)
        X_scaled = self.scaler.fit_transform(X_aug)
        self.model.fit(X_scaled, y)
        self.aug_cols_ = list(X_aug.columns)
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        X_aug = self._build_lags(X_df)
        X_scaled = self.scaler.transform(X_aug)
        return self.model.predict(X_scaled)


# Backward compatibility alias
RidgeARXWrapper = RegularizedARDLWrapper


class DFMWrapper(BaseEstimator, RegressorMixin):
    """
    Block-Structured Dynamic Factor Model (Block-DFM / FAVAR).
    Decomposes macroeconomic covariates into structural economic blocks (Global Macro,
    Domestic Food Commodity, and Agro-Climate), extracting block-specific dynamic factors
    to prevent collinearity and capture structural macroeconomic shock propagation.
    """
    def __init__(self, factors_per_block=2, alpha=2.0):
        self.factors_per_block = factors_per_block
        self.alpha = alpha
        self.scalers = {}
        self.pcas = {}
        self.reg = Ridge(alpha=self.alpha, random_state=42)
        self.feature_names_in_ = None

    def get_params(self, deep=True):
        return {"factors_per_block": self.factors_per_block, "alpha": self.alpha}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        y_arr = np.asarray(y)
        
        factor_list = []
        p_cols = [c for c in P if c in X_df.columns]
        if p_cols:
            factor_list.append(X_df[p_cols].values)
            
        for g_name, g_cols in GROUPS.items():
            active_cols = [c for c in g_cols if c in X_df.columns]
            if not active_cols:
                continue
            scaler = StandardScaler()
            X_blk = scaler.fit_transform(X_df[active_cols])
            self.scalers[g_name] = scaler
            
            n_comp = min(self.factors_per_block, len(active_cols), X_blk.shape[0])
            pca = PCA(n_components=n_comp, random_state=42)
            blk_factors = pca.fit_transform(X_blk)
            self.pcas[g_name] = pca
            factor_list.append(blk_factors)
            
        X_fac = np.column_stack(factor_list)
        self.reg.fit(X_fac, y_arr)
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        factor_list = []
        p_cols = [c for c in P if c in X_df.columns]
        if p_cols:
            factor_list.append(X_df[p_cols].values)
            
        for g_name, g_cols in GROUPS.items():
            active_cols = [c for c in g_cols if c in X_df.columns]
            if not active_cols:
                continue
            X_blk = self.scalers[g_name].transform(X_df[active_cols])
            blk_factors = self.pcas[g_name].transform(X_blk)
            factor_list.append(blk_factors)
            
        X_fac = np.column_stack(factor_list)
        return self.reg.predict(X_fac)


# Alias
BlockDFMWrapper = DFMWrapper


class VARXWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha, random_state=42))
        ])
        self.feature_names_in_ = None

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        self.pipeline.fit(X_df, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


class AutoARIMAWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, order=(2, 0, 1)):
        self.order = order
        self.model_res_ = None
        self.scaler_ = StandardScaler()
        self.feature_names_in_ = None

    def get_params(self, deep=True):
        return {"order": self.order}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        X_scaled = self.scaler_.fit_transform(X_df.values)
        model = ARIMA(endog=np.asarray(y), exog=X_scaled, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
        self.model_res_ = model.fit()
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        X_scaled = self.scaler_.transform(X_df.values)
        preds = self.model_res_.forecast(steps=len(X_df), exog=X_scaled)
        return preds


class BatesGrangerCombinationWrapper(BaseEstimator, RegressorMixin):
    """
    Bates-Granger (1969) Optimal Econometric Forecast Combination.
    Calculates the theoretical variance-minimizing non-negative combination weights
    (w_i >= 0, sum w_i = 1) across competing econometric model forecasts via
    chronological out-of-fold validation residuals.
    """
    def __init__(self, val_ratio=0.25):
        self.val_ratio = val_ratio
        self.m_ardl = RegularizedARDLWrapper(q_lags=2, alpha=0.5, l1_ratio=0.1)
        self.m_dfm = BlockDFMWrapper(factors_per_block=2, alpha=2.0)
        self.m_varx = VARXWrapper(alpha=1.0)
        self.weights_ = None
        self.feature_names_in_ = None

    def get_params(self, deep=True):
        return {"val_ratio": self.val_ratio}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
        
        split_idx = int(len(X_df) * (1.0 - self.val_ratio))
        X_sub, y_sub = X_df.iloc[:split_idx], y_ser.iloc[:split_idx]
        X_val, y_val = X_df.iloc[split_idx:], y_ser.iloc[split_idx:]
        
        # Fit on sub-train
        self.m_ardl.fit(X_sub, y_sub)
        self.m_dfm.fit(X_sub, y_sub)
        self.m_varx.fit(X_sub, y_sub)
        
        # OOF Predictions
        p_ardl = self.m_ardl.predict(X_val)
        p_dfm = self.m_dfm.predict(X_val)
        p_varx = self.m_varx.predict(X_val)
        
        P_val = np.column_stack([p_ardl, p_dfm, p_varx])
        y_v = y_val.values
        
        def obj(w):
            return np.mean((y_v - P_val @ w) ** 2)
        
        k = P_val.shape[1]
        w0 = np.ones(k) / k
        bnds = [(0.0, 1.0) for _ in range(k)]
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        res = minimize(obj, w0, bounds=bnds, constraints=cons)
        self.weights_ = res.x
        
        # Refit on full training set
        self.m_ardl.fit(X_df, y_ser)
        self.m_dfm.fit(X_df, y_ser)
        self.m_varx.fit(X_df, y_ser)
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        p_ardl = self.m_ardl.predict(X_df)
        p_dfm = self.m_dfm.predict(X_df)
        p_varx = self.m_varx.predict(X_df)
        P_mat = np.column_stack([p_ardl, p_dfm, p_varx])
        return P_mat @ self.weights_


class SARIMAXWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res_ = None
        self.scaler_ = StandardScaler()
        self.feature_names_in_ = None

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        X_scaled = self.scaler_.fit_transform(X_df.values)
        model = SARIMAX(endog=np.asarray(y), exog=X_scaled, order=self.order, seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        self.model_res_ = model.fit(disp=False)
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        X_scaled = self.scaler_.transform(X_df.values)
        preds = self.model_res_.forecast(steps=len(X_df), exog=X_scaled)
        return preds
