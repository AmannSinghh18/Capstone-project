import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

class SARIMAXWrapper:
    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res_ = None
        self.scaler_ = StandardScaler()
        self.feature_names_in_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_val = X.values
        else:
            X_val = np.asarray(X)
        y_val = np.asarray(y)
        
        X_scaled = self.scaler_.fit_transform(X_val)
        model = SARIMAX(endog=y_val, exog=X_scaled, order=self.order, seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        self.model_res_ = model.fit(disp=False)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_val = X.values
        else:
            X_val = np.asarray(X)
        X_scaled = self.scaler_.transform(X_val)
        preds = self.model_res_.forecast(steps=len(X_val), exog=X_scaled)
        return preds


class AutoARIMAWrapper:
    def __init__(self, order=(2, 0, 1)):
        self.order = order
        self.model_res_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X, y):
        X_val = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_val = np.asarray(y)
        X_scaled = self.scaler_.fit_transform(X_val)
        model = ARIMA(endog=y_val, exog=X_scaled, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
        self.model_res_ = model.fit()
        return self

    def predict(self, X):
        X_val = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler_.transform(X_val)
        preds = self.model_res_.forecast(steps=len(X_val), exog=X_scaled)
        return preds


class VARXWrapper:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha, random_state=42))
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


class RidgeARXWrapper:
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha, random_state=42))
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
