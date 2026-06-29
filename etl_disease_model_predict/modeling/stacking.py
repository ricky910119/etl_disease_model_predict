from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_meta_model(base_oof: pd.DataFrame, y_true: pd.Series, method: str = "ridge"):
    x = base_oof.fillna(0)
    y = pd.Series(y_true).astype(float)
    if method == "elasticnet":
        model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=5000, random_state=9101))
    else:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0]))
    model.fit(x, y)
    return model


def predict_meta(model, base_future: pd.DataFrame) -> np.ndarray:
    pred = model.predict(base_future.fillna(0))
    return np.clip(np.rint(pred), 0, None).astype(int)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.sum(np.abs(y_true))
    wape = np.nan if denom == 0 else np.sum(np.abs(y_true - y_pred)) / denom
    bias = float(np.mean(y_pred - y_true))
    smape = float(np.nanmean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)))
    return {"mae": mae, "rmse": rmse, "wape": wape, "smape": smape, "bias": bias}
