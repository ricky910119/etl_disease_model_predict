from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from pmdarima import auto_arima
except Exception:  # pragma: no cover
    auto_arima = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    scope: str  # global_panel, local_series, naive
    factory: Callable[[], object]
    enabled: bool = True


class SeasonalNaiveRegressor:
    """以去年同週優先，否則以最近 season_length 週均值補足。"""

    def __init__(self, season_length: int = 52):
        self.season_length = season_length
        self.y_: list[float] = []
        self.fallback_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.y_ = pd.Series(y).astype(float).dropna().tolist()
        self.fallback_ = float(np.nanmean(self.y_[-self.season_length:])) if self.y_ else 0.0
        if np.isnan(self.fallback_):
            self.fallback_ = 0.0
        return self

    def predict(self, x: pd.DataFrame):
        preds = []
        history = list(self.y_)
        for _ in range(len(x)):
            pred = history[-self.season_length] if len(history) >= self.season_length else self.fallback_
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"), _one_hot_encoder())
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_ridge(numeric_cols: list[str], categorical_cols: list[str]) -> BaseEstimator:
    return make_pipeline(make_preprocessor(numeric_cols, categorical_cols), Ridge(alpha=1.0))


def make_elasticnet(numeric_cols: list[str], categorical_cols: list[str]) -> BaseEstimator:
    return make_pipeline(
        make_preprocessor(numeric_cols, categorical_cols),
        ElasticNet(alpha=0.02, l1_ratio=0.15, max_iter=5000, random_state=9101),
    )


def make_xgboost(numeric_cols: list[str], categorical_cols: list[str], use_gpu: bool = False):
    try:
        from xgboost import XGBRegressor

        kwargs = {"device": "cuda"} if use_gpu else {}
        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=9101,
                n_jobs=-1,
                **kwargs,
            ),
        )
    except Exception as exc:
        print(f"[WARN] XGBoost unavailable, fallback DummyRegressor: {type(exc).__name__}: {exc}")
        return make_pipeline(make_preprocessor(numeric_cols, categorical_cols), DummyRegressor(strategy="mean"))


def make_lightgbm(numeric_cols: list[str], categorical_cols: list[str], use_gpu: bool = False):
    try:
        from lightgbm import LGBMRegressor

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            LGBMRegressor(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="regression",
                device_type="gpu" if use_gpu else "cpu",
                random_state=9101,
                n_jobs=-1,
                verbose=-1,
            ),
        )
    except Exception as exc:
        print(f"[WARN] LightGBM unavailable, fallback DummyRegressor: {type(exc).__name__}: {exc}")
        return make_pipeline(make_preprocessor(numeric_cols, categorical_cols), DummyRegressor(strategy="mean"))


def make_catboost(numeric_cols: list[str], categorical_cols: list[str], use_gpu: bool = False):
    try:
        from catboost import CatBoostRegressor

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            CatBoostRegressor(
                iterations=500,
                depth=5,
                learning_rate=0.03,
                loss_function="RMSE",
                task_type="GPU" if use_gpu else "CPU",
                random_seed=9101,
                verbose=False,
            ),
        )
    except Exception as exc:
        print(f"[WARN] CatBoost unavailable, fallback DummyRegressor: {type(exc).__name__}: {exc}")
        return make_pipeline(make_preprocessor(numeric_cols, categorical_cols), DummyRegressor(strategy="mean"))


def fit_sarimax(y: pd.Series, x: pd.DataFrame | None = None):
    if auto_arima is None:
        raise ImportError("pmdarima is required for SARIMAX/ARIMA models")
    x = None if x is None or x.shape[1] == 0 else x.fillna(0)
    return auto_arima(
        y.astype(float),
        X=x,
        seasonal=True,
        m=52,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3,
        max_q=3,
        max_P=1,
        max_Q=1,
    )


def predict_sarimax(model, x_future: pd.DataFrame | None, steps: int) -> np.ndarray:
    names = [c for c in getattr(model.arima_res_.model, "exog_names", []) if c != "const"]
    if names and x_future is not None:
        aligned = x_future.reindex(columns=names).fillna(0)
        return np.asarray(model.predict(n_periods=steps, X=aligned))
    return np.asarray(model.predict(n_periods=steps))


def build_base_registry(numeric_cols: list[str], categorical_cols: list[str], use_gpu: bool = False) -> list[ModelSpec]:
    """第一版 base model registry。要增減模型只改這裡。"""
    return [
        ModelSpec("seasonal_naive", "naive", lambda: SeasonalNaiveRegressor(52)),
        ModelSpec("ridge", "global_panel", lambda: make_ridge(numeric_cols, categorical_cols)),
        ModelSpec("elasticnet", "global_panel", lambda: make_elasticnet(numeric_cols, categorical_cols)),
        ModelSpec("xgboost", "global_panel", lambda: make_xgboost(numeric_cols, categorical_cols, use_gpu)),
        ModelSpec("lightgbm", "global_panel", lambda: make_lightgbm(numeric_cols, categorical_cols, use_gpu)),
        ModelSpec("catboost", "global_panel", lambda: make_catboost(numeric_cols, categorical_cols, use_gpu)),
        ModelSpec("sarimax", "local_series", lambda: "sarimax"),
    ]


def new_model(spec: ModelSpec):
    return spec.factory()
