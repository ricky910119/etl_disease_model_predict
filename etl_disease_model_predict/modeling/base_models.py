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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:
    from pmdarima import auto_arima
except Exception:  # pragma: no cover
    auto_arima = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    scope: str
    factory: Callable[[], object]
    enabled: bool = True


class SeasonalNaiveRegressor:
    """
    以去年同週優先，否則以最近 season_length 週均值補足。
    """

    def __init__(self, season_length: int = 52):
        self.season_length = season_length
        self.y_: list[float] = []
        self.fallback_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.y_ = pd.Series(y).astype(float).dropna().tolist()
        self.fallback_ = (
            float(np.nanmean(self.y_[-self.season_length:]))
            if self.y_
            else 0.0
        )

        if np.isnan(self.fallback_):
            self.fallback_ = 0.0

        return self

    def predict(self, x: pd.DataFrame):
        preds = []
        history = list(self.y_)

        for _ in range(len(x)):
            pred = (
                history[-self.season_length]
                if len(history) >= self.season_length
                else self.fallback_
            )
            preds.append(pred)
            history.append(pred)

        return np.asarray(preds, dtype=float)


class ConstantSeriesForecaster:
    """
    當某個 county / disease / source 的時間序列完全固定時，
    不跑 SARIMAX，直接回傳常數預測。
    """

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, n_periods: int, X=None):
        return np.repeat(self.value, n_periods)


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _as_numpy_transformer() -> FunctionTransformer:
    return FunctionTransformer(
        lambda x: np.asarray(x),
        validate=False,
        feature_names_out="one-to-one",
    )


def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    numeric = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        _one_hot_encoder(),
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_ridge(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> BaseEstimator:
    return make_pipeline(
        make_preprocessor(numeric_cols, categorical_cols),
        Ridge(alpha=1.0),
    )


def make_elasticnet(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> BaseEstimator:
    return make_pipeline(
        make_preprocessor(numeric_cols, categorical_cols),
        ElasticNet(
            alpha=0.02,
            l1_ratio=0.15,
            max_iter=5000,
            random_state=9101,
        ),
    )


def make_xgboost(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from xgboost import XGBRegressor

        kwargs = {"device": "cuda"} if use_gpu else {}

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            _as_numpy_transformer(),
            XGBRegressor(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.04,
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
        print(
            f"[WARN] XGBoost unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}"
        )
        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )


def make_lightgbm(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from lightgbm import LGBMRegressor

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            _as_numpy_transformer(),
            LGBMRegressor(
                n_estimators=450,
                learning_rate=0.04,
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
        print(
            f"[WARN] LightGBM unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}"
        )
        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )


def make_catboost(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from catboost import CatBoostRegressor

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            _as_numpy_transformer(),
            CatBoostRegressor(
                iterations=350,
                depth=5,
                learning_rate=0.04,
                loss_function="RMSE",
                task_type="GPU" if use_gpu else "CPU",
                random_seed=9101,
                verbose=False,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] CatBoost unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}"
        )
        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )


def fit_sarimax(y: pd.Series, x: pd.DataFrame | None = None):
    y = pd.Series(y).astype(float).dropna()

    if y.empty:
        return ConstantSeriesForecaster(0.0)

    if y.nunique(dropna=True) <= 1:
        return ConstantSeriesForecaster(float(y.iloc[-1]))

    if auto_arima is None:
        raise ImportError("pmdarima is required for SARIMAX/ARIMA models")

    x = None if x is None or x.shape[1] == 0 else x.fillna(0)

    return auto_arima(
        y,
        X=x,
        seasonal=True,
        m=52,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
    )


def predict_sarimax(
    model,
    x_future: pd.DataFrame | None,
    steps: int,
) -> np.ndarray:
    if isinstance(model, ConstantSeriesForecaster):
        return np.asarray(model.predict(n_periods=steps))

    names = [
        c for c in getattr(model.arima_res_.model, "exog_names", [])
        if c != "const"
    ]

    if names and x_future is not None:
        aligned = x_future.reindex(columns=names).fillna(0)
        return np.asarray(model.predict(n_periods=steps, X=aligned))

    return np.asarray(model.predict(n_periods=steps))


def build_base_registry(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
    model_names: list[str] | None = None,
    enable_sarimax: bool = False,
) -> list[ModelSpec]:
    """
    Base model registry。

    model_names 控制本次要跑哪些模型。
    SARIMAX 預設關閉，避免測試與排程耗時過長。
    """
    allowed = set(model_names or [])

    specs = [
        ModelSpec(
            "seasonal_naive",
            "naive",
            lambda: SeasonalNaiveRegressor(52),
            enabled=("seasonal_naive" in allowed),
        ),
        ModelSpec(
            "ridge",
            "global_panel",
            lambda: make_ridge(numeric_cols, categorical_cols),
            enabled=("ridge" in allowed),
        ),
        ModelSpec(
            "elasticnet",
            "global_panel",
            lambda: make_elasticnet(numeric_cols, categorical_cols),
            enabled=("elasticnet" in allowed),
        ),
        ModelSpec(
            "xgboost",
            "global_panel",
            lambda: make_xgboost(numeric_cols, categorical_cols, use_gpu),
            enabled=("xgboost" in allowed),
        ),
        ModelSpec(
            "lightgbm",
            "global_panel",
            lambda: make_lightgbm(numeric_cols, categorical_cols, use_gpu),
            enabled=("lightgbm" in allowed),
        ),
        ModelSpec(
            "catboost",
            "global_panel",
            lambda: make_catboost(numeric_cols, categorical_cols, use_gpu),
            enabled=("catboost" in allowed),
        ),
        ModelSpec(
            "sarimax",
            "local_series",
            lambda: "sarimax",
            enabled=("sarimax" in allowed and enable_sarimax),
        ),
    ]

    return specs


def new_model(spec: ModelSpec):
    return spec.factory()