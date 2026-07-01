from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# stacking.py 只負責「迴歸型」meta model：
# 用 sklearn estimator 從 base model 的 OOF 預測值學出組合權重。
# 規則型 ensemble（simple_average / weighted_average / topk_average）
# 定義在 modeling/ensemble.py，兩者互不依賴，統一由 modeling/combiner.py 路由。
REGRESSION_METHODS = {"ridge", "elasticnet", "lasso", "huber", "nonnegative_linear"}


@dataclass
class MetaModel:
    method: str
    model: BaseEstimator
    feature_names: list[str]


def _as_frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()

    return pd.DataFrame(x)


def _clean_meta_x(x: pd.DataFrame) -> pd.DataFrame:
    out = _as_frame(x)

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def _clean_meta_y(y) -> pd.Series:
    out = pd.to_numeric(pd.Series(y), errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _make_meta_estimator(method: str) -> BaseEstimator:
    if method == "ridge":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0, random_state=9101),
        )

    if method == "elasticnet":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.25,
                positive=False,
                max_iter=20000,
                random_state=9101,
            ),
        )

    if method == "lasso":
        return make_pipeline(
            StandardScaler(),
            Lasso(
                alpha=0.01,
                positive=False,
                max_iter=20000,
                random_state=9101,
            ),
        )

    if method == "huber":
        return make_pipeline(
            StandardScaler(),
            HuberRegressor(
                epsilon=1.35,
                alpha=0.0001,
                max_iter=1000,
            ),
        )

    if method == "nonnegative_linear":
        return LinearRegression(
            positive=True,
        )

    raise ValueError(
        f"Unknown regression meta model method={method}. "
        f"Allowed: {sorted(REGRESSION_METHODS)}"
    )


def fit_meta_model(
    x,
    y,
    method: str = "ridge",
) -> MetaModel:
    """用 sklearn estimator 訓練一個迴歸型 meta model。"""
    method = str(method).lower().strip()

    x_df = _clean_meta_x(x)
    y_sr = _clean_meta_y(y)

    mask = y_sr.notna()

    for col in x_df.columns:
        mask &= x_df[col].notna()

    x_fit = x_df.loc[mask].copy()
    y_fit = y_sr.loc[mask].astype(float).copy()

    if x_fit.empty:
        raise RuntimeError(
            f"Cannot fit meta model method={method}: empty training data"
        )

    estimator = _make_meta_estimator(method)
    estimator.fit(x_fit, y_fit)

    return MetaModel(
        method=method,
        model=estimator,
        feature_names=x_fit.columns.astype(str).tolist(),
    )


def predict_meta(
    meta: MetaModel,
    x,
) -> np.ndarray:
    x_df = _clean_meta_x(x)

    for col in meta.feature_names:
        if col not in x_df.columns:
            x_df[col] = np.nan

    x_df = x_df[meta.feature_names].copy()

    if x_df.isna().any().any():
        x_df = x_df.fillna(x_df.median(numeric_only=True))
        x_df = x_df.fillna(0)

    pred = meta.model.predict(x_df)

    pred = np.asarray(pred, dtype=float)
    pred = np.clip(pred, 0, None)

    return pred


def extract_meta_weights(meta: MetaModel) -> pd.DataFrame:
    model = meta.model

    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    else:
        estimator = model

    if not hasattr(estimator, "coef_"):
        return pd.DataFrame(
            columns=[
                "meta_model",
                "base_model",
                "weight",
            ]
        )

    coef = np.asarray(estimator.coef_, dtype=float).ravel()

    return pd.DataFrame(
        {
            "meta_model": meta.method,
            "base_model": meta.feature_names,
            "weight": coef,
        }
    )