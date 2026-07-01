from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ensemble.py 只負責「規則型」ensemble：不對 base model 的 OOF 預測值做任何迴歸擬合，
# 直接依固定規則計算組合權重，參數量趨近於 0，適合跟 stacking.py 的迴歸型 meta model
# 放在一起比較，尤其是在樣本數較小（例如 branch 任務 n_obs 僅約 120）時，
# 迴歸型 meta model 的權重容易被雜訊帶偏，規則型 ensemble 相對更穩健。
# 本模組不依賴 modeling/stacking.py，兩者完全獨立，由 modeling/combiner.py 統一路由。

RULE_BASED_METHODS = {"simple_average", "weighted_average", "topk_average"}

DEFAULT_ENSEMBLE_TOPK = 2


@dataclass
class EnsembleModel:
    method: str
    weights: np.ndarray
    feature_names: list[str]


def _as_frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()

    return pd.DataFrame(x)


def _clean_ensemble_x(x: pd.DataFrame) -> pd.DataFrame:
    out = _as_frame(x)

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def _clean_ensemble_y(y) -> pd.Series:
    out = pd.to_numeric(pd.Series(y), errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def base_model_wape(x_fit: pd.DataFrame, y_fit: pd.Series) -> dict[str, float]:
    """計算每個 base model 自己在這批資料上的 WAPE，供權重計算使用。"""
    y_abs_sum = float(np.abs(y_fit).sum())
    wapes: dict[str, float] = {}

    for col in x_fit.columns:
        if y_abs_sum == 0:
            wapes[col] = np.inf
            continue

        wapes[col] = float(np.abs(x_fit[col] - y_fit).sum() / y_abs_sum)

    return wapes


def _compute_weights(
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    method: str,
    topk: int,
) -> np.ndarray:
    """
    依固定規則（而非迴歸擬合）計算每個 base model 的組合權重。

    simple_average：
        所有 base model 權重相等，是最保守、最不容易 overfit 的組合方式。

    weighted_average：
        依各 base model 自己在這批資料上的 WAPE 反比加權，WAPE 越低（越準）權重越高，
        不經過任何迴歸擬合，因此沒有額外可調參數。

    topk_average：
        只取 WAPE 最低的 topk 個 base model 做等權重平均，其餘 base model 權重為 0，
        用於排除表現明顯較差的 base model 對整體 ensemble 的拖累。
    """
    cols = x_fit.columns.tolist()
    n = len(cols)

    if method == "simple_average":
        return np.repeat(1.0 / n, n)

    wapes = base_model_wape(x_fit, y_fit)

    if method == "weighted_average":
        inv = {
            c: (1.0 / max(w, 1e-6)) if np.isfinite(w) else 0.0
            for c, w in wapes.items()
        }
        total = sum(inv.values())

        if total <= 0:
            return np.repeat(1.0 / n, n)

        return np.array([inv[c] / total for c in cols])

    if method == "topk_average":
        k = max(1, min(topk, n))
        top_cols = set(sorted(wapes, key=wapes.get)[:k])
        return np.array([1.0 / k if c in top_cols else 0.0 for c in cols])

    raise ValueError(
        f"Unknown rule-based ensemble method={method}. "
        f"Allowed: {sorted(RULE_BASED_METHODS)}"
    )


def fit_ensemble(
    x,
    y,
    method: str = "simple_average",
    topk: int | None = None,
) -> EnsembleModel:
    """
    依固定規則計算 ensemble 權重（不做任何迴歸擬合）。

    topk 只在 method="topk_average" 時生效，預設為 DEFAULT_ENSEMBLE_TOPK。
    """
    method = str(method).lower().strip()

    x_df = _clean_ensemble_x(x)
    y_sr = _clean_ensemble_y(y)

    mask = y_sr.notna()

    for col in x_df.columns:
        mask &= x_df[col].notna()

    x_fit = x_df.loc[mask].copy()
    y_fit = y_sr.loc[mask].astype(float).copy()

    if x_fit.empty:
        raise RuntimeError(
            f"Cannot fit ensemble method={method}: empty training data"
        )

    weights = _compute_weights(
        x_fit, y_fit, method, topk=topk or DEFAULT_ENSEMBLE_TOPK
    )

    return EnsembleModel(
        method=method,
        weights=weights,
        feature_names=x_fit.columns.astype(str).tolist(),
    )


def predict_ensemble(
    model: EnsembleModel,
    x,
) -> np.ndarray:
    x_df = _clean_ensemble_x(x)

    for col in model.feature_names:
        if col not in x_df.columns:
            x_df[col] = np.nan

    x_df = x_df[model.feature_names].copy()

    if x_df.isna().any().any():
        x_df = x_df.fillna(x_df.median(numeric_only=True))
        x_df = x_df.fillna(0)

    weights = np.asarray(model.weights, dtype=float)
    pred = x_df.to_numpy(dtype=float) @ weights

    pred = np.asarray(pred, dtype=float)
    pred = np.clip(pred, 0, None)

    return pred


def extract_ensemble_weights(model: EnsembleModel) -> pd.DataFrame:
    weights = np.asarray(model.weights, dtype=float)

    return pd.DataFrame(
        {
            "meta_model": model.method,
            "base_model": model.feature_names,
            "weight": weights,
        }
    )