from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


MANDATORY_NUMERIC_FEATURES = {
    "year",
    "week",
    "week_sin",
    "week_cos",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_52",
    "roll4_mean",
    "roll8_mean",
}


@dataclass(frozen=True)
class FeatureSelectionResult:
    numeric_cols: list[str]
    report: pd.DataFrame


def profile_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        rows.append(
            {
                "feature": col,
                "dtype": str(df[col].dtype),
                "missing_rate": float(s.isna().mean()),
                "zero_rate": float((s.fillna(0) == 0).mean()),
                "n_unique": int(s.nunique(dropna=True)),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "std": float(s.std()) if s.notna().any() else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows)


def filter_numeric_features(
    train: pd.DataFrame,
    numeric_cols: list[str],
    missing_threshold: float = 0.90,
    corr_threshold: float = 0.98,
) -> FeatureSelectionResult:
    """
    快速 feature filtering。

    不看 y，因此適合每次測試與排程都執行。
    """
    if not numeric_cols:
        return FeatureSelectionResult([], pd.DataFrame())

    profile = profile_features(train, numeric_cols)

    keep = profile[
        (profile["missing_rate"] <= missing_threshold)
        & (profile["n_unique"] > 1)
    ]["feature"].tolist()

    mandatory = [c for c in numeric_cols if c in MANDATORY_NUMERIC_FEATURES]
    keep = list(dict.fromkeys(mandatory + keep))

    if not keep:
        report = profile.copy()
        report["selected"] = False
        report["selection_stage"] = "filter"
        return FeatureSelectionResult([], report)

    x = train[keep].copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x)

    vt = VarianceThreshold(threshold=0.0)
    x_vt = vt.fit_transform(x_imp)
    vt_cols = [c for c, ok in zip(keep, vt.get_support()) if ok]

    keep = list(dict.fromkeys(mandatory + vt_cols))

    if len(keep) > 1:
        corr = (
            pd.DataFrame(x_imp, columns=x.columns)[keep]
            .corr()
            .abs()
        )

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        drop_cols = set()

        for col in upper.columns:
            if col in mandatory:
                continue

            if any(upper[col] > corr_threshold):
                drop_cols.add(col)

        keep = [c for c in keep if c not in drop_cols]

    report = profile.copy()
    report["selected"] = report["feature"].isin(keep)
    report["selection_stage"] = "filter"
    report["importance"] = np.nan

    return FeatureSelectionResult(keep, report)


def select_lgbm_topk_features(
    train: pd.DataFrame,
    numeric_cols: list[str],
    target_col: str = "count",
    top_k: int = 60,
) -> FeatureSelectionResult:
    """
    使用小型 LightGBM importance 選 top-k numeric features。

    注意：
    這版是快速測試用 selection。
    正式嚴格評估時，應再升級為 fold-level feature selection。
    """
    filtered = filter_numeric_features(train, numeric_cols)
    candidate_cols = filtered.numeric_cols

    if not candidate_cols:
        return filtered

    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        print(
            f"[WARN] LightGBM unavailable for feature selection, "
            f"fallback to filter only: {type(exc).__name__}: {exc}"
        )
        return filtered

    x = train[candidate_cols].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(train[target_col], errors="coerce")

    valid_mask = y.notna()

    x = x.loc[valid_mask]
    y = y.loc[valid_mask]

    if x.empty:
        return filtered

    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x)

    model = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="regression",
        random_state=9101,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(x_imp, y.astype(float))

    importance_df = pd.DataFrame(
        {
            "feature": candidate_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    mandatory = [c for c in candidate_cols if c in MANDATORY_NUMERIC_FEATURES]
    top_features = importance_df.head(top_k)["feature"].tolist()

    selected = list(dict.fromkeys(mandatory + top_features))

    profile = profile_features(train, numeric_cols)
    report = profile.merge(
        importance_df,
        on="feature",
        how="left",
    )
    report["importance"] = report["importance"].fillna(0)
    report["selected"] = report["feature"].isin(selected)
    report["selection_stage"] = "lgbm_topk"

    return FeatureSelectionResult(selected, report)