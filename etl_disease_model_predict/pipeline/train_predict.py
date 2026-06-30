from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import json

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from etl_disease_model_predict.features.dataset import (
    build_feature_table,
    get_exog_columns,
)
from etl_disease_model_predict.features.selection import (
    filter_numeric_features,
    select_lgbm_topk_features,
)
from etl_disease_model_predict.modeling.base_models import (
    build_base_registry,
    fit_sarimax,
    new_model,
    predict_sarimax,
)
from etl_disease_model_predict.modeling.stacking import (
    fit_meta_model,
    predict_meta,
)
from etl_disease_model_predict.utils.week import (
    forecast_yearweeks,
    latest_closed_yearweek,
)

KEY_COLS = ["data_source", "disease", "county"]
CATEGORICAL_COLS = ["county", "disease", "data_source"]


@dataclass(frozen=True)
class RunModeConfig:
    n_splits: int
    min_train_weeks: int
    recent_weeks: int | None
    feature_set: str
    feature_select: str
    top_k: int | None
    model_names: list[str]


RUN_MODE_CONFIG = {
    "smoke": RunModeConfig(
        n_splits=1,
        min_train_weeks=52,
        recent_weeks=156,
        feature_set="base",
        feature_select="filter",
        top_k=None,
        model_names=["seasonal_naive", "ridge", "lightgbm"],
    ),
    "fast": RunModeConfig(
        n_splits=2,
        min_train_weeks=104,
        recent_weeks=260,
        feature_set="medium",
        feature_select="lgbm_topk",
        top_k=60,
        model_names=["seasonal_naive", "ridge", "lightgbm"],
    ),
    "full": RunModeConfig(
        n_splits=5,
        min_train_weeks=156,
        recent_weeks=None,
        feature_set="full",
        feature_select="lgbm_topk",
        top_k=80,
        model_names=[
            "seasonal_naive",
            "ridge",
            "elasticnet",
            "xgboost",
            "lightgbm",
            "catboost",
            "keras_mlp",
        ],
    ),
    "forecast": RunModeConfig(
        n_splits=1,
        min_train_weeks=104,
        recent_weeks=260,
        feature_set="medium",
        feature_select="lgbm_topk",
        top_k=60,
        model_names=["seasonal_naive", "ridge", "lightgbm"],
    ),
}


FEATURE_SET_CONFIG = {
    "base": {
        "lags": [1, 2, 4, 8],
        "rolling_windows": [4, 8],
        "diff_lags": [],
        "growth_lags": [],
    },
    "medium": {
        "lags": [1, 2, 3, 4, 8, 12, 26, 52],
        "rolling_windows": [4, 8, 12, 26],
        "diff_lags": [1, 4],
        "growth_lags": [1, 4],
    },
    "full": {
        "lags": [1, 2, 3, 4, 8, 12, 26, 52],
        "rolling_windows": [4, 8, 12, 26, 52],
        "diff_lags": [1, 2, 4, 8],
        "growth_lags": [1, 2, 4, 8],
    },
}


def _resolve_config(
    run_mode: str,
    recent_weeks: int | None,
    feature_set: str | None,
    feature_select: str | None,
    top_k: int | None,
) -> RunModeConfig:
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown run_mode={run_mode}")

    base = RUN_MODE_CONFIG[run_mode]

    return RunModeConfig(
        n_splits=base.n_splits,
        min_train_weeks=base.min_train_weeks,
        recent_weeks=base.recent_weeks if recent_weeks is None else recent_weeks,
        feature_set=base.feature_set if feature_set is None else feature_set,
        feature_select=base.feature_select if feature_select is None else feature_select,
        top_k=base.top_k if top_k is None else top_k,
        model_names=base.model_names,
    )


def add_lag_rolling_features(
    df: pd.DataFrame,
    feature_set: str,
) -> pd.DataFrame:
    if feature_set not in FEATURE_SET_CONFIG:
        raise ValueError(f"Unknown feature_set={feature_set}")

    cfg = FEATURE_SET_CONFIG[feature_set]

    out = df.sort_values(KEY_COLS + ["yearweek"]).copy()

    if "week" in out.columns:
        out["week_sin"] = np.sin(2 * np.pi * out["week"].astype(float) / 52)
        out["week_cos"] = np.cos(2 * np.pi * out["week"].astype(float) / 52)

    grp = out.groupby(KEY_COLS, dropna=False)["count"]

    for lag in cfg["lags"]:
        out[f"lag_{lag}"] = grp.shift(lag)

    for window in cfg["rolling_windows"]:
        out[f"roll{window}_mean"] = grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        out[f"roll{window}_std"] = grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )

    for lag in cfg["diff_lags"]:
        lag_col = f"lag_{lag}"
        if lag_col in out.columns:
            out[f"diff_{lag}"] = out["count"] - out[lag_col]

    for lag in cfg["growth_lags"]:
        lag_col = f"lag_{lag}"
        if lag_col in out.columns:
            denom = out[lag_col].replace(0, np.nan)
            out[f"growth_{lag}"] = (out["count"] - out[lag_col]) / denom

    return out


def _feature_columns(
    df: pd.DataFrame,
    exog_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    engineered_cols = [
        c for c in df.columns
        if c.startswith("lag_")
        or c.startswith("roll")
        or c.startswith("diff_")
        or c.startswith("growth_")
        or c in {"week_sin", "week_cos"}
    ]

    numeric_cols = list(
        dict.fromkeys(engineered_cols + [c for c in exog_cols if c in df.columns])
    )

    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    feature_cols = categorical_cols + numeric_cols

    return feature_cols, numeric_cols, categorical_cols


def _apply_recent_weeks(
    df: pd.DataFrame,
    recent_weeks: int | None,
) -> pd.DataFrame:
    if recent_weeks is None:
        return df

    weeks = sorted(df["yearweek"].dropna().astype(int).unique())

    if len(weeks) <= recent_weeks:
        return df

    keep_weeks = set(weeks[-recent_weeks:])

    return df[df["yearweek"].isin(keep_weeks)].copy()


def _rolling_splits(
    df: pd.DataFrame,
    n_splits: int,
    min_train_weeks: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    weeks = np.array(sorted(df["yearweek"].unique()))

    if len(weeks) < 12:
        raise RuntimeError(
            f"Not enough training weeks for rolling validation: {len(weeks)}"
        )

    if n_splits <= 1:
        cut = int(len(weeks) * 0.8)

        if cut <= 0 or cut >= len(weeks):
            raise RuntimeError("Cannot create single fallback train/validation split")

        return [
            (
                df.index[df["yearweek"].isin(set(weeks[:cut]))].to_numpy(),
                df.index[df["yearweek"].isin(set(weeks[cut:]))].to_numpy(),
            )
        ]

    if len(weeks) < min_train_weeks + n_splits:
        n_splits = max(2, min(3, len(weeks) // 26))
        min_train_weeks = max(52, len(weeks) - n_splits * 8)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []

    for tr_w_idx, va_w_idx in tscv.split(weeks):
        if len(tr_w_idx) < min_train_weeks:
            continue

        tr_weeks = set(weeks[tr_w_idx])
        va_weeks = set(weeks[va_w_idx])

        tr_idx = df.index[df["yearweek"].isin(tr_weeks)].to_numpy()
        va_idx = df.index[df["yearweek"].isin(va_weeks)].to_numpy()

        if len(tr_idx) and len(va_idx):
            splits.append((tr_idx, va_idx))

    if not splits:
        cut = int(len(weeks) * 0.8)

        if cut <= 0 or cut >= len(weeks):
            raise RuntimeError("Cannot create fallback train/validation split")

        splits = [
            (
                df.index[df["yearweek"].isin(set(weeks[:cut]))].to_numpy(),
                df.index[df["yearweek"].isin(set(weeks[cut:]))].to_numpy(),
            )
        ]

    return splits


def _recursive_future_features(
    history: pd.DataFrame,
    future_exog: pd.DataFrame,
    forecast_period: int,
    feature_set: str,
) -> pd.DataFrame:
    cfg = FEATURE_SET_CONFIG[feature_set]

    hist = history.sort_values("yearweek").copy()
    future = future_exog.sort_values("yearweek").head(forecast_period).copy()

    count_history = hist["count"].astype(float).dropna().tolist()
    rows = []

    for _, row in future.iterrows():
        item = row.to_dict()

        if "week" in item and pd.notna(item["week"]):
            item["week_sin"] = np.sin(2 * np.pi * float(item["week"]) / 52)
            item["week_cos"] = np.cos(2 * np.pi * float(item["week"]) / 52)

        for lag in cfg["lags"]:
            item[f"lag_{lag}"] = (
                count_history[-lag]
                if len(count_history) >= lag
                else np.nan
            )

        for window in cfg["rolling_windows"]:
            vals = count_history[-window:]

            item[f"roll{window}_mean"] = (
                float(np.mean(vals))
                if vals
                else np.nan
            )
            item[f"roll{window}_std"] = (
                float(np.std(vals, ddof=1))
                if len(vals) > 1
                else 0.0
            )

        current_proxy = count_history[-1] if count_history else 0.0

        for lag in cfg["diff_lags"]:
            lag_value = item.get(f"lag_{lag}", np.nan)
            item[f"diff_{lag}"] = (
                current_proxy - lag_value
                if pd.notna(lag_value)
                else np.nan
            )

        for lag in cfg["growth_lags"]:
            lag_value = item.get(f"lag_{lag}", np.nan)

            if pd.notna(lag_value) and lag_value != 0:
                item[f"growth_{lag}"] = (current_proxy - lag_value) / lag_value
            else:
                item[f"growth_{lag}"] = np.nan

        rows.append(item)

        count_history.append(current_proxy)

    return pd.DataFrame(rows)


def _select_features(
    train: pd.DataFrame,
    numeric_cols: list[str],
    feature_select: str,
    top_k: int | None,
) -> tuple[list[str], pd.DataFrame]:
    if feature_select == "none":
        report = pd.DataFrame(
            {
                "feature": numeric_cols,
                "selected": True,
                "selection_stage": "none",
                "importance": np.nan,
            }
        )
        return numeric_cols, report

    if feature_select == "filter":
        result = filter_numeric_features(train, numeric_cols)
        return result.numeric_cols, result.report

    if feature_select == "lgbm_topk":
        result = select_lgbm_topk_features(
            train=train,
            numeric_cols=numeric_cols,
            target_col="count",
            top_k=top_k or 60,
        )
        return result.numeric_cols, result.report

    raise ValueError(f"Unknown feature_select={feature_select}")

def _apply_covid_policy(
    df: pd.DataFrame,
    covid_policy: str,
) -> pd.DataFrame:
    """
    COVID 年份處理策略。

    include: 保留 2020-2022
    exclude: 排除歷史訓練資料中的 2020-2022，未來預測列不刪
    flag: 保留資料，新增 covid_period=1/0
    """
    if covid_policy not in {"include", "exclude", "flag"}:
        raise ValueError(
            "covid_policy must be one of: include, exclude, flag"
        )

    out = df.copy()
    year = (out["yearweek"].astype(int) // 100).astype(int)
    covid_mask = year.between(2020, 2022)

    if covid_policy == "flag":
        out["covid_period"] = covid_mask.astype(int)
        return out

    if covid_policy == "exclude":
        if "is_future" in out.columns:
            out = out.loc[(~covid_mask) | (out["is_future"] == True)].copy()
        else:
            out = out.loc[~covid_mask].copy()

    return out


def _wape_score_func(y_true, y_pred) -> float:
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce")
    yp = pd.to_numeric(pd.Series(y_pred), errors="coerce")

    mask = (
        yt.notna()
        & yp.notna()
        & np.isfinite(yt)
        & np.isfinite(yp)
    )

    yt = yt.loc[mask].astype(float)
    yp = yp.loc[mask].astype(float)

    denom = float(np.abs(yt).sum())

    if denom == 0:
        return 0.0

    return float(np.abs(yp - yt).sum() / denom)


WAPE_SCORER = make_scorer(
    _wape_score_func,
    greater_is_better=False,
)


def _yearweek_cv_splits(
    df: pd.DataFrame,
    n_splits: int,
    min_train_weeks: int = 52,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    給 GridSearchCV 使用的 inner CV。

    使用 yearweek 做 time split，避免一般 KFold 造成時間洩漏。
    回傳的是 row position index，不是 DataFrame index。
    """
    if "yearweek" not in df.columns:
        raise KeyError("yearweek is required for grid search CV")

    work = df.reset_index(drop=True).copy()
    weeks = np.array(sorted(work["yearweek"].astype(int).unique()))

    if len(weeks) < n_splits + 2:
        return []

    actual_splits = min(n_splits, max(2, len(weeks) // 26))

    tscv = TimeSeriesSplit(n_splits=actual_splits)
    splits = []

    for tr_w_idx, va_w_idx in tscv.split(weeks):
        if len(tr_w_idx) < min_train_weeks:
            continue

        tr_weeks = set(weeks[tr_w_idx])
        va_weeks = set(weeks[va_w_idx])

        tr_idx = work.index[work["yearweek"].isin(tr_weeks)].to_numpy()
        va_idx = work.index[work["yearweek"].isin(va_weeks)].to_numpy()

        if len(tr_idx) and len(va_idx):
            splits.append((tr_idx, va_idx))

    return splits


def _param_grid_for_model(model_name: str) -> dict:
    """
    小型 GridSearch 參數表。

    這裡刻意不要做太大的 grid，否則 full mode 會跑非常久。
    """
    grids = {
        "ridge": {
            "ridge__alpha": [0.1, 1.0, 10.0, 50.0],
        },
        "elasticnet": {
            "elasticnet__alpha": [0.005, 0.02, 0.08],
            "elasticnet__l1_ratio": [0.1, 0.3, 0.6],
        },
        "lightgbm": {
            "lgbmregressor__n_estimators": [300, 500],
            "lgbmregressor__learning_rate": [0.03, 0.05],
            "lgbmregressor__num_leaves": [15, 31],
            "lgbmregressor__min_child_samples": [20, 50],
        },
        "xgboost": {
            "xgbregressor__n_estimators": [300, 500],
            "xgbregressor__learning_rate": [0.03, 0.05],
            "xgbregressor__max_depth": [3, 5],
            "xgbregressor__subsample": [0.8, 1.0],
        },
        "catboost": {
            "catboostregressor__iterations": [300, 500],
            "catboostregressor__depth": [4, 6],
            "catboostregressor__learning_rate": [0.03, 0.05],
        },
         "keras_mlp": {
            "kerasmlpregressor__hidden_dims": [
                (128, 64),
                (256, 128),
            ],
            "kerasmlpregressor__dropout": [0.05, 0.15],
            "kerasmlpregressor__lr": [0.001, 0.0005],
            "kerasmlpregressor__batch_size": [1024],
            "kerasmlpregressor__max_epochs": [60],
            "kerasmlpregressor__patience": [8],
        },
    }

    return grids.get(model_name, {})


def _filter_valid_param_grid(model, param_grid: dict) -> dict:
    valid_params = set(model.get_params().keys())

    return {
        key: value
        for key, value in param_grid.items()
        if key in valid_params
    }


def _fit_model_with_optional_grid_search(
    model,
    model_name: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    y_col: str,
    enable_grid_search: bool,
    grid_cv_splits: int,
):
    """
    Global panel model 的 fit helper。

    enable_grid_search=False 時，直接 fit。
    enable_grid_search=True 時，對支援的模型跑 GridSearchCV。
    """
    x = train_df[feature_cols]
    y = train_df[y_col].astype(float)

    if not enable_grid_search:
        model.fit(x, y)
        return model

    param_grid = _param_grid_for_model(model_name)
    param_grid = _filter_valid_param_grid(model, param_grid)

    if not param_grid:
        model.fit(x, y)
        return model

    cv = _yearweek_cv_splits(
        train_df,
        n_splits=grid_cv_splits,
        min_train_weeks=52,
    )

    if not cv:
        model.fit(x, y)
        return model

    print(
        f"[GRID] model={model_name}, "
        f"candidates={np.prod([len(v) for v in param_grid.values()])}, "
        f"cv_splits={len(cv)}"
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=WAPE_SCORER,
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
        error_score=np.nan,
    )

    search.fit(x, y)

    print(
        f"[GRID] model={model_name}, "
        f"best_score={search.best_score_}, "
        f"best_params={json.dumps(search.best_params_, ensure_ascii=False)}"
    )

    return search.best_estimator_

def _metric_row(
    y_true,
    y_pred,
    context: dict,
) -> dict:
    """
    建立單一 metric row。

    這裡直接計算 MAE / RMSE / MAPE / sMAPE / WAPE / Bias，
    不依賴 regression_metrics()，避免欄位名稱不一致造成 output 缺欄位。
    """
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce")
    yp = pd.to_numeric(pd.Series(y_pred), errors="coerce")

    mask = (
        yt.notna()
        & yp.notna()
        & np.isfinite(yt)
        & np.isfinite(yp)
    )

    yt = yt.loc[mask].astype(float)
    yp = yp.loc[mask].astype(float)

    row = context.copy()
    row["n_obs"] = int(len(yt))

    if len(yt) == 0:
        row.update(
            {
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "sMAPE": np.nan,
                "WAPE": np.nan,
                "Bias": np.nan,
                "y_true_sum": np.nan,
                "y_pred_sum": np.nan,
                "y_true_mean": np.nan,
                "y_pred_mean": np.nan,
            }
        )
        return row

    error = yp - yt
    abs_error = np.abs(error)

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean(error ** 2)))

    nonzero_mask = yt != 0

    if nonzero_mask.any():
        mape = float((abs_error.loc[nonzero_mask] / yt.loc[nonzero_mask].abs()).mean())
    else:
        mape = np.nan

    smape_denominator = yt.abs() + yp.abs()
    smape_mask = smape_denominator != 0

    if smape_mask.any():
        smape = float(
            (
                2 * abs_error.loc[smape_mask]
                / smape_denominator.loc[smape_mask]
            ).mean()
        )
    else:
        smape = np.nan

    y_true_sum = float(yt.sum())
    y_pred_sum = float(yp.sum())

    if y_true_sum != 0:
        wape = float(abs_error.sum() / abs(y_true_sum))
        bias = float((yp.sum() - yt.sum()) / y_true_sum)
    else:
        wape = np.nan
        bias = np.nan

    row.update(
        {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "sMAPE": smape,
            "WAPE": wape,
            "Bias": bias,
            "y_true_sum": y_true_sum,
            "y_pred_sum": y_pred_sum,
            "y_true_mean": float(yt.mean()),
            "y_pred_mean": float(yp.mean()),
        }
    )

    return row


def _build_eval_metric_df(
    train: pd.DataFrame,
    oof: pd.DataFrame,
    stacked_oof: np.ndarray,
    valid_mask: pd.Series,
    data_source: str,
    run_mode: str,
    feature_set: str,
    feature_select: str,
    top_k: int | None,
    covid_policy: str,
    enable_grid_search: bool,
    meta_model: str,
    now: str,
) -> pd.DataFrame:
    """
    建立模型測試用 metric_df。

    不輸出 OOF 明細，只輸出聚合後的模型效能。
    """
    rows = []

    base_context = {
        "data_source": data_source,
        "run_mode": run_mode,
        "feature_set": feature_set,
        "feature_select": feature_select,
        "top_k": top_k,
        "covid_policy": covid_policy,
        "enable_grid_search": enable_grid_search,
        "prediction_type": "oof",
        "created_at": now,
    }

    eval_df = train.loc[valid_mask, KEY_COLS + ["yearweek", "count"]].copy()
    eval_df = eval_df.rename(columns={"count": "y_true"})

    # -----------------------------
    # base model metrics
    # -----------------------------
    for model_name in oof.columns:
        tmp = eval_df.copy()
        tmp["y_pred"] = oof.loc[valid_mask, model_name].values
        tmp = tmp.loc[tmp["y_pred"].notna()].copy()

        if tmp.empty:
            continue

        model_context = base_context.copy()
        model_context["model_layer"] = "base"
        model_context["model_name"] = model_name

        # overall
        context = model_context.copy()
        context["metric_level"] = "overall"
        rows.append(
            _metric_row(
                tmp["y_true"],
                tmp["y_pred"],
                context,
            )
        )

        # by disease
        for disease, g in tmp.groupby("disease", dropna=False):
            context = model_context.copy()
            context["metric_level"] = "by_disease"
            context["disease"] = disease
            rows.append(_metric_row(g["y_true"], g["y_pred"], context))

        # by county
        for county, g in tmp.groupby("county", dropna=False):
            context = model_context.copy()
            context["metric_level"] = "by_county"
            context["county"] = county
            rows.append(_metric_row(g["y_true"], g["y_pred"], context))

        # by disease + county
        for (disease, county), g in tmp.groupby(
            ["disease", "county"],
            dropna=False,
        ):
            context = model_context.copy()
            context["metric_level"] = "by_disease_county"
            context["disease"] = disease
            context["county"] = county
            rows.append(_metric_row(g["y_true"], g["y_pred"], context))

    # -----------------------------
    # stacking model metrics
    # -----------------------------
    stack_df = eval_df.copy()
    stack_df["y_pred"] = stacked_oof

    stack_context = base_context.copy()
    stack_context["model_layer"] = "stacking"
    stack_context["model_name"] = f"stacking_{meta_model}"

    context = stack_context.copy()
    context["metric_level"] = "overall"
    rows.append(_metric_row(stack_df["y_true"], stack_df["y_pred"], context))

    for disease, g in stack_df.groupby("disease", dropna=False):
        context = stack_context.copy()
        context["metric_level"] = "by_disease"
        context["disease"] = disease
        rows.append(_metric_row(g["y_true"], g["y_pred"], context))

    for county, g in stack_df.groupby("county", dropna=False):
        context = stack_context.copy()
        context["metric_level"] = "by_county"
        context["county"] = county
        rows.append(_metric_row(g["y_true"], g["y_pred"], context))

    for (disease, county), g in stack_df.groupby(
        ["disease", "county"],
        dropna=False,
    ):
        context = stack_context.copy()
        context["metric_level"] = "by_disease_county"
        context["disease"] = disease
        context["county"] = county
        rows.append(_metric_row(g["y_true"], g["y_pred"], context))

    return pd.DataFrame(rows)

def _fit_predict_global_oof(
    train: pd.DataFrame,
    feature_cols: list[str],
    y_col: str,
    registry,
    n_splits: int,
    min_train_weeks: int,
    enable_grid_search: bool,
    grid_cv_splits: int,
) -> pd.DataFrame:
    oof = pd.DataFrame(index=train.index)
    splits = _rolling_splits(train, n_splits=n_splits, min_train_weeks=min_train_weeks)

    for spec in registry:
        if spec.scope != "global_panel" or not spec.enabled:
            continue

        oof[spec.name] = np.nan

        for tr_idx, va_idx in splits:
            model = new_model(spec)

            model = _fit_model_with_optional_grid_search(
                model=model,
                model_name=spec.name,
                train_df=train.loc[tr_idx].copy(),
                feature_cols=feature_cols,
                y_col=y_col,
                enable_grid_search=enable_grid_search,
                grid_cv_splits=grid_cv_splits,
            )

            oof.loc[va_idx, spec.name] = model.predict(
                train.loc[va_idx, feature_cols]
            )

    return oof


def _fit_predict_local_oof(
    train: pd.DataFrame,
    exog_cols: list[str],
    registry,
    n_splits: int,
    min_train_weeks: int,
) -> pd.DataFrame:
    oof = pd.DataFrame(index=train.index)
    splits = _rolling_splits(train, n_splits=n_splits, min_train_weeks=min_train_weeks)

    for spec in registry:
        if spec.scope == "naive" and spec.enabled:
            oof[spec.name] = np.nan

            for _, g in train.groupby(KEY_COLS, dropna=False):
                g = g.sort_values("yearweek")

                for tr_idx, va_idx in splits:
                    tr = g.loc[g.index.intersection(tr_idx)]
                    va = g.loc[g.index.intersection(va_idx)]

                    if len(tr) == 0 or len(va) == 0:
                        continue

                    model = new_model(spec)
                    model.fit(pd.DataFrame(index=tr.index), tr["count"])
                    oof.loc[va.index, spec.name] = model.predict(
                        pd.DataFrame(index=va.index)
                    )

        if spec.name == "sarimax" and spec.enabled:
            oof[spec.name] = np.nan

            for _, g in train.groupby(KEY_COLS, dropna=False):
                g = g.sort_values("yearweek")

                for tr_idx, va_idx in splits:
                    tr = g.loc[g.index.intersection(tr_idx)]
                    va = g.loc[g.index.intersection(va_idx)]

                    if len(tr) < 80 or len(va) == 0:
                        continue

                    try:
                        model = fit_sarimax(tr["count"], tr[exog_cols].fillna(0))
                        oof.loc[va.index, spec.name] = predict_sarimax(
                            model,
                            va[exog_cols].fillna(0),
                            len(va),
                        )

                    except Exception as exc:
                        print(
                            f"[WARN] sarimax OOF failed "
                            f"keys={tuple(g[KEY_COLS].iloc[0])}: "
                            f"{type(exc).__name__}: {exc}"
                        )

    return oof


def _fit_final_base_predictions(
    train: pd.DataFrame,
    future: pd.DataFrame,
    feature_cols: list[str],
    exog_cols: list[str],
    registry,
    forecast_period: int,
    enable_grid_search: bool,
    grid_cv_splits: int,
) -> pd.DataFrame:
    future_pred = pd.DataFrame(index=future.index)

    for spec in registry:
        if spec.scope == "global_panel" and spec.enabled:
            model = new_model(spec)

            model = _fit_model_with_optional_grid_search(
                model=model,
                model_name=spec.name,
                train_df=train.copy(),
                feature_cols=feature_cols,
                y_col="count",
                enable_grid_search=enable_grid_search,
                grid_cv_splits=grid_cv_splits,
            )

            future_pred[spec.name] = model.predict(future[feature_cols])

    for spec in registry:
        if spec.scope == "naive" and spec.enabled:
            future_pred[spec.name] = np.nan

            for keys, g in train.groupby(KEY_COLS, dropna=False):
                mask = (
                    (future["data_source"] == keys[0])
                    & (future["disease"] == keys[1])
                    & (future["county"] == keys[2])
                )

                fg = (
                    future.loc[mask]
                    .sort_values("yearweek")
                    .head(forecast_period)
                )

                if fg.empty:
                    continue

                model = new_model(spec)
                model.fit(
                    pd.DataFrame(index=g.index),
                    g.sort_values("yearweek")["count"],
                )
                future_pred.loc[fg.index, spec.name] = model.predict(
                    pd.DataFrame(index=fg.index)
                )

        if spec.name == "sarimax" and spec.enabled:
            future_pred[spec.name] = np.nan

            for keys, g in train.groupby(KEY_COLS, dropna=False):
                mask = (
                    (future["data_source"] == keys[0])
                    & (future["disease"] == keys[1])
                    & (future["county"] == keys[2])
                )

                fg = (
                    future.loc[mask]
                    .sort_values("yearweek")
                    .head(forecast_period)
                )

                g = g.sort_values("yearweek")

                if len(g) < 80 or fg.empty:
                    continue

                try:
                    model = fit_sarimax(g["count"], g[exog_cols].fillna(0))
                    future_pred.loc[fg.index, spec.name] = predict_sarimax(
                        model,
                        fg[exog_cols].fillna(0),
                        len(fg),
                    )

                except Exception as exc:
                    print(
                        f"[WARN] sarimax final failed keys={keys}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    return future_pred


def run_source(
    data_source: str,
    forecast_period: int,
    start_week: int | None,
    recent_weeks: int | None,
    run_mode: str,
    feature_set: str | None,
    feature_select: str | None,
    top_k: int | None,
    covid_policy: str,
    enable_grid_search: bool,
    grid_cv_splits: int,
    use_gpu: bool = False,
    enable_sarimax: bool = False,
    meta_model: str = "ridge",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = _resolve_config(
        run_mode=run_mode,
        recent_weeks=recent_weeks,
        feature_set=feature_set,
        feature_select=feature_select,
        top_k=top_k,
    )

    start_week_label = "ALL" if start_week is None else start_week

    print(
        f"[RUN] source={data_source}, "
        f"run_mode={run_mode}, "
        f"forecast_period={forecast_period}, "
        f"start_week={start_week_label}, "
        f"recent_weeks={cfg.recent_weeks}, "
        f"feature_set={cfg.feature_set}, "
        f"feature_select={cfg.feature_select}, "
        f"top_k={cfg.top_k}, "
        f"covid_policy={covid_policy}, "
        f"enable_grid_search={enable_grid_search}, "
        f"models={cfg.model_names}, "
        f"enable_sarimax={enable_sarimax}"
    )

    feature = build_feature_table(
        data_source=data_source,
        start_week=start_week,
        forecast_period=forecast_period,
    )
    feature = _apply_covid_policy(
        df=feature,
        covid_policy=covid_policy,
    )
    exog_cols = get_exog_columns(feature)
    forecast_weeks = forecast_yearweeks(forecast_period)
    train_cut_week = latest_closed_yearweek()

    base_train = feature[
        (feature["is_future"] == False)
        & feature["count"].notna()
        & (feature["yearweek"] <= train_cut_week)
    ].copy()

    base_train = _apply_recent_weeks(base_train, cfg.recent_weeks)

    if base_train.empty:
        raise RuntimeError(
            f"No training data available for source={data_source} "
            f"before yearweek={train_cut_week}"
        )

    future_raw = feature[
        (feature["is_future"] == True)
        & feature["yearweek"].isin(forecast_weeks)
    ].copy()

    if future_raw.empty:
        raise RuntimeError(
            f"No future rows available for source={data_source}; "
            f"check dim_weekdate and forecast_period"
        )

    train = (
        add_lag_rolling_features(base_train, feature_set=cfg.feature_set)
        .dropna(subset=["lag_1", "count"])
        .reset_index(drop=True)
    )

    if train.empty:
        raise RuntimeError(
            f"Training data became empty after lag feature generation "
            f"for source={data_source}"
        )

    future_parts = []

    for keys, hist in base_train.groupby(KEY_COLS, dropna=False):
        mask = (
            (future_raw["data_source"] == keys[0])
            & (future_raw["disease"] == keys[1])
            & (future_raw["county"] == keys[2])
        )

        fg = (
            future_raw.loc[mask]
            .sort_values("yearweek")
            .head(forecast_period)
        )

        if len(fg) == forecast_period:
            future_parts.append(
                _recursive_future_features(
                    history=hist,
                    future_exog=fg,
                    forecast_period=forecast_period,
                    feature_set=cfg.feature_set,
                )
            )
        else:
            print(
                f"[WARN] incomplete future rows keys={keys}, "
                f"rows={len(fg)}, expected={forecast_period}"
            )

    if not future_parts:
        raise RuntimeError(f"No complete future rows available for source={data_source}")

    future = pd.concat(future_parts, ignore_index=True)

    _, numeric_cols_all, categorical_cols = _feature_columns(train, exog_cols)

    selected_numeric_cols, selected_report = _select_features(
        train=train,
        numeric_cols=numeric_cols_all,
        feature_select=cfg.feature_select,
        top_k=cfg.top_k,
    )

    feature_cols = categorical_cols + selected_numeric_cols

    if len(feature_cols) == 0:
        raise RuntimeError(f"No feature columns selected for source={data_source}")

    registry = build_base_registry(
        numeric_cols=selected_numeric_cols,
        categorical_cols=categorical_cols,
        use_gpu=use_gpu,
        model_names=cfg.model_names + (["sarimax"] if enable_sarimax else []),
        enable_sarimax=enable_sarimax,
    )

    print(
        f"[RUN] source={data_source} "
        f"train_rows={len(train)}, "
        f"future_rows={len(future)}, "
        f"numeric_features_all={len(numeric_cols_all)}, "
        f"numeric_features_selected={len(selected_numeric_cols)}, "
        f"categorical_features={len(categorical_cols)}"
    )

    global_oof = _fit_predict_global_oof(
        train=train,
        feature_cols=feature_cols,
        y_col="count",
        registry=registry,
        n_splits=cfg.n_splits,
        min_train_weeks=cfg.min_train_weeks,
        enable_grid_search=enable_grid_search,
        grid_cv_splits=grid_cv_splits,
    )

    local_oof = _fit_predict_local_oof(
        train=train,
        exog_cols=selected_numeric_cols,
        registry=registry,
        n_splits=cfg.n_splits,
        min_train_weeks=cfg.min_train_weeks,
    )

    oof = pd.concat([global_oof, local_oof], axis=1)

    # 移除整欄都沒有預測的模型
    oof = oof.dropna(axis=1, how="all")

    # stacking 不吃有缺值的列，避免 meta model 因 NaN 出錯
    valid_mask = oof.notna().all(axis=1)

    if valid_mask.sum() == 0:
        raise RuntimeError(f"OOF prediction is empty for source={data_source}")

    if oof.shape[1] < 2:
        raise RuntimeError(
            f"Need at least 2 base model predictions for stacking, "
            f"got {oof.shape[1]} for source={data_source}"
        )

    meta = fit_meta_model(
        oof.loc[valid_mask],
        train.loc[valid_mask, "count"],
        method=meta_model,
    )

    stacked_oof = predict_meta(meta, oof.loc[valid_mask])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metric_df = _build_eval_metric_df(
        train=train,
        oof=oof,
        stacked_oof=stacked_oof,
        valid_mask=valid_mask,
        data_source=data_source,
        run_mode=run_mode,
        feature_set=cfg.feature_set,
        feature_select=cfg.feature_select,
        top_k=cfg.top_k,
        covid_policy=covid_policy,
        enable_grid_search=enable_grid_search,
        meta_model=meta_model,
        now=now,
    )
    future_base = _fit_final_base_predictions(
        train=train,
        future=future,
        feature_cols=feature_cols,
        exog_cols=selected_numeric_cols,
        registry=registry,
        forecast_period=forecast_period,
        enable_grid_search=enable_grid_search,
        grid_cv_splits=grid_cv_splits,
    )

    future_base = future_base.reindex(columns=oof.columns)

    if future_base.isna().any().any():
        future_base = future_base.fillna(future_base.median(numeric_only=True))
        future_base = future_base.fillna(0)

    final_pred = predict_meta(meta, future_base)


    id_cols = KEY_COLS + ["yearweek"]

    forecast_df = future[id_cols].copy()
    forecast_df["forecast_count"] = final_pred
    forecast_df["model_name"] = f"stacking_{meta_model}"
    forecast_df["run_mode"] = run_mode
    forecast_df["feature_set"] = cfg.feature_set
    forecast_df["feature_select"] = cfg.feature_select
    forecast_df["top_k"] = cfg.top_k
    forecast_df["created_at"] = now

    base_rows = []

    for model_name in future_base.columns:
        tmp = future[id_cols].copy()
        tmp["base_model"] = model_name
        tmp["forecast_count"] = (
            np.clip(np.rint(future_base[model_name].fillna(0)), 0, None)
            .astype(int)
        )
        tmp["run_mode"] = run_mode
        tmp["feature_set"] = cfg.feature_set
        tmp["created_at"] = now
        base_rows.append(tmp)

    base_df = (
        pd.concat(base_rows, ignore_index=True)
        if base_rows
        else pd.DataFrame()
    )

    selected_report = selected_report.copy()

    if "feature_type" not in selected_report.columns:
        selected_report["feature_type"] = "numeric"

    selected_report["data_source"] = data_source
    selected_report["run_mode"] = run_mode
    selected_report["feature_set"] = cfg.feature_set
    selected_report["feature_select"] = cfg.feature_select
    selected_report["top_k"] = cfg.top_k
    selected_report["covid_policy"] = covid_policy
    selected_report["enable_grid_search"] = enable_grid_search
    selected_report["created_at"] = now
    selected_report["covid_policy"] = covid_policy
    selected_report["enable_grid_search"] = enable_grid_search
    categorical_report = pd.DataFrame(
        [
            {
                "feature": col,
                "feature_type": "categorical",
                "selected": True,
                "selection_stage": "mandatory_categorical",
                "importance": np.nan,
                "missing_rate": np.nan,
                "zero_rate": np.nan,
                "n_unique": train[col].nunique(dropna=True)
                if col in train.columns
                else np.nan,
                "dtype": str(train[col].dtype)
                if col in train.columns
                else "category",
                "data_source": data_source,
                "run_mode": run_mode,
                "feature_set": cfg.feature_set,
                "feature_select": cfg.feature_select,
                "top_k": cfg.top_k,
                "covid_policy": covid_policy,
                "enable_grid_search": enable_grid_search,
                "created_at": now,
            }
            for col in categorical_cols
        ]
    )

    selected_report = pd.concat(
        [selected_report, categorical_report],
        ignore_index=True,
    )

    return forecast_df, base_df, metric_df, selected_report


def run_all(
    data_sources: list[str],
    forecast_period: int,
    start_week: int | None,
    recent_weeks: int | None,
    run_mode: str,
    feature_set: str | None,
    feature_select: str | None,
    top_k: int | None,
    covid_policy: str,
    enable_grid_search: bool,
    grid_cv_splits: int,
    use_gpu: bool = False,
    enable_sarimax: bool = False,
    meta_model: str = "ridge",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecasts = []
    bases = []
    metrics = []
    selected_reports = []

    for source in data_sources:
        f, b, m, s = run_source(
            data_source=source,
            forecast_period=forecast_period,
            start_week=start_week,
            recent_weeks=recent_weeks,
            run_mode=run_mode,
            feature_set=feature_set,
            feature_select=feature_select,
            top_k=top_k,
            use_gpu=use_gpu,
            enable_sarimax=enable_sarimax,
            meta_model=meta_model,
            covid_policy=covid_policy,
            enable_grid_search=enable_grid_search,
            grid_cv_splits=grid_cv_splits,
        )

        forecasts.append(f)
        bases.append(b)
        metrics.append(m)
        selected_reports.append(s)

    return (
        pd.concat(forecasts, ignore_index=True),
        pd.concat(bases, ignore_index=True),
        pd.concat(metrics, ignore_index=True),
        pd.concat(selected_reports, ignore_index=True),
    )