from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import json

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.dummy import DummyRegressor

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
from etl_disease_model_predict.features.dim_features import (
    load_dim_location_mapping,
    _normalize_location_name,
)

KEY_COLS = ["data_source", "disease", "county"]
CATEGORICAL_COLS = ["county", "disease", "data_source"]
OFFSHORE_COUNTIES = {
    "金門縣",
    "連江縣",
    "澎湖縣",
}


WEATHER_COLUMN_KEYWORDS = (
    "weather",
    "temp",
    "temperature",
    "airtemperature",
    "humidity",
    "humd",
    "rain",
    "precip",
    "pressure",
    "pres",
    "wind",
    "wdsd",
    "wd15",
    "ws15",
    "sun",
    "sunshine",
    "uv",
    "cloud",
    "dew",
)

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

def _merge_ev_county_to_branch_t(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 EV 的 county 轉為 dim_location.branch_t。

    使用 old_name / new_name 對照，避免 2005-2026 期間縣市名稱變動造成 mapping 失敗。
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    if "county" not in out.columns:
        raise RuntimeError("EV branch mapping requires county column")

    mapping = load_dim_location_mapping()

    out["county_key"] = out["county"].apply(_normalize_location_name)

    out = out.merge(
        mapping[["county_key", "new_name", "branch_t"]],
        on="county_key",
        how="left",
    )

    missing = (
        out.loc[out["branch_t"].isna(), "county"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if missing:
        raise RuntimeError(
            "Some EV counties cannot map to dim_location.branch_t: "
            f"{missing}"
        )

    out["county"] = out["branch_t"].astype(str)

    out = out.drop(
        columns=["county_key", "new_name", "branch_t"],
        errors="ignore",
    )

    out = _collapse_panel_rows(out)

    return out

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

    # =====================================================
    # 1. disease count lag / rolling
    # =====================================================
    count_grp = out.groupby(KEY_COLS, dropna=False)["count"]

    for lag in cfg["lags"]:
        out[f"lag_{lag}"] = count_grp.shift(lag)

    for window in cfg["rolling_windows"]:
        out[f"roll{window}_mean"] = count_grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        out[f"roll{window}_std"] = count_grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )

    for lag in cfg["diff_lags"]:
        # diff_lag 定義為「最近一期已知值」與「lag+1 週前已知值」的差，
        # 兩者都在目標週之前，不會用到當週 count，避免 target leakage。
        # 這裡直接用 shift(lag + 1) 取比較值，不依賴 lag_(lag+1) 是否剛好
        # 也在 cfg["lags"] 清單中，確保設定的 diff_lags 都能真正產生對應欄位。
        compare_series = count_grp.shift(lag + 1)
        out[f"diff_{lag}"] = out["lag_1"] - compare_series

    for lag in cfg["growth_lags"]:
        # growth_lag 定義與 diff_lag 相同的比較基準，
        # 分母同樣是 lag+1 週前已知值，避免除以當週 count。
        compare_series = count_grp.shift(lag + 1)
        denom = compare_series.replace(0, np.nan)
        out[f"growth_{lag}"] = (out["lag_1"] - compare_series) / denom

    # =====================================================
    # 2. source total count lag / rolling
    #    不直接使用同週 source_total_count，避免 future 不穩定
    # =====================================================
    if "source_total_count" in out.columns:
        total_grp = out.groupby(KEY_COLS, dropna=False)["source_total_count"]

        for lag in cfg["lags"]:
            out[f"source_total_count_lag_{lag}"] = total_grp.shift(lag)

        for window in cfg["rolling_windows"]:
            out[f"source_total_count_roll{window}_mean"] = total_grp.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
            out[f"source_total_count_roll{window}_std"] = total_grp.transform(
                lambda s: s.shift(1).rolling(window, min_periods=2).std()
            )

    # =====================================================
    # 3. disease rate lag / rolling
    #    disease_rate = count / source_total_count
    #    只使用 lag / rolling，不使用同週 disease_rate
    # =====================================================
    if "disease_rate" in out.columns:
        rate_grp = out.groupby(KEY_COLS, dropna=False)["disease_rate"]

        for lag in cfg["lags"]:
            out[f"disease_rate_lag_{lag}"] = rate_grp.shift(lag)

        for window in cfg["rolling_windows"]:
            out[f"disease_rate_roll{window}_mean"] = rate_grp.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
            out[f"disease_rate_roll{window}_std"] = rate_grp.transform(
                lambda s: s.shift(1).rolling(window, min_periods=2).std()
            )

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
        or c.startswith("source_total_count_lag_")
        or c.startswith("source_total_count_roll")
        or c.startswith("disease_rate_lag_")
        or c.startswith("disease_rate_roll")
        or c in {"week_sin", "week_cos"}
    ]

    numeric_cols = list(
        dict.fromkeys(engineered_cols + [c for c in exog_cols if c in df.columns])
    )

    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    feature_cols = categorical_cols + numeric_cols

    return feature_cols, numeric_cols, categorical_cols

def _is_weather_column(col: str) -> bool:
    col_lower = str(col).lower()

    return any(keyword in col_lower for keyword in WEATHER_COLUMN_KEYWORDS)


def _drop_weather_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    weather_cols = [
        c for c in df.columns
        if _is_weather_column(c)
    ]

    if not weather_cols:
        return df.copy(), []

    out = df.drop(columns=weather_cols, errors="ignore").copy()

    return out, weather_cols

def _sum_with_nan_if_all_missing(s: pd.Series):
    return s.sum(min_count=1)

def _collapse_panel_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    將模型前調整後可能重複的 panel rows 合併。

    例如：
        金門縣、連江縣、澎湖縣 -> 離島
        RODS EV 各縣市 -> 全國

    合併邏輯：
        count              加總
        source_total_count 加總
        disease_rate       重新計算
        其他 numeric        平均
        其他欄位            取第一筆
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    required_cols = {
        "data_source",
        "disease",
        "county",
        "yearweek",
        "is_future",
    }

    missing_cols = required_cols - set(out.columns)

    if missing_cols:
        raise ValueError(
            f"_collapse_panel_rows missing columns: {sorted(missing_cols)}"
        )

    group_cols = [
        "data_source",
        "disease",
        "county",
        "yearweek",
        "is_future",
    ]

    agg_map = {}

    for col in out.columns:
        if col in group_cols:
            continue

        if col in {"count", "source_total_count"}:
            agg_map[col] = _sum_with_nan_if_all_missing
        elif pd.api.types.is_numeric_dtype(out[col]):
            agg_map[col] = "mean"
        else:
            agg_map[col] = "first"

    collapsed = (
        out
        .groupby(group_cols, as_index=False, dropna=False)
        .agg(agg_map)
    )

    if {"count", "source_total_count"}.issubset(collapsed.columns):
        collapsed["disease_rate"] = np.where(
            collapsed["count"].notna()
            & collapsed["source_total_count"].notna()
            & (collapsed["source_total_count"] > 0),
            collapsed["count"] / collapsed["source_total_count"],
            np.nan,
        )

    collapsed = (
        collapsed
        .sort_values(["data_source", "disease", "county", "yearweek", "is_future"])
        .reset_index(drop=True)
    )

    return collapsed


def _merge_offshore_counties(df: pd.DataFrame) -> pd.DataFrame:
    """
    模型前將非本島縣市整併成「離島」。

    不改資料庫資料，只改本次進模型的 feature table。
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    if "county" not in out.columns:
        return out

    out["county"] = out["county"].astype(str).str.strip()
    out.loc[out["county"].isin(OFFSHORE_COUNTIES), "county"] = "離島"

    out = _collapse_panel_rows(out)

    return out


def _prepare_feature_for_model_task(
    feature: pd.DataFrame,
    data_source: str,
    model_task: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    依照 modeling task 做模型前資料調整。

    model_task:
        nhi_ev_branch
            NHI ER / OPD 的 EV 改為 branch_t 分區建模。

        nhi_non_ev_county
            NHI ER / OPD 的 DI / ILI 維持縣市建模，非本島整併為離島。

        rods_ev_national
            RODS EV 全國建模，移除 weather。

        rods_non_ev
            RODS 非 EV 縣市建模，非本島整併為離島。
    """
    if feature.empty:
        return feature.copy(), []

    out = feature.copy()
    removed_weather_cols: list[str] = []

    if data_source in {"nhi_er", "nhi_opd"} and model_task == "nhi_ev_branch":
        out = out[out["disease"].astype(str) == "EV"].copy()

        if out.empty:
            raise RuntimeError(
                f"{data_source} EV feature table is empty after filtering disease=EV"
            )

        out = _merge_ev_county_to_branch_t(out)

        return out, removed_weather_cols

    if data_source in {"nhi_er", "nhi_opd"} and model_task == "nhi_non_ev_county":
        out = out[out["disease"].astype(str) != "EV"].copy()

        if out.empty:
            raise RuntimeError(
                f"{data_source} non-EV feature table is empty after filtering disease!=EV"
            )

        out = _merge_offshore_counties(out)

        return out, removed_weather_cols

    if data_source == "rods" and model_task == "rods_ev_national":
        out = out[out["disease"].astype(str) == "EV"].copy()

        if out.empty:
            raise RuntimeError("RODS EV feature table is empty after filtering disease=EV")

        out["county"] = "全國"
        out = _collapse_panel_rows(out)

        out, removed_weather_cols = _drop_weather_columns(out)

        return out, removed_weather_cols

    if data_source == "rods" and model_task == "rods_non_ev":
        out = out[out["disease"].astype(str) != "EV"].copy()

        if out.empty:
            raise RuntimeError("RODS non-EV feature table is empty after filtering disease!=EV")

        out = _merge_offshore_counties(out)

        return out, removed_weather_cols

    out = _merge_offshore_counties(out)

    return out, removed_weather_cols

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

    total_history = (
        hist["source_total_count"].astype(float).dropna().tolist()
        if "source_total_count" in hist.columns
        else []
    )

    rate_history = (
        hist["disease_rate"].astype(float).dropna().tolist()
        if "disease_rate" in hist.columns
        else []
    )

    rows = []

    for _, row in future.iterrows():
        item = row.to_dict()

        if "week" in item and pd.notna(item["week"]):
            item["week_sin"] = np.sin(2 * np.pi * float(item["week"]) / 52)
            item["week_cos"] = np.cos(2 * np.pi * float(item["week"]) / 52)

        # =====================================================
        # 1. count lag
        # =====================================================
        for lag in cfg["lags"]:
            item[f"lag_{lag}"] = (
                count_history[-lag]
                if len(count_history) >= lag
                else np.nan
            )

        # =====================================================
        # 2. count rolling
        # =====================================================
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

        # =====================================================
        # 3. count diff / growth
        #    定義與訓練時完全一致：base 為 lag_1（最近一期已知值），
        #    compare 為 lag_(lag+1)（lag+1 週前已知值），
        #    直接從 count_history 取值，不透過 item 裡的 lag_{lag} 欄位，
        #    避免與訓練定義產生一位之差。
        # =====================================================
        base_value = item.get("lag_1", np.nan)

        for lag in cfg["diff_lags"]:
            compare_value = (
                count_history[-(lag + 1)]
                if len(count_history) >= lag + 1
                else np.nan
            )

            item[f"diff_{lag}"] = (
                base_value - compare_value
                if pd.notna(base_value) and pd.notna(compare_value)
                else np.nan
            )

        for lag in cfg["growth_lags"]:
            compare_value = (
                count_history[-(lag + 1)]
                if len(count_history) >= lag + 1
                else np.nan
            )

            if (
                pd.notna(base_value)
                and pd.notna(compare_value)
                and compare_value != 0
            ):
                item[f"growth_{lag}"] = (base_value - compare_value) / compare_value
            else:
                item[f"growth_{lag}"] = np.nan

        # =====================================================
        # 4. source_total_count lag / rolling
        # =====================================================
        if total_history:
            for lag in cfg["lags"]:
                item[f"source_total_count_lag_{lag}"] = (
                    total_history[-lag]
                    if len(total_history) >= lag
                    else np.nan
                )

            for window in cfg["rolling_windows"]:
                vals = total_history[-window:]

                item[f"source_total_count_roll{window}_mean"] = (
                    float(np.mean(vals))
                    if vals
                    else np.nan
                )

                item[f"source_total_count_roll{window}_std"] = (
                    float(np.std(vals, ddof=1))
                    if len(vals) > 1
                    else 0.0
                )

        # =====================================================
        # 5. disease_rate lag / rolling
        # =====================================================
        if rate_history:
            for lag in cfg["lags"]:
                item[f"disease_rate_lag_{lag}"] = (
                    rate_history[-lag]
                    if len(rate_history) >= lag
                    else np.nan
                )

            for window in cfg["rolling_windows"]:
                vals = rate_history[-window:]

                item[f"disease_rate_roll{window}_mean"] = (
                    float(np.mean(vals))
                    if vals
                    else np.nan
                )

                item[f"disease_rate_roll{window}_std"] = (
                    float(np.std(vals, ddof=1))
                    if len(vals) > 1
                    else 0.0
                )

        rows.append(item)

        # 未來 recursive：count 先用最近一期 proxy 延續
        count_history.append(current_proxy)

        # total/rate 未來值暫時沿用最近一期，避免第 2 週後 lag 斷掉
        if total_history:
            total_history.append(total_history[-1])

        if rate_history:
            rate_history.append(rate_history[-1])

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

def _is_constant_target(y) -> bool:
    y_series = pd.to_numeric(pd.Series(y), errors="coerce").dropna()

    if y_series.empty:
        return True

    return y_series.nunique(dropna=True) <= 1


def _fit_constant_target_model(x: pd.DataFrame, y, model_name: str):
    y_series = pd.to_numeric(pd.Series(y), errors="coerce").dropna()

    if y_series.empty:
        constant_value = 0.0
    else:
        constant_value = float(y_series.iloc[0])

    print(
        f"[WARN] model={model_name} skipped normal training because "
        f"all train targets are equal. fallback=DummyRegressor, "
        f"constant={constant_value}",
        flush=True,
    )

    fallback = DummyRegressor(
        strategy="constant",
        constant=constant_value,
    )

    fallback.fit(
        x,
        np.repeat(constant_value, len(x)),
    )

    return fallback

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

    保護機制：
        1. y 全部相同時，直接使用 DummyRegressor。
        2. GridSearch 關閉時，直接 fit。
        3. GridSearch 沒有參數表或 CV 不足時，直接 fit。
        4. GridSearch 失敗時，退回原模型 fit。
        5. CatBoost / 其他模型遇到 All train targets are equal 時，退回 DummyRegressor。
    """
    x = train_df[feature_cols]
    y = train_df[y_col].astype(float)

    def _plain_fit():
        try:
            model.fit(x, y)
            return model

        except Exception as exc:
            if _is_constant_target(y) or "All train targets are equal" in str(exc):
                return _fit_constant_target_model(
                    x=x,
                    y=y,
                    model_name=model_name,
                )

            raise

    if _is_constant_target(y):
        return _fit_constant_target_model(
            x=x,
            y=y,
            model_name=model_name,
        )

    if not enable_grid_search:
        return _plain_fit()

    param_grid = _param_grid_for_model(model_name)
    param_grid = _filter_valid_param_grid(model, param_grid)

    if not param_grid:
        return _plain_fit()

    cv = _yearweek_cv_splits(
        train_df,
        n_splits=grid_cv_splits,
        min_train_weeks=52,
    )

    if not cv:
        return _plain_fit()

    print(
        f"[GRID] model={model_name}, "
        f"candidates={np.prod([len(v) for v in param_grid.values()])}, "
        f"cv_splits={len(cv)}",
        flush=True,
    )

    try:
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
            f"best_params={json.dumps(search.best_params_, ensure_ascii=False)}",
            flush=True,
        )

        return search.best_estimator_

    except ValueError as exc:
        print(
            f"[WARN] GridSearch failed for model={model_name}; "
            f"fallback to plain fit. error={type(exc).__name__}: {exc}",
            flush=True,
        )

        return _plain_fit()

    except Exception as exc:
        if _is_constant_target(y) or "All train targets are equal" in str(exc):
            return _fit_constant_target_model(
                x=x,
                y=y,
                model_name=model_name,
            )

        raise

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


def _make_base_metric_context(
    data_source: str,
    run_mode: str,
    feature_set: str,
    feature_select: str,
    top_k: int | None,
    covid_policy: str,
    enable_grid_search: bool,
    now: str,
) -> dict:
    return {
        "data_source": data_source,
        "run_mode": run_mode,
        "feature_set": feature_set,
        "feature_select": feature_select,
        "top_k": top_k,
        "covid_policy": covid_policy,
        "enable_grid_search": enable_grid_search,
        "prediction_type": "oof",
        "metric_level": "unknown",
        "disease": "ALL",
        "county": "ALL",
        "created_at": now,
    }


def _build_base_metric_rows(
    train: pd.DataFrame,
    oof: pd.DataFrame,
    base_context: dict,
) -> list[dict]:
    """
    建立 base model 的 metric rows。

    跟 meta model 選擇無關，一個 data_source/model_task 只需要算一次，
    比較多個 meta model 時可以重複使用同一份結果，不用重算。

    每個 base model 各自使用自己的 OOF 有效列（notna），
    不會因為另一個模型（例如 sarimax 只在序列夠長時才有預測）缺值，
    就連帶縮小其他模型的評估樣本。
    """
    rows: list[dict] = []
    id_cols = KEY_COLS + ["yearweek", "count"]

    for model_name in oof.columns:
        model_mask = oof[model_name].notna()

        tmp = train.loc[model_mask, id_cols].rename(columns={"count": "y_true"}).copy()
        tmp["y_pred"] = oof.loc[model_mask, model_name].values

        if tmp.empty:
            continue

        model_context = base_context.copy()
        model_context["model_layer"] = "base"
        model_context["model_name"] = model_name

        context = model_context.copy()
        context["metric_level"] = "overall"
        rows.append(_metric_row(tmp["y_true"], tmp["y_pred"], context))

        for disease, g in tmp.groupby("disease", dropna=False):
            context = model_context.copy()
            context["metric_level"] = "by_disease"
            context["disease"] = disease
            rows.append(_metric_row(g["y_true"], g["y_pred"], context))

        for county, g in tmp.groupby("county", dropna=False):
            context = model_context.copy()
            context["metric_level"] = "by_county"
            context["county"] = county
            rows.append(_metric_row(g["y_true"], g["y_pred"], context))

        for (disease, county), g in tmp.groupby(
            ["disease", "county"],
            dropna=False,
        ):
            context = model_context.copy()
            context["metric_level"] = "by_disease_county"
            context["disease"] = disease
            context["county"] = county
            rows.append(_metric_row(g["y_true"], g["y_pred"], context))

    return rows


def _build_stacking_metric_rows(
    train: pd.DataFrame,
    stack_valid_mask: pd.Series,
    stacked_oof: np.ndarray,
    meta_model: str,
    base_context: dict,
) -> list[dict]:
    """
    建立單一 meta model 的 stacking metric rows。

    每比較一個 meta model 就呼叫一次，因為 base model 的部分
    （_build_base_metric_rows）不會重複計算，這裡開銷很小。
    """
    rows: list[dict] = []
    id_cols = KEY_COLS + ["yearweek", "count"]

    stack_df = train.loc[stack_valid_mask, id_cols].rename(columns={"count": "y_true"}).copy()
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

    return rows


def _finalize_metric_df(rows: list[dict]) -> pd.DataFrame:
    """把 metric rows 組成 DataFrame，並確保 metric_level/disease/county 不缺值。"""
    metric_df = pd.DataFrame(rows)

    for col in ["metric_level", "disease", "county"]:
        if col not in metric_df.columns:
            metric_df[col] = "ALL"

    metric_df["metric_level"] = metric_df["metric_level"].fillna("unknown")
    metric_df["disease"] = metric_df["disease"].fillna("ALL")
    metric_df["county"] = metric_df["county"].fillna("ALL")

    return metric_df


def _build_eval_metric_df(
    train: pd.DataFrame,
    oof: pd.DataFrame,
    stacked_oof: np.ndarray,
    stack_valid_mask: pd.Series,
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
    建立單一 meta model 情境下的完整 metric_df（base + 這個 meta model 的 stacking）。

    內部呼叫 _build_base_metric_rows() 與 _build_stacking_metric_rows()，
    只是把兩者組起來，維持舊有單一 meta model 呼叫方式的相容性。
    比較多個 meta model 時請改用這兩個拆開的函式，避免 base rows 重算。
    """
    base_context = _make_base_metric_context(
        data_source=data_source,
        run_mode=run_mode,
        feature_set=feature_set,
        feature_select=feature_select,
        top_k=top_k,
        covid_policy=covid_policy,
        enable_grid_search=enable_grid_search,
        now=now,
    )

    rows = _build_base_metric_rows(train, oof, base_context)
    rows += _build_stacking_metric_rows(
        train, stack_valid_mask, stacked_oof, meta_model, base_context
    )

    return _finalize_metric_df(rows)

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

def _fit_predict_meta_oof(
    oof_valid: pd.DataFrame,
    y_valid: pd.Series,
    yearweek_valid: pd.Series,
    method: str,
    n_splits: int,
    min_train_weeks: int = 12,
) -> np.ndarray:
    """
    對 meta model 做 rolling OOF。

    目的：
        避免用同一批 base OOF 訓練 meta model 後，
        又回頭預測同一批資料，造成 stacking metric 過度樂觀。

    注意：
        這是 meta 層級的 OOF，不是重新訓練 base model。
    """
    x = oof_valid.reset_index(drop=True).copy()
    y = pd.to_numeric(y_valid.reset_index(drop=True), errors="coerce")
    weeks = pd.to_numeric(yearweek_valid.reset_index(drop=True), errors="coerce")

    work = pd.DataFrame(
        {
            "yearweek": weeks,
            "count": y,
        }
    )

    mask = work["yearweek"].notna() & work["count"].notna()

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        mask &= x[col].notna()

    x = x.loc[mask].reset_index(drop=True)
    work = work.loc[mask].reset_index(drop=True)

    pred = np.full(len(oof_valid), np.nan, dtype=float)

    if x.empty:
        return pred

    try:
        splits = _rolling_splits(
            work,
            n_splits=max(2, min(n_splits, 5)),
            min_train_weeks=min_train_weeks,
        )
    except Exception:
        return pred

    original_positions = np.where(mask.to_numpy())[0]

    for tr_idx, va_idx in splits:
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        try:
            meta_fold = fit_meta_model(
                x.loc[tr_idx],
                work.loc[tr_idx, "count"],
                method=method,
            )

            fold_pred = predict_meta(
                meta_fold,
                x.loc[va_idx],
            )

            pred[original_positions[va_idx]] = fold_pred

        except Exception as exc:
            print(
                f"[WARN] meta OOF failed: "
                f"method={method}, "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )

    return pred


@dataclass(frozen=True)
class MetaModelResult:
    """單一 meta model 的評估結果，供多 meta model 比較時使用。"""
    method: str
    stacked_oof: np.ndarray
    meta: object
    meta_eval_type: str


def _evaluate_meta_models(
    oof_valid: pd.DataFrame,
    y_valid: pd.Series,
    yearweek_valid: pd.Series,
    methods: list[str],
    n_splits: int,
) -> dict[str, MetaModelResult]:
    """
    對多個 meta model 方法各自做 rolling OOF 評估，重複使用同一份 base OOF。

    base model（xgboost/lightgbm/catboost...）只需要在呼叫這個函式之前訓練一次，
    這裡只針對 meta model 這一層（輸入欄位數 = base model 數量，通常很小）重複訓練，
    比每個 meta model 都重跑一次完整 pipeline 快得多，也是 --compare-meta-models
    的效能基礎。
    """
    results: dict[str, MetaModelResult] = {}

    for method in methods:
        stacked_oof = _fit_predict_meta_oof(
            oof_valid=oof_valid,
            y_valid=y_valid,
            yearweek_valid=yearweek_valid,
            method=method,
            n_splits=n_splits,
            min_train_weeks=12,
        )

        if np.isnan(stacked_oof).all():
            print(
                f"[WARN] honest meta OOF unavailable for meta_model={method}; "
                f"fallback to in-sample meta metric.",
                flush=True,
            )
            meta_eval_type = "insample_meta"
            meta = fit_meta_model(oof_valid, y_valid, method=method)
            stacked_oof = predict_meta(meta, oof_valid)
        else:
            meta_eval_type = "rolling_meta_oof"
            meta = fit_meta_model(oof_valid, y_valid, method=method)

        results[method] = MetaModelResult(
            method=method,
            stacked_oof=stacked_oof,
            meta=meta,
            meta_eval_type=meta_eval_type,
        )

    return results


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


def resolve_task_metadata(model_task: str) -> dict:
    """
    依 model_task 決定輸出用的中繼資料。

    這裡是唯一定義 model_scope / is_rods_ev_national / is_offshore_collapsed
    的地方，train_predict.run_source() 與 holdout_eval.run_task() 都呼叫這個
    函式，避免兩邊各自重複實作而彼此漂移不一致。

    model_scope：
        rods_ev_national → national
        nhi_ev_branch     → branch
        其他（county 建模） → county
    """
    if model_task == "rods_ev_national":
        model_scope = "national"
    elif model_task == "nhi_ev_branch":
        model_scope = "branch"
    else:
        model_scope = "county"

    return {
        "model_scope": model_scope,
        "is_rods_ev_national": model_task == "rods_ev_national",
        "is_offshore_collapsed": model_task in {
            "default",
            "nhi_non_ev_county",
            "rods_non_ev",
        },
    }


def resolve_allowed_registry(
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_task: str,
    cfg_model_names: list[str],
    use_gpu: bool = False,
    enable_sarimax: bool = False,
) -> list:
    """
    建立並篩選出實際要使用的 base model registry。

    這裡是唯一決定「哪些模型會真正參與 OOF / stacking」的地方，
    train_predict.run_source() 與 holdout_eval.run_task() 都呼叫這個函式，
    避免各自實作 allowed_model_names 篩選邏輯而彼此不一致。

    篩選規則：
        1. 從 run_mode 設定的 model_names 開始（排除尚未正式啟用的 keras_mlp）。
        2. EV 任務（nhi_ev_branch / rods_ev_national）額外加入 poisson 系列模型，
           並移除 ridge（count-oriented 模型在 EV 任務上通常更穩定）。
           elasticnet 已經在 build_base_registry() 依 model_task 排除，這裡不用重複處理。
        3. enable_sarimax 決定是否加入 sarimax。
    """
    registry = build_base_registry(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_gpu=use_gpu,
        model_task=model_task,
    )

    allowed_model_names = set(cfg_model_names) - {"keras_mlp"}

    if model_task in {"nhi_ev_branch", "rods_ev_national"}:
        allowed_model_names.discard("ridge")
        allowed_model_names.update(
            {
                "xgboost_poisson",
                "lightgbm_poisson",
                "catboost_poisson",
            }
        )

    if enable_sarimax:
        allowed_model_names.add("sarimax")
    else:
        allowed_model_names.discard("sarimax")

    registry = [spec for spec in registry if spec.name in allowed_model_names]

    if len(registry) < 2:
        raise RuntimeError(
            f"Need at least 2 models after registry filtering, "
            f"got {len(registry)} for model_task={model_task}, "
            f"models={[spec.name for spec in registry]}"
        )

    return registry


def _select_final_model(
    metric_df: pd.DataFrame,
    meta_eval_type: str,
    meta_model: str,
    stacked_future_pred: np.ndarray,
    future_base: pd.DataFrame,
) -> tuple[np.ndarray, str, str, float]:
    """
    在 base model 與 stacking model 之間選出最終預測。

    規則（對應限制條件：不能固定使用 stacking）：
        1. stacking 只有在 meta_eval_type == "rolling_meta_oof"（honest OOF）時，
           才有資格被拿來跟 base model 比較。
        2. meta_eval_type == "insample_meta" 時，stacking metric 是用同一批 OOF
           訓練後又回頭預測，屬於過度樂觀的評估，不能拿來做最終選模依據，
           一律優先採用 base model 中 overall WAPE 最低者。
        3. 兩者都可比較時，取 overall WAPE 較低者作為最終預測。

    注意：
        metric_df 可能同時包含多個 meta model 的 stacking 比較結果（--compare-meta-models），
        這裡只挑 model_name == "stacking_{meta_model}" 這一列（也就是實際要拿來
        產生 forecast_df 的「主要」meta model），不會跟其他純比較用的 meta model 搞混。

    回傳：
        (final_pred, final_model_name, final_model_layer, selected_wape)
    """
    overall = metric_df.loc[metric_df["metric_level"].eq("overall")].copy()

    base_overall = overall.loc[
        overall["model_layer"].eq("base")
        & overall["WAPE"].notna()
        & np.isfinite(overall["WAPE"])
    ].sort_values("WAPE", ascending=True)

    stack_overall = overall.loc[
        overall["model_layer"].eq("stacking")
        & overall["model_name"].eq(f"stacking_{meta_model}")
    ]
    stack_wape = np.nan

    if not stack_overall.empty:
        candidate = stack_overall["WAPE"].iloc[0]
        if pd.notna(candidate) and np.isfinite(candidate):
            stack_wape = float(candidate)

    stacking_is_honest = meta_eval_type == "rolling_meta_oof" and pd.notna(stack_wape)

    best_base_wape = (
        float(base_overall["WAPE"].iloc[0]) if not base_overall.empty else np.nan
    )
    best_base_name = (
        base_overall["model_name"].iloc[0] if not base_overall.empty else None
    )

    use_stacking = stacking_is_honest and (
        pd.isna(best_base_wape) or stack_wape < best_base_wape
    )

    if use_stacking:
        return (
            stacked_future_pred,
            f"stacking_{meta_model}",
            "stacking",
            stack_wape,
        )

    if best_base_name is not None and best_base_name in future_base.columns:
        return (
            future_base[best_base_name].to_numpy(),
            str(best_base_name),
            "base",
            best_base_wape,
        )

    # 沒有可信的 base model metric，也沒有 honest stacking 可用時，
    # 仍以 stacking 預測作為保底輸出，避免完全沒有 forecast 結果。
    return (
        stacked_future_pred,
        f"stacking_{meta_model}",
        "stacking",
        np.nan,
    )


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
    model_task: str = "default",
    compare_meta_models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    compare_meta_models：
        除了 meta_model（用來產生正式 forecast_df 的主要 meta model）之外，
        額外要一併比較 OOF 表現的 meta model 方法清單。
        base model 只會訓練一次，不會因為比較多個 meta model 而重複訓練。
        最終 forecast_df 仍然只由 meta_model 這個「主要」方法（或 best base model）產生，
        其餘方法只會出現在 metric_df / leaderboard / breakdown 裡作比較用。
    """
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
        f"enable_sarimax={enable_sarimax}",
        f"model_task={model_task}"
    )

    feature = build_feature_table(
        data_source=data_source,
        start_week=start_week,
        forecast_period=forecast_period,
    )

    feature, removed_weather_cols = _prepare_feature_for_model_task(
        feature=feature,
        data_source=data_source,
        model_task=model_task,
    )

    if removed_weather_cols:
        print(
            f"[MODEL_ADJUST] source={data_source}, "
            f"model_task={model_task}, "
            f"removed_weather_cols={len(removed_weather_cols)}",
            flush=True,
        )

    print(
        f"[MODEL_ADJUST] source={data_source}, "
        f"model_task={model_task}, "
        f"rows={len(feature)}, "
        f"diseases={sorted(feature['disease'].dropna().astype(str).unique().tolist())}, "
        f"county_count={feature['county'].nunique(dropna=True)}, "
        f"counties={sorted(feature['county'].dropna().astype(str).unique().tolist())[:30]}",
        flush=True,
    )

    feature = _apply_covid_policy(
        df=feature,
        covid_policy=covid_policy,
    )

    exog_cols = get_exog_columns(feature)

    # 避免目標洩漏：
    # source_total_count 與 disease_rate 會直接或間接包含當週 count，
    # 只能使用 add_lag_rolling_features() 產生的 lag / rolling 版本，
    # 不能把原始同週欄位放進模型特徵。
    target_derived_exog_cols = {
        "source_total_count",
        "disease_rate",
    }
    leaked_exog_cols = [
        c for c in exog_cols
        if c in target_derived_exog_cols
    ]
    if leaked_exog_cols:
        print(
            f"[LEAKAGE_GUARD] source={data_source}, "
            f"model_task={model_task}, "
            f"removed_exog_cols={leaked_exog_cols}",
            flush=True,
        )

    exog_cols = [
        c for c in exog_cols
        if c not in target_derived_exog_cols
    ]
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

    registry = resolve_allowed_registry(
        numeric_cols=selected_numeric_cols,
        categorical_cols=categorical_cols,
        model_task=model_task,
        cfg_model_names=cfg.model_names,
        use_gpu=use_gpu,
        enable_sarimax=enable_sarimax,
    )

    print(
        f"[MODEL_POOL] source={data_source}, "
        f"model_task={model_task}, "
        f"models={[spec.name for spec in registry]}",
        flush=True,
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

    # stacking 需要完整的 base 特徵，這裡先篩出所有 base model 都有預測的列，
    # 只用於建立 meta model 的訓練矩陣，不會拿來限制 base model 自己的評估樣本。
    stack_valid_mask = oof.notna().all(axis=1)

    if stack_valid_mask.sum() == 0:
        raise RuntimeError(f"OOF prediction is empty for source={data_source}")

    if oof.shape[1] < 2:
        raise RuntimeError(
            f"Need at least 2 base model predictions for stacking, "
            f"got {oof.shape[1]} for source={data_source}"
        )

    oof_valid = oof.loc[stack_valid_mask].copy()
    y_valid = train.loc[stack_valid_mask, "count"].copy()
    yearweek_valid = train.loc[stack_valid_mask, "yearweek"].copy()

    # 主要 meta model（用來產生 forecast_df）永遠會被評估，
    # compare_meta_models 是額外要一起比較、但不會拿去產生 forecast 的方法。
    meta_methods_to_run = list(
        dict.fromkeys([meta_model] + list(compare_meta_models or []))
    )

    meta_results = _evaluate_meta_models(
        oof_valid=oof_valid,
        y_valid=y_valid,
        yearweek_valid=yearweek_valid,
        methods=meta_methods_to_run,
        n_splits=cfg.n_splits,
    )

    primary_result = meta_results[meta_model]
    stacked_oof = primary_result.stacked_oof
    meta = primary_result.meta
    meta_eval_type = primary_result.meta_eval_type

    if len(meta_methods_to_run) > 1:
        print(
            f"[META_COMPARE] source={data_source}, model_task={model_task}, "
            f"methods={meta_methods_to_run}",
            flush=True,
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_meta = resolve_task_metadata(model_task)
    model_scope = task_meta["model_scope"]
    is_rods_ev_national = task_meta["is_rods_ev_national"]
    is_offshore_collapsed = task_meta["is_offshore_collapsed"]

    weather_removed = len(removed_weather_cols) > 0

    base_context = _make_base_metric_context(
        data_source=data_source,
        run_mode=run_mode,
        feature_set=cfg.feature_set,
        feature_select=cfg.feature_select,
        top_k=cfg.top_k,
        covid_policy=covid_policy,
        enable_grid_search=enable_grid_search,
        now=now,
    )

    # base model 的 metric 只算一次，不會因為比較多個 meta model 而重算。
    rows = _build_base_metric_rows(train, oof, base_context)

    # 每個 meta model（含主要與比較用的）各自產生自己的 stacking rows，
    # 並各自標記正確的 meta_model / meta_eval_type，不會互相混淆。
    for method, result in meta_results.items():
        method_rows = _build_stacking_metric_rows(
            train, stack_valid_mask, result.stacked_oof, method, base_context
        )
        for row in method_rows:
            row["meta_model"] = method
            row["meta_eval_type"] = result.meta_eval_type
        rows.extend(method_rows)

    metric_df = _finalize_metric_df(rows)
    metric_df["model_task"] = model_task
    metric_df["model_scope"] = model_scope
    metric_df["is_rods_ev_national"] = is_rods_ev_national
    metric_df["is_offshore_collapsed"] = is_offshore_collapsed
    metric_df["weather_removed"] = weather_removed

    # base model rows 沒有自己的 meta model，這裡統一標記成主要 meta model 作參考，
    # 並固定 meta_eval_type=base_oof，維持既有 schema 相容性。
    base_mask = metric_df["model_layer"].eq("base")
    metric_df.loc[base_mask, "meta_model"] = meta_model
    metric_df.loc[base_mask, "meta_eval_type"] = "base_oof"

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

    # 只用主要 meta model 產生實際要輸出的 forecast，比較用的 meta model
    # 不會拿去做未來預測，只出現在 metric_df 裡供比較。
    stacked_future_pred = predict_meta(meta, future_base)

    # 最終選模：只有在 meta_eval_type == "rolling_meta_oof"（honest OOF）時，
    # stacking 才有資格跟 base model 比較 overall WAPE；
    # 若 stacking 仍是 insample_meta（過度樂觀的評估），一律優先採用 base model。
    final_pred, final_model_name, final_model_layer, best_wape = _select_final_model(
        metric_df=metric_df,
        meta_eval_type=meta_eval_type,
        meta_model=meta_model,
        stacked_future_pred=stacked_future_pred,
        future_base=future_base,
    )

    print(
        f"[FINAL_MODEL] source={data_source}, "
        f"model_task={model_task}, "
        f"final_model={final_model_layer}/{final_model_name}, "
        f"meta_eval_type={meta_eval_type}, "
        f"selected_wape={best_wape}",
        flush=True,
    )

    id_cols = KEY_COLS + ["yearweek"]

    forecast_df = future[id_cols].copy()
    forecast_df["forecast_count"] = final_pred
    forecast_df["forecast_count_rounded"] = (
        np.clip(np.rint(forecast_df["forecast_count"].fillna(0)), 0, None)
        .astype(int)
    )
    forecast_df["model_name"] = final_model_name
    forecast_df["model_layer"] = final_model_layer
    forecast_df["model_task"] = model_task
    forecast_df["model_scope"] = model_scope
    forecast_df["is_rods_ev_national"] = is_rods_ev_national
    forecast_df["is_offshore_collapsed"] = is_offshore_collapsed
    forecast_df["weather_removed"] = weather_removed
    forecast_df["run_mode"] = run_mode
    forecast_df["feature_set"] = cfg.feature_set
    forecast_df["feature_select"] = cfg.feature_select
    forecast_df["top_k"] = cfg.top_k
    forecast_df["covid_policy"] = covid_policy
    forecast_df["enable_grid_search"] = enable_grid_search
    forecast_df["created_at"] = now

    base_rows = []

    for model_name in future_base.columns:
        tmp = future[id_cols].copy()
        tmp["base_model"] = model_name
        tmp["model_task"] = model_task
        tmp["model_scope"] = model_scope
        tmp["is_rods_ev_national"] = is_rods_ev_national
        tmp["is_offshore_collapsed"] = is_offshore_collapsed
        tmp["weather_removed"] = weather_removed
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
    selected_report["model_task"] = model_task
    selected_report["model_scope"] = model_scope
    selected_report["is_rods_ev_national"] = is_rods_ev_national
    selected_report["is_offshore_collapsed"] = is_offshore_collapsed
    selected_report["weather_removed"] = weather_removed
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
                "model_task": model_task,
                "model_scope": model_scope,
                "is_rods_ev_national": is_rods_ev_national,
                "is_offshore_collapsed": is_offshore_collapsed,
                "weather_removed": weather_removed,
            }
            for col in categorical_cols
        ]
    )

    selected_report = pd.concat(
        [selected_report, categorical_report],
        ignore_index=True,
    )

    return forecast_df, base_df, metric_df, selected_report

def _expand_model_tasks(data_sources: list[str]) -> list[dict]:
    """
    將 data_sources 展開成實際 modeling tasks。

    NHI：
        EV      → branch_t 分區
        non-EV  → county

    RODS：
        EV      → national
        non-EV  → county
    """
    tasks = []

    for source in data_sources:
        if source in {"nhi_er", "nhi_opd"}:
            tasks.append(
                {
                    "data_source": source,
                    "model_task": "nhi_ev_branch",
                }
            )
            tasks.append(
                {
                    "data_source": source,
                    "model_task": "nhi_non_ev_county",
                }
            )

        elif source == "rods":
            tasks.append(
                {
                    "data_source": "rods",
                    "model_task": "rods_ev_national",
                }
            )
            tasks.append(
                {
                    "data_source": "rods",
                    "model_task": "rods_non_ev",
                }
            )

        else:
            tasks.append(
                {
                    "data_source": source,
                    "model_task": "default",
                }
            )

    return tasks

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
    compare_meta_models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecasts = []
    bases = []
    metrics = []
    selected_reports = []

    tasks = _expand_model_tasks(data_sources)

    for task in tasks:
        source = task["data_source"]
        model_task = task["model_task"]

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
            compare_meta_models=compare_meta_models,
            covid_policy=covid_policy,
            enable_grid_search=enable_grid_search,
            grid_cv_splits=grid_cv_splits,
            model_task=model_task,
        )

        f["model_task"] = model_task
        b["model_task"] = model_task
        m["model_task"] = model_task
        s["model_task"] = model_task

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