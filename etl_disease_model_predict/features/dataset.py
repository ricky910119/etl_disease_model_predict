from __future__ import annotations

import numpy as np
import pandas as pd

from etl_disease_model_predict.config import settings
from etl_disease_model_predict.db.postgres import read_sql
from etl_disease_model_predict.features.dim_features import load_dim_features
from etl_disease_model_predict.utils.week import forecast_yearweeks

COUNT_COL_CANDIDATES = ["count", "weekly_count", "case_count", "cnt", "total", "value"]
COUNTY_COL_CANDIDATES = ["county", "city", "county_name", "city_name", "COUNTY", "CITY"]
DISEASE_COL_CANDIDATES = ["disease", "disease_name", "target_disease", "DISEASE"]
DROP_COLS = {"id", "created_at", "updated_at", "inserted_at", "modified_at"}


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise KeyError(f"Cannot find any column in {candidates}. Existing columns: {list(df.columns)}")
    return None


def _safe_numeric_yearweek(df: pd.DataFrame) -> pd.DataFrame:
    if "yearweek" not in df.columns:
        raise KeyError(f"Missing yearweek column. Existing columns: {list(df.columns)}")
    out = df.copy()
    out["yearweek"] = out["yearweek"].astype(int)
    return out


def load_source_weekly(data_source: str, start_week: int, end_week: int | None = None) -> pd.DataFrame:
    if data_source not in settings.source_tables:
        raise ValueError(f"Unknown data_source={data_source}. Allowed: {list(settings.source_tables)}")

    table = settings.source_tables[data_source]
    where = "WHERE yearweek >= :start_week"
    params = {"start_week": int(start_week)}
    if end_week is not None:
        where += " AND yearweek <= :end_week"
        params["end_week"] = int(end_week)

    df = read_sql(f"SELECT * FROM {table} {where}", params, dbname="postgres")
    if df.empty:
        raise RuntimeError(f"No source data found: source={data_source}, table={table}, start_week={start_week}, end_week={end_week}")
    df = _safe_numeric_yearweek(df)
    df["data_source"] = data_source
    return df


def load_weather_weekly(start_week: int, end_week: int | None = None) -> pd.DataFrame:
    where = "WHERE yearweek >= :start_week"
    params = {"start_week": int(start_week)}
    if end_week is not None:
        where += " AND yearweek <= :end_week"
        params["end_week"] = int(end_week)
    df = read_sql(f"SELECT * FROM {settings.weather_table} {where}", params, dbname="postgres")
    if df.empty:
        print(f"[WARN] weather table returned empty rows from {start_week} to {end_week}")
        return pd.DataFrame(columns=["yearweek", "county"])
    return _safe_numeric_yearweek(df)


def _normalize_target(target: pd.DataFrame) -> pd.DataFrame:
    target_county = _pick_col(target, COUNTY_COL_CANDIDATES)
    count_col = _pick_col(target, COUNT_COL_CANDIDATES)
    disease_col = _pick_col(target, DISEASE_COL_CANDIDATES, required=False)

    target = target.rename(columns={target_county: "county", count_col: "count"}).copy()
    if disease_col:
        target = target.rename(columns={disease_col: "disease"})
    else:
        target["disease"] = "ALL"

    target["county"] = target["county"].astype(str)
    target["disease"] = target["disease"].astype(str)
    target["count"] = pd.to_numeric(target["count"], errors="coerce")
    return target


def _normalize_weather(weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return pd.DataFrame(columns=["yearweek", "county"])

    weather_county = _pick_col(weather, COUNTY_COL_CANDIDATES)
    weather = weather.rename(columns={weather_county: "county"}).copy()
    weather["county"] = weather["county"].astype(str)

    drop_cols = [c for c in weather.columns if c.lower() in DROP_COLS]
    weather = weather.drop(columns=drop_cols, errors="ignore")
    keep_cols = [c for c in weather.columns if c not in ["yearweek", "county"]]
    return weather[["yearweek", "county"] + keep_cols].drop_duplicates(subset=["yearweek", "county"])


def build_feature_table(
    data_source: str,
    start_week: int,
    forecast_period: int,
    end_week: int | None = None,
) -> pd.DataFrame:
    """建立歷史訓練列與未來預測列。

    歷史列來自 model_*_weekly_county，未來列由歷史 county/disease 組合 cross forecast yearweek 產生。
    這樣即使 target 表沒有未來週，也能正常產出 forecast rows。
    """
    future_weeks = forecast_yearweeks(forecast_period)
    max_needed_week = max(future_weeks + ([end_week] if end_week else []))

    target = _normalize_target(load_source_weekly(data_source, start_week, end_week))
    weather = _normalize_weather(load_weather_weekly(start_week, max_needed_week))
    dim = load_dim_features()

    historical = target[["yearweek", "county", "disease", "data_source", "count"]].copy()
    historical["is_future"] = False

    combos = historical[["county", "disease", "data_source"]].drop_duplicates().reset_index(drop=True)
    future = combos.merge(pd.DataFrame({"yearweek": future_weeks}), how="cross")
    future["count"] = np.nan
    future["is_future"] = True
    future = future[["yearweek", "county", "disease", "data_source", "count", "is_future"]]

    df = pd.concat([historical, future], ignore_index=True)
    df = (
        df.merge(weather, on=["yearweek", "county"], how="left")
          .merge(dim, on="yearweek", how="left")
          .sort_values(["data_source", "disease", "county", "yearweek", "is_future"])
          .reset_index(drop=True)
    )
    return df


def get_exog_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "id", "yearweek", "county", "disease", "data_source", "count", "is_future",
        "created_at", "updated_at", "inserted_at", "modified_at",
    }
    cols = [c for c in df.columns if c not in excluded]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
