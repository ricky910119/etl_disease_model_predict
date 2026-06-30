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

DROP_COLS = {
    "id",
    "created_at",
    "updated_at",
    "inserted_at",
    "modified_at",
}

WIDE_DISEASE_COUNT_COLUMNS = {
    "ev_total": "EV",
    "ili_total": "ILI",
    "di_total": "DI",
    "u071_total": "U071",
}


def _pick_col(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise KeyError(
            f"Cannot find any column in {candidates}. "
            f"Existing columns: {list(df.columns)}"
        )

    return None


def _safe_numeric_yearweek(df: pd.DataFrame) -> pd.DataFrame:
    if "yearweek" not in df.columns:
        raise KeyError(f"Missing yearweek column. Existing columns: {list(df.columns)}")

    out = df.copy()
    out["yearweek"] = out["yearweek"].astype(int)
    return out


def load_source_weekly(
    data_source: str,
    start_week: int | None = None,
    end_week: int | None = None,
) -> pd.DataFrame:
    """
    讀取疾病週資料。

    start_week=None 時，不加起始週限制，使用資料表內所有歷史資料。
    """
    if data_source not in settings.source_tables:
        raise ValueError(
            f"Unknown data_source={data_source}. "
            f"Allowed: {list(settings.source_tables)}"
        )

    table = settings.source_tables[data_source]

    where_parts = []
    params = {}

    if start_week is not None:
        where_parts.append("yearweek >= :start_week")
        params["start_week"] = int(start_week)

    if end_week is not None:
        where_parts.append("yearweek <= :end_week")
        params["end_week"] = int(end_week)

    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    df = read_sql(
        f"SELECT * FROM {table} {where}",
        params if params else None,
        dbname="postgres",
    )

    if df.empty:
        raise RuntimeError(
            f"No source data found: "
            f"source={data_source}, table={table}, "
            f"start_week={start_week}, end_week={end_week}"
        )

    df = _safe_numeric_yearweek(df)
    df["data_source"] = data_source

    print(
        f"[DATA] source={data_source}, "
        f"rows={len(df)}, "
        f"min_yearweek={df['yearweek'].min()}, "
        f"max_yearweek={df['yearweek'].max()}"
    )

    return df


def load_weather_weekly(
    start_week: int | None = None,
    end_week: int | None = None,
) -> pd.DataFrame:
    """
    讀取週天氣資料。

    start_week=None 時，不加起始週限制。
    """
    where_parts = []
    params = {}

    if start_week is not None:
        where_parts.append("yearweek >= :start_week")
        params["start_week"] = int(start_week)

    if end_week is not None:
        where_parts.append("yearweek <= :end_week")
        params["end_week"] = int(end_week)

    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    df = read_sql(
        f"SELECT * FROM {settings.weather_table} {where}",
        params if params else None,
        dbname="postgres",
    )

    if df.empty:
        print(
            f"[WARN] weather table returned empty rows "
            f"from {start_week} to {end_week}"
        )
        return pd.DataFrame(columns=["yearweek", "county"])

    df = _safe_numeric_yearweek(df)

    print(
        f"[DATA] weather rows={len(df)}, "
        f"min_yearweek={df['yearweek'].min()}, "
        f"max_yearweek={df['yearweek'].max()}"
    )

    return df

def load_source_total_weekly(
    data_source: str,
    start_week: int | None = None,
    end_week: int | None = None,
) -> pd.DataFrame:
    """
    讀取各資料源每週縣市總就診人數。

    來源表：
        disease_forecast_data.model_source_total_weekly_county

    粒度：
        data_source × yearweek × county

    注意：
        這裡只讀 total_count，不直接作為同週 feature。
        後續只使用 lag / rolling features。
    """
    where_parts = ["data_source = :data_source"]
    params = {"data_source": data_source}

    if start_week is not None:
        where_parts.append("yearweek >= :start_week")
        params["start_week"] = int(start_week)

    if end_week is not None:
        where_parts.append("yearweek <= :end_week")
        params["end_week"] = int(end_week)

    where = "WHERE " + " AND ".join(where_parts)

    sql = f"""
        SELECT
            data_source,
            yearweek,
            county,
            total_count AS source_total_count
        FROM {settings.source_total_table}
        {where}
    """

    df = read_sql(sql, params, dbname="postgres")

    if df.empty:
        print(
            f"[WARN] source total table returned empty rows: "
            f"source={data_source}, start_week={start_week}, end_week={end_week}"
        )
        return pd.DataFrame(
            columns=[
                "data_source",
                "yearweek",
                "county",
                "source_total_count",
            ]
        )

    df = _safe_numeric_yearweek(df)
    df["data_source"] = df["data_source"].astype(str)
    df["county"] = df["county"].astype(str)
    df["source_total_count"] = pd.to_numeric(
        df["source_total_count"],
        errors="coerce",
    )

    df = df.drop_duplicates(
        subset=["data_source", "yearweek", "county"]
    )

    return df

def _find_wide_disease_columns(df: pd.DataFrame) -> dict[str, str]:
    found: dict[str, str] = {}
    lower_to_original = {c.lower(): c for c in df.columns}

    for lower_col, disease in WIDE_DISEASE_COUNT_COLUMNS.items():
        if lower_col in lower_to_original:
            found[lower_to_original[lower_col]] = disease

    return found


def _normalize_target(target: pd.DataFrame) -> pd.DataFrame:
    """
    將來源 model_*_weekly_county 統一成：

        yearweek
        county
        disease
        data_source
        count

    支援：
    1. 長表：yearweek, county, disease, count
    2. 寬表：yearweek, county, ev_total, ili_total, di_total
    """
    target_county = _pick_col(target, COUNTY_COL_CANDIDATES)
    disease_col = _pick_col(target, DISEASE_COL_CANDIDATES, required=False)
    count_col = _pick_col(target, COUNT_COL_CANDIDATES, required=False)

    if count_col is not None:
        out = target.rename(
            columns={
                target_county: "county",
                count_col: "count",
            }
        ).copy()

        if disease_col:
            out = out.rename(columns={disease_col: "disease"})
        else:
            out["disease"] = "ALL"

        out["county"] = out["county"].astype(str)
        out["disease"] = out["disease"].astype(str)
        out["count"] = pd.to_numeric(out["count"], errors="coerce")

        return out

    wide_cols = _find_wide_disease_columns(target)

    if not wide_cols:
        raise KeyError(
            "Cannot find target count column. "
            f"Expected long-table columns={COUNT_COL_CANDIDATES} "
            f"or wide disease columns={list(WIDE_DISEASE_COUNT_COLUMNS)}. "
            f"Existing columns: {list(target.columns)}"
        )

    keep_base_cols = ["yearweek", target_county]

    if "data_source" in target.columns:
        keep_base_cols.append("data_source")

    long_parts = []

    for count_column, disease_name in wide_cols.items():
        tmp = target[keep_base_cols + [count_column]].copy()
        tmp = tmp.rename(
            columns={
                target_county: "county",
                count_column: "count",
            }
        )
        tmp["disease"] = disease_name
        long_parts.append(tmp)

    out = pd.concat(long_parts, ignore_index=True)

    out["county"] = out["county"].astype(str)
    out["disease"] = out["disease"].astype(str)
    out["count"] = pd.to_numeric(out["count"], errors="coerce")

    return out


def _normalize_weather(weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return pd.DataFrame(columns=["yearweek", "county"])

    weather_county = _pick_col(weather, COUNTY_COL_CANDIDATES)
    weather = weather.rename(columns={weather_county: "county"}).copy()
    weather["county"] = weather["county"].astype(str)

    drop_cols = [c for c in weather.columns if c.lower() in DROP_COLS]
    weather = weather.drop(columns=drop_cols, errors="ignore")

    keep_cols = [c for c in weather.columns if c not in ["yearweek", "county"]]

    return (
        weather[["yearweek", "county"] + keep_cols]
        .drop_duplicates(subset=["yearweek", "county"])
    )


def build_feature_table(
    data_source: str,
    start_week: int | None,
    forecast_period: int,
    end_week: int | None = None,
) -> pd.DataFrame:
    """
    建立歷史訓練列與未來預測列。

    歷史列來自 model_*_weekly_county。
    未來列由歷史 county / disease 組合 cross forecast yearweek 產生。
    """
    future_weeks = forecast_yearweeks(forecast_period)
    max_needed_week = max(future_weeks + ([end_week] if end_week else []))

    target = _normalize_target(load_source_weekly(data_source, start_week, end_week))
    weather = _normalize_weather(load_weather_weekly(start_week, max_needed_week))
    source_total = load_source_total_weekly(
        data_source=data_source,
        start_week=start_week,
        end_week=max_needed_week,
    )
    dim = load_dim_features()

    historical = target[["yearweek", "county", "disease", "data_source", "count"]].copy()

    historical = historical.merge(
        source_total,
        on=["data_source", "yearweek", "county"],
        how="left",
    )

    historical["source_total_count"] = pd.to_numeric(
        historical["source_total_count"],
        errors="coerce",
    )

    historical["disease_rate"] = np.where(
        historical["source_total_count"].fillna(0) > 0,
        historical["count"] / historical["source_total_count"],
        np.nan,
    )

    historical["is_future"] = False

    combos = (
        historical[["county", "disease", "data_source"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    future = combos.merge(pd.DataFrame({"yearweek": future_weeks}), how="cross")

    future = future.merge(
        source_total,
        on=["data_source", "yearweek", "county"],
        how="left",
    )

    future["count"] = np.nan
    future["source_total_count"] = pd.to_numeric(
        future["source_total_count"],
        errors="coerce",
    )
    future["disease_rate"] = np.nan
    future["is_future"] = True

    future = future[
        [
            "yearweek",
            "county",
            "disease",
            "data_source",
            "count",
            "source_total_count",
            "disease_rate",
            "is_future",
        ]
    ]

    df = pd.concat([historical, future], ignore_index=True)

    df = (
        df.merge(weather, on=["yearweek", "county"], how="left")
        .merge(dim, on="yearweek", how="left")
        .sort_values(["data_source", "disease", "county", "yearweek", "is_future"])
        .reset_index(drop=True)
        
    )
    missing_total = (
        df[(df["is_future"] == False)]
        ["source_total_count"]
        .isna()
        .sum()
    )

    if missing_total > 0:
        print(
            f"[WARN] source_total_count missing rows: "
            f"source={data_source}, missing_rows={missing_total}"
        )
    return df


def get_exog_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "id",
        "yearweek",
        "county",
        "disease",
        "data_source",
        "count",
        "is_future",
        "source_total_count",
        "disease_rate",
        "created_at",
        "updated_at",
        "inserted_at",
        "modified_at",
    }

    cols = [c for c in df.columns if c not in excluded]

    return [
        c for c in cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]