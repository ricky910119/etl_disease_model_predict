from __future__ import annotations

from datetime import date, timedelta

from etl_disease_model_predict.db.postgres import read_sql


def get_yearweek_range(start_date: date, end_date: date | None = None) -> list[int]:
    """依 DIM_DATA.public.dim_weekdate 取得 date range 對應 yearweek。"""
    end_date = end_date or start_date
    sql = """
        SELECT DISTINCT yearweek
        FROM public.dim_weekdate
        WHERE date BETWEEN :start_date AND :end_date
        ORDER BY yearweek
    """
    df = read_sql(
        sql,
        {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        dbname="DIM_DATA",
    )
    return df["yearweek"].astype(int).tolist()


def latest_closed_yearweek(base_date: date | None = None) -> int:
    """取得預測日往前 7 天所在的 yearweek，作為訓練截止週。"""
    base_date = base_date or date.today()
    weeks = get_yearweek_range(base_date - timedelta(days=7))
    if not weeks:
        raise RuntimeError("Cannot resolve latest closed yearweek from dim_weekdate")
    return int(weeks[0])


def forecast_yearweeks(forecast_period: int, base_date: date | None = None) -> list[int]:
    """取得從 base_date 開始往後 forecast_period 週的 yearweek。"""
    base_date = base_date or date.today()
    weeks = get_yearweek_range(base_date, base_date + timedelta(days=7 * (forecast_period - 1)))
    if len(weeks) < forecast_period:
        raise RuntimeError(f"dim_weekdate returned only {len(weeks)} forecast weeks, expected {forecast_period}")
    return weeks[:forecast_period]
