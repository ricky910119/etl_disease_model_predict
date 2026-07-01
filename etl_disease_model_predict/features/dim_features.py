from __future__ import annotations

import pandas as pd

from etl_disease_model_predict.db.postgres import read_sql

def _normalize_location_name(value) -> str:
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .replace("臺", "台")
        .replace("　", "")
        .replace(" ", "")
    )


def load_dim_location_mapping() -> pd.DataFrame:
    """
    讀取 DIM_DATA.public.dim_location。

    用 old_name / new_name 建立縣市名稱對照，並取得 branch_t。
    目的：
        1. 處理 2005-2026 期間縣市名稱變動
        2. 將 EV 建模粒度由 county 轉為 branch_t
    """
    df = read_sql(
        """
        SELECT DISTINCT
            old_name,
            new_name,
            branch_t
        FROM public.dim_location
        WHERE branch_t IS NOT NULL
        """,
        dbname="DIM_DATA",
    )

    if df.empty:
        raise RuntimeError("DIM_DATA.public.dim_location returned empty rows")

    required_cols = {"old_name", "new_name", "branch_t"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise RuntimeError(
            f"dim_location missing columns: {sorted(missing_cols)}"
        )

    rows = []

    for _, row in df.iterrows():
        old_name = row.get("old_name")
        new_name = row.get("new_name")
        branch_t = row.get("branch_t")

        if pd.notna(old_name) and str(old_name).strip():
            rows.append(
                {
                    "county_key": _normalize_location_name(old_name),
                    "new_name": str(new_name).strip(),
                    "branch_t": str(branch_t).strip(),
                }
            )

        if pd.notna(new_name) and str(new_name).strip():
            rows.append(
                {
                    "county_key": _normalize_location_name(new_name),
                    "new_name": str(new_name).strip(),
                    "branch_t": str(branch_t).strip(),
                }
            )

    mapping = pd.DataFrame(rows).drop_duplicates()

    duplicated = (
        mapping.groupby("county_key")["branch_t"]
        .nunique()
        .reset_index(name="n_branch")
    )

    duplicated = duplicated[duplicated["n_branch"] > 1]

    if not duplicated.empty:
        raise RuntimeError(
            "dim_location has duplicated county_key mapping to multiple branch_t: "
            f"{duplicated['county_key'].tolist()}"
        )

    mapping = (
        mapping.sort_values(["branch_t", "new_name", "county_key"])
        .drop_duplicates(subset=["county_key"])
        .reset_index(drop=True)
    )

    return mapping

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Cannot find any column in {candidates}. Existing columns: {list(df.columns)}")


def load_dim_features() -> pd.DataFrame:
    """載入週維度特徵。

    對齊舊版 DimData.Data() 的邏輯：leave 轉 cnt、voc、eve、ev_period、di_period、covid。
    維度表在 DIM_DATA database。
    """
    holiday = read_sql("SELECT * FROM public.dim_new_year_holiday", dbname="DIM_DATA")
    weekdate = read_sql("SELECT * FROM public.dim_weekdate", dbname="DIM_DATA")

    date_col_holiday = _pick_col(holiday, ["DATE", "date"])
    name_col = _pick_col(holiday, ["NAME", "name"])
    leave_col = _pick_col(holiday, ["LEAVE", "leave"])
    ly_col = _pick_col(holiday, ["LY", "ly"])
    date_col_week = _pick_col(weekdate, ["date", "DATE"])
    yearweek_col = _pick_col(weekdate, ["yearweek", "YEARWEEK"])

    holiday = holiday.rename(columns={date_col_holiday: "holiday_date", name_col: "NAME", leave_col: "LEAVE", ly_col: "LY"})
    weekdate = weekdate.rename(columns={date_col_week: "date", yearweek_col: "yearweek"})

    holiday["holiday_date"] = pd.to_datetime(holiday["holiday_date"])
    weekdate["date"] = pd.to_datetime(weekdate["date"])
    weekdate["yearweek"] = weekdate["yearweek"].astype(int)

    df = holiday.merge(weekdate, left_on="holiday_date", right_on="date", how="left")
    df = df[df["yearweek"].notna()].copy()
    df["yearweek"] = df["yearweek"].astype(int)

    df = df.assign(
        ev_period=df["date"].apply(lambda x: 1 if x.month in [3, 4, 5, 6, 9] else 0),
        di_period=df["NAME"].apply(lambda x: 1 if x in ["農曆除夕", "春節", "中秋節", "國慶日"] else 0),
        voc=df["yearweek"].apply(lambda x: 1 if (int(x) - round(int(x), -2)) in [5, 6, 7, 28, 29, 30, 31, 32, 33, 34, 35, 36] else 0),
        leave=df["LEAVE"].apply(lambda x: 1 if x == "是" else 0),
        eve=df["LY"].apply(lambda x: 1 if x in ["小年夜", "除夕", "初一", "初二", "初三", "初四", "初五"] else 0),
    )

    out = df.groupby("yearweek", as_index=False).agg(
        leave_sum=("leave", "sum"),
        voc=("voc", "max"),
        eve=("eve", "sum"),
        ev_period=("ev_period", "max"),
        di_period=("di_period", "max"),
    )
    out["cnt"] = out["leave_sum"].apply(lambda x: 7 - int(x))
    out = out.drop(columns=["leave_sum"])
    out["covid"] = out["yearweek"].apply(lambda x: 1 if str(int(x))[:4] in ["2020", "2021", "2022"] else 0)
    return out[["yearweek", "cnt", "voc", "eve", "ev_period", "di_period", "covid"]].sort_values("yearweek").reset_index(drop=True)

