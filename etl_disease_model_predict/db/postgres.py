from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from eic_utils import conn


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_PARAM_RE = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")


def _validate_table_name(table_name: str) -> tuple[str, str]:
    """檢查 schema.table 格式，避免動態 SQL 表名被注入。"""
    if not _IDENTIFIER_RE.match(table_name):
        raise ValueError(f"Invalid table_name: {table_name}. Expected format: schema.table")
    return table_name.split(".", 1)


def _convert_named_params(sql: str) -> str:
    """將 :name 參數格式轉成 psycopg2/eic_utils 可接受的 %(name)s。"""
    return _PARAM_RE.sub(r"%(\1)s", sql)


def _python_value(value: Any) -> Any:
    """將 pandas/numpy 型別轉成 psycopg2 較穩定可寫入的 Python 型別。"""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (datetime, date)):
        return value
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {col: _python_value(row[col]) for col in df.columns}
        for _, row in df.iterrows()
    ]


@conn.deco.postgres(dbname="postgres")
def _read_sql_postgres(sql: str, params: dict[str, Any] | None = None, cur=None) -> pd.DataFrame:
    sql = _convert_named_params(sql)
    cur.execute(sql, params or {})
    rows = cur.fetchall()
    columns = [c[0] for c in cur.description]
    return pd.DataFrame.from_records(rows, columns=columns)


@conn.deco.postgres(dbname="DIM_DATA")
def _read_sql_dim_data(sql: str, params: dict[str, Any] | None = None, cur=None) -> pd.DataFrame:
    sql = _convert_named_params(sql)
    cur.execute(sql, params or {})
    rows = cur.fetchall()
    columns = [c[0] for c in cur.description]
    return pd.DataFrame.from_records(rows, columns=columns)


def read_sql(
    sql: str,
    params: dict[str, Any] | None = None,
    dbname: str = "postgres",
) -> pd.DataFrame:
    """讀取 PostgreSQL/DIM_DATA。

    Parameters
    ----------
    sql:
        SQL 字串，可使用 :name 參數格式。
    params:
        SQL 參數。
    dbname:
        預設讀 postgres；維度資料請傳 DIM_DATA。
    """
    if dbname.upper() == "DIM_DATA":
        return _read_sql_dim_data(sql, params)
    return _read_sql_postgres(sql, params)


@conn.deco.postgres(dbname="postgres")
def execute_sql(sql: str, params: dict[str, Any] | None = None, cur=None) -> None:
    """執行不回傳資料的 SQL。"""
    sql = _convert_named_params(sql)
    cur.execute(sql, params or {})


@conn.deco.postgres(dbname="postgres")
def append_dataframe(df: pd.DataFrame, table_name: str, cur=None) -> None:
    """將 DataFrame append 到 PostgreSQL 指定 schema.table。"""
    if df is None or df.empty:
        print(f"[DB] skip append empty dataframe: {table_name}")
        return

    schema, table = _validate_table_name(table_name)
    columns = list(df.columns)
    col_sql = ", ".join([f'"{c}"' for c in columns])
    placeholder = ", ".join([f"%({c})s" for c in columns])
    sql = f'INSERT INTO "{schema}"."{table}" ({col_sql}) VALUES ({placeholder})'

    cur.executemany(sql, _records(df))
    print(f"[DB] appended {len(df)} rows into {schema}.{table}")


@conn.deco.postgres(dbname="postgres")
def replace_partition(
    df: pd.DataFrame,
    table_name: str,
    keys: dict[str, Any],
    cur=None,
) -> None:
    """先依 keys 刪除同一批資料，再 append 新資料。"""
    if not keys:
        raise ValueError("replace_partition requires non-empty keys")

    schema, table = _validate_table_name(table_name)
    where_sql = " AND ".join([f'"{k}" = %({k})s' for k in keys])
    delete_sql = f'DELETE FROM "{schema}"."{table}" WHERE {where_sql}'
    cur.execute(delete_sql, {k: _python_value(v) for k, v in keys.items()})

    if df is None or df.empty:
        print(f"[DB] partition deleted, no rows to append: {schema}.{table}")
        return

    columns = list(df.columns)
    col_sql = ", ".join([f'"{c}"' for c in columns])
    placeholder = ", ".join([f"%({c})s" for c in columns])
    insert_sql = f'INSERT INTO "{schema}"."{table}" ({col_sql}) VALUES ({placeholder})'
    cur.executemany(insert_sql, _records(df))
    print(f"[DB] replaced partition and appended {len(df)} rows into {schema}.{table}")
