from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from eic_utils import conn


_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$"
)

_NAMED_PARAM_RE = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")


def _validate_table_name(table_name: str) -> tuple[str, str]:
    """
    檢查 schema.table 格式，避免動態 SQL 表名被注入。
    """
    if not _IDENTIFIER_RE.match(table_name):
        raise ValueError(
            f"Invalid table_name: {table_name}. Expected format: schema.table"
        )

    return table_name.split(".", 1)


def _python_value(value: Any) -> Any:
    """
    將 pandas / numpy 型別轉成 pyodbc 較穩定可接受的 Python 型別。
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, (datetime, date)):
        return value

    return value


def _to_qmark_sql(
    sql: str,
    params: dict[str, Any] | None = None,
) -> tuple[str, list[Any]]:
    """
    將 SQL 裡面的 :name 參數轉成 pyodbc 可接受的 ?。

    Example
    -------
    SQL:
        SELECT * FROM table WHERE yearweek >= :start_week

    params:
        {"start_week": 201401}

    轉換後:
        SELECT * FROM table WHERE yearweek >= ?
        [201401]
    """
    if not params:
        return sql, []

    ordered_values: list[Any] = []

    def repl(match: re.Match) -> str:
        key = match.group(1)

        if key not in params:
            raise KeyError(f"Missing SQL parameter: {key}")

        ordered_values.append(_python_value(params[key]))
        return "?"

    converted_sql = _NAMED_PARAM_RE.sub(repl, sql)

    return converted_sql, ordered_values


def _records_as_tuples(df: pd.DataFrame) -> list[tuple[Any, ...]]:
    """
    將 DataFrame 轉成 pyodbc executemany 可用的 tuple records。
    """
    records: list[tuple[Any, ...]] = []

    for _, row in df.iterrows():
        records.append(tuple(_python_value(row[col]) for col in df.columns))

    return records


@conn.deco.postgres(dbname="postgres")
def _read_sql_postgres(
    sql: str,
    params: dict[str, Any] | None = None,
    cur=None,
) -> pd.DataFrame:
    """
    讀取 postgres 資料庫。

    注意：
    目前 eic_utils.conn.deco.postgres 回傳的是 pyodbc cursor，
    所以即使 dbname='postgres'，SQL 參數也必須使用 ?。
    """
    sql, ordered_values = _to_qmark_sql(sql, params)

    if ordered_values:
        cur.execute(sql, *ordered_values)
    else:
        cur.execute(sql)

    rows = cur.fetchall()
    columns = [c[0] for c in cur.description]

    return pd.DataFrame.from_records(rows, columns=columns)


@conn.deco.postgres(dbname="DIM_DATA")
def _read_sql_dim_data(
    sql: str,
    params: dict[str, Any] | None = None,
    cur=None,
) -> pd.DataFrame:
    """
    讀取 DIM_DATA 資料庫。
    """
    sql, ordered_values = _to_qmark_sql(sql, params)

    if ordered_values:
        cur.execute(sql, *ordered_values)
    else:
        cur.execute(sql)

    rows = cur.fetchall()
    columns = [c[0] for c in cur.description]

    return pd.DataFrame.from_records(rows, columns=columns)


def read_sql(
    sql: str,
    params: dict[str, Any] | None = None,
    dbname: str = "postgres",
) -> pd.DataFrame:
    """
    專案統一讀取 SQL 函式。

    Parameters
    ----------
    sql:
        SQL 字串。參數請使用 :name 寫法。
        例如：
            SELECT * FROM table WHERE yearweek >= :start_week

    params:
        SQL 參數 dict。
        例如：
            {"start_week": 201401}

    dbname:
        postgres 或 DIM_DATA。
    """
    if dbname.upper() == "DIM_DATA":
        return _read_sql_dim_data(sql, params)

    return _read_sql_postgres(sql, params)


@conn.deco.postgres(dbname="postgres")
def execute_sql(
    sql: str,
    params: dict[str, Any] | None = None,
    cur=None,
) -> None:
    """
    執行不回傳資料的 SQL。
    """
    sql, ordered_values = _to_qmark_sql(sql, params)

    if ordered_values:
        cur.execute(sql, *ordered_values)
    else:
        cur.execute(sql)


@conn.deco.postgres(dbname="postgres")
def append_dataframe(
    df: pd.DataFrame,
    table_name: str,
    cur=None,
) -> None:
    """
    將 DataFrame append 到 PostgreSQL 指定 schema.table。

    table_name 格式：
        disease_forecast_data.xxx
    """
    if df is None or df.empty:
        print(f"[DB] skip append empty dataframe: {table_name}")
        return

    schema, table = _validate_table_name(table_name)

    columns = list(df.columns)
    col_sql = ", ".join([f'"{c}"' for c in columns])
    placeholder_sql = ", ".join(["?" for _ in columns])

    sql = (
        f'INSERT INTO "{schema}"."{table}" '
        f"({col_sql}) VALUES ({placeholder_sql})"
    )

    cur.executemany(sql, _records_as_tuples(df))

    print(f"[DB] appended {len(df)} rows into {schema}.{table}")


@conn.deco.postgres(dbname="postgres")
def replace_partition(
    df: pd.DataFrame,
    table_name: str,
    keys: dict[str, Any],
    cur=None,
) -> None:
    """
    先依 keys 刪除同一批資料，再 append 新資料。
    """
    if not keys:
        raise ValueError("replace_partition requires non-empty keys")

    schema, table = _validate_table_name(table_name)

    where_sql = " AND ".join([f'"{k}" = ?' for k in keys])
    delete_sql = f'DELETE FROM "{schema}"."{table}" WHERE {where_sql}'

    key_values = [_python_value(v) for v in keys.values()]
    cur.execute(delete_sql, *key_values)

    if df is None or df.empty:
        print(f"[DB] partition deleted, no rows to append: {schema}.{table}")
        return

    columns = list(df.columns)
    col_sql = ", ".join([f'"{c}"' for c in columns])
    placeholder_sql = ", ".join(["?" for _ in columns])

    insert_sql = (
        f'INSERT INTO "{schema}"."{table}" '
        f"({col_sql}) VALUES ({placeholder_sql})"
    )

    cur.executemany(insert_sql, _records_as_tuples(df))

    print(
        f"[DB] replaced partition and appended {len(df)} rows into {schema}.{table}"
    )