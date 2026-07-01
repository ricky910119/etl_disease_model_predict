from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from etl_disease_model_predict.features.dataset import build_feature_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check weekly disease counts by data source and disease."
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=["nhi_er", "nhi_opd", "rods"],
        help="Data sources to check. Example: --sources nhi_er nhi_opd rods",
    )

    parser.add_argument(
        "--start-week",
        type=int,
        default=None,
        help="Optional start yearweek. Example: --start-week 200501",
    )

    parser.add_argument(
        "--forecast-period",
        type=int,
        default=4,
        help="Only used to satisfy build_feature_table. Future rows will be excluded.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/data_check",
        help="Output directory.",
    )

    return parser.parse_args()


def require_columns(df: pd.DataFrame, source: str):
    required = {
        "data_source",
        "disease",
        "county",
        "yearweek",
        "count",
        "is_future",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"source={source} missing required columns: {sorted(missing)}"
        )


def load_source(source: str, start_week: int | None, forecast_period: int) -> pd.DataFrame:
    print(f"[LOAD] source={source}")

    df = build_feature_table(
        data_source=source,
        start_week=start_week,
        forecast_period=forecast_period,
    )

    require_columns(df, source)

    df = df.copy()

    df = df[df["is_future"] == False].copy()
    df["yearweek"] = pd.to_numeric(df["yearweek"], errors="coerce").astype("Int64")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    df = df[df["yearweek"].notna()].copy()
    df["yearweek"] = df["yearweek"].astype(int)

    return df


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for source in args.sources:
        df = load_source(
            source=source,
            start_week=args.start_week,
            forecast_period=args.forecast_period,
        )

        print(
            f"[INFO] source={source}, "
            f"rows={len(df)}, "
            f"diseases={sorted(df['disease'].dropna().astype(str).unique().tolist())}, "
            f"min_week={df['yearweek'].min()}, "
            f"max_week={df['yearweek'].max()}, "
            f"count_sum={df['count'].sum()}"
        )

        frames.append(df)

    if not frames:
        raise RuntimeError("No data loaded.")

    data = pd.concat(frames, ignore_index=True)

    weekly_source_disease = (
        data
        .groupby(["data_source", "disease", "yearweek"], as_index=False, dropna=False)
        .agg(
            weekly_count=("count", "sum"),
            county_count=("county", "nunique"),
            row_count=("count", "size"),
        )
        .sort_values(["data_source", "disease", "yearweek"])
        .reset_index(drop=True)
    )

    weekly_source_disease_county = (
        data
        .groupby(
            ["data_source", "disease", "county", "yearweek"],
            as_index=False,
            dropna=False,
        )
        .agg(
            weekly_count=("count", "sum"),
            row_count=("count", "size"),
        )
        .sort_values(["data_source", "disease", "county", "yearweek"])
        .reset_index(drop=True)
    )

    summary = (
        weekly_source_disease
        .groupby(["data_source", "disease"], as_index=False, dropna=False)
        .agg(
            min_week=("yearweek", "min"),
            max_week=("yearweek", "max"),
            n_weeks=("yearweek", "nunique"),
            total_count=("weekly_count", "sum"),
            mean_weekly_count=("weekly_count", "mean"),
            median_weekly_count=("weekly_count", "median"),
            min_weekly_count=("weekly_count", "min"),
            max_weekly_count=("weekly_count", "max"),
            zero_week_count=("weekly_count", lambda s: int((s == 0).sum())),
        )
        .sort_values(["data_source", "disease"])
        .reset_index(drop=True)
    )

    latest_20_weeks = (
        weekly_source_disease
        .sort_values(["data_source", "disease", "yearweek"])
        .groupby(["data_source", "disease"], as_index=False, group_keys=False)
        .tail(20)
        .reset_index(drop=True)
    )

    weekly_source_disease_path = output_dir / "weekly_count_by_source_disease.csv"
    weekly_source_disease_county_path = output_dir / "weekly_count_by_source_disease_county.csv"
    summary_path = output_dir / "weekly_count_summary_by_source_disease.csv"
    latest_20_path = output_dir / "latest_20_weeks_by_source_disease.csv"

    weekly_source_disease.to_csv(weekly_source_disease_path, index=False, encoding="utf-8-sig")
    weekly_source_disease_county.to_csv(
        weekly_source_disease_county_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    latest_20_weeks.to_csv(latest_20_path, index=False, encoding="utf-8-sig")

    print("\n========== SUMMARY ==========")
    print(summary.to_string(index=False))

    print("\n========== LATEST 20 WEEKS ==========")
    print(latest_20_weeks.to_string(index=False))

    print("\n========== OUTPUT FILES ==========")
    print(weekly_source_disease_path)
    print(weekly_source_disease_county_path)
    print(summary_path)
    print(latest_20_path)


if __name__ == "__main__":
    main()
