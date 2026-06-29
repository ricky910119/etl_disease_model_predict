from __future__ import annotations

import argparse
from pathlib import Path

from etl_disease_model_predict.config import settings
from etl_disease_model_predict.db.postgres import append_dataframe
from etl_disease_model_predict.pipeline.train_predict import run_all


def parse_args():
    parser = argparse.ArgumentParser(description="Run disease weekly county model prediction pipeline.")
    parser.add_argument("--sources", nargs="+", default=["nhi_er", "nhi_opd", "rods"], choices=list(settings.source_tables.keys()))
    parser.add_argument("--forecast-period", type=int, default=settings.forecast_period)
    parser.add_argument("--start-week", type=int, default=settings.train_start_yearweek)
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU parameters for XGBoost/LightGBM/CatBoost when installed and available.")
    parser.add_argument("--meta-model", choices=["ridge", "elasticnet"], default="ridge")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_df, base_df, metric_df = run_all(
        args.sources,
        args.forecast_period,
        args.start_week,
        use_gpu=args.use_gpu,
        meta_model=args.meta_model,
    )

    forecast_df.to_csv(output_dir / "forecast_result.csv", index=False, encoding="utf-8-sig")
    base_df.to_csv(output_dir / "base_model_prediction.csv", index=False, encoding="utf-8-sig")
    metric_df.to_csv(output_dir / "model_metric.csv", index=False, encoding="utf-8-sig")

    if args.write_db:
        append_dataframe(forecast_df, settings.forecast_table)
        append_dataframe(base_df, settings.base_prediction_table)
        append_dataframe(metric_df, settings.metric_table)

    print(f"forecast rows={len(forecast_df)}")
    print(f"base prediction rows={len(base_df)}")
    print(f"metric rows={len(metric_df)}")


if __name__ == "__main__":
    main()
