from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from etl_disease_model_predict.config import settings
from etl_disease_model_predict.db.postgres import append_dataframe
from etl_disease_model_predict.pipeline.train_predict import run_all
from etl_disease_model_predict.pipeline.holdout_eval import run_holdout_all


RUN_MODE_CHOICES = ["smoke", "fast", "full", "forecast"]
FEATURE_SET_CHOICES = ["base", "medium", "full"]
FEATURE_SELECT_CHOICES = ["none", "filter", "lgbm_topk"]
COVID_POLICY_CHOICES = ["include", "exclude", "flag"]
TASK_CHOICES = ["forecast", "holdout"]


METRIC_OUTPUT_COLS = [
    "data_source",
    "model_method",
    "MAE",
    "RMSE",
    "MAPE",
    "sMAPE",
    "WAPE",
    "Bias",
    "y_true_sum",
    "y_pred_sum",
    "y_true_mean",
    "y_pred_mean",
    "disease",
    "county",
]


PREDICTION_OUTPUT_COLS = [
    "data_source",
    "model_method",
    "disease",
    "county",
    "yearweek",
    "actual_count",
    "forecast_count",
    "forecast_count_rounded",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run disease forecast or holdout validation pipeline."
    )

    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default="forecast",
        help="forecast=正式未來預測；holdout=最後N週回測驗證與近一年折線圖。",
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=["nhi_er", "nhi_opd", "rods"],
        choices=list(settings.source_tables.keys()),
    )

    parser.add_argument(
        "--forecast-period",
        type=int,
        default=settings.forecast_period,
        help="Forecast task 的未來預測週數。",
    )

    parser.add_argument(
        "--end-week",
        type=int,
        default=None,
        help="Holdout task 使用的最新實際年週，例如 202626。",
    )

    parser.add_argument(
        "--holdout-weeks",
        type=int,
        default=8,
        help="Holdout task 最後幾週作為驗證資料。",
    )

    parser.add_argument(
        "--plot-weeks",
        type=int,
        default=52,
        help="Holdout task 圖表顯示最近幾週實際趨勢。",
    )

    parser.add_argument(
        "--start-week",
        type=int,
        default=None,
        help="Training start yearweek. Default=None means use all available data.",
    )

    parser.add_argument(
        "--recent-weeks",
        type=int,
        default=None,
        help="Only use recent N weeks for training.",
    )

    parser.add_argument(
        "--run-mode",
        choices=RUN_MODE_CHOICES,
        default="fast",
    )

    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default=None,
    )

    parser.add_argument(
        "--feature-select",
        choices=FEATURE_SELECT_CHOICES,
        default=None,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--covid-policy",
        choices=COVID_POLICY_CHOICES,
        default="include",
        help="include=keep 2020-2022, exclude=remove 2020-2022 training rows, flag=keep with covid_period feature.",
    )

    parser.add_argument(
        "--enable-grid-search",
        action="store_true",
        help="Enable GridSearchCV for global panel models.",
    )

    parser.add_argument(
        "--grid-cv-splits",
        type=int,
        default=3,
        help="Inner time-series CV splits for GridSearchCV.",
    )

    parser.add_argument(
        "--meta-model",
        choices=[
            "ridge",
            "elasticnet",
            "lasso",
            "huber",
            "nonnegative_linear",
        ],
        default="ridge",
        help="正式產生 forecast 用的主要 meta model。",
    )

    parser.add_argument(
        "--compare-meta-models",
        nargs="+",
        choices=[
            "ridge",
            "elasticnet",
            "lasso",
            "huber",
            "nonnegative_linear",
        ],
        default=None,
        help=(
            "額外比較的 meta model 清單（不含 --meta-model 也沒關係，會自動併入比較）。"
            "base model 只會訓練一次，不會因為比較多個 meta model 而重複訓練，"
            "比較結果會出現在 leaderboard.csv / breakdown.csv，"
            "並額外輸出 meta_model_comparison.csv 方便直接比較。"
            "最終 forecast.csv 仍然只由 --meta-model 指定的方法（或 best base model）產生。"
        ),
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
    )

    parser.add_argument(
        "--enable-sarimax",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: outputs/forecast or outputs/holdout_{N}w.",
    )

    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Write raw evaluation metric table to PostgreSQL.",
    )

    return parser.parse_args()


def _is_valid_text(value) -> bool:
    if value is None:
        return False

    text = str(value)

    return text not in {"", "nan", "None", "NaN", "<NA>"}


_INVALID_TEXT_TOKENS = {"", "nan", "None", "NaN", "<NA>"}


def _clean_text_col(df: pd.DataFrame, col: str) -> pd.Series:
    """把欄位轉成字串並將無效值統一標記為 NA，供向量化字串組合使用。"""
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")

    s = df[col].astype("string")
    return s.where(s.notna() & ~s.isin(_INVALID_TEXT_TOKENS))


def _model_method_series(df: pd.DataFrame) -> pd.Series:
    """
    向量化版本的 model_method 組合邏輯。

    語意與逐列版本完全一致：
        model_task != default 且有 model_layer -> "{task}/{layer}/{name}"
        model_task != default 且無 model_layer -> "{task}/{name}"
        model_task == default 且有 model_layer -> "{layer}/{name}"
        其餘                                   -> "{name}"

    用向量化字串操作取代 df.apply(axis=1)，在列數較多時可大幅縮短輸出組裝時間。
    """
    model_task = _clean_text_col(df, "model_task")
    model_layer = _clean_text_col(df, "model_layer")
    model_name = _clean_text_col(df, "model_name")
    base_model = _clean_text_col(df, "base_model")

    name = model_name.fillna(base_model).fillna("unknown")

    has_task = model_task.notna() & (model_task != "default")
    has_layer = model_layer.notna()

    result = pd.Series(pd.NA, index=df.index, dtype="string")

    mask = has_task & has_layer
    result.loc[mask] = model_task[mask] + "/" + model_layer[mask] + "/" + name[mask]

    mask = has_task & ~has_layer
    result.loc[mask] = model_task[mask] + "/" + name[mask]

    mask = ~has_task & has_layer
    result.loc[mask] = model_layer[mask] + "/" + name[mask]

    mask = ~has_task & ~has_layer
    result.loc[mask] = name[mask]

    return result.astype(str)


def _compact_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=METRIC_OUTPUT_COLS)

    out = df.copy()
    out["model_method"] = _model_method_series(out)

    for col in ["disease", "county"]:
        if col not in out.columns:
            out[col] = "ALL"

        out[col] = out[col].fillna("ALL")

    keep_cols = [c for c in METRIC_OUTPUT_COLS if c in out.columns]

    return out[keep_cols]


def _compact_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PREDICTION_OUTPUT_COLS)

    out = df.copy()
    out["model_method"] = _model_method_series(out)

    for col in ["disease", "county"]:
        if col not in out.columns:
            out[col] = ""

        out[col] = out[col].fillna("")

    if "actual_count" not in out.columns:
        out["actual_count"] = pd.NA

    if "forecast_count_rounded" not in out.columns and "forecast_count" in out.columns:
        out["forecast_count_rounded"] = (
            pd.to_numeric(out["forecast_count"], errors="coerce")
            .round()
            .clip(lower=0)
            .astype("Int64")
        )

    keep_cols = [c for c in PREDICTION_OUTPUT_COLS if c in out.columns]

    return out[keep_cols]


def _make_leaderboard(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame(columns=METRIC_OUTPUT_COLS)

    if "metric_level" not in metric_df.columns:
        return _compact_metric_columns(metric_df)

    leaderboard = metric_df.loc[
        metric_df["metric_level"].eq("overall")
    ].copy()

    if leaderboard.empty:
        return pd.DataFrame(columns=METRIC_OUTPUT_COLS)

    sort_cols = [
        c for c in ["data_source", "model_task", "WAPE"]
        if c in leaderboard.columns
    ]

    if sort_cols:
        leaderboard = leaderboard.sort_values(sort_cols)

    return _compact_metric_columns(leaderboard)


def _make_breakdown(metric_df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    if metric_df.empty or "metric_level" not in metric_df.columns:
        return pd.DataFrame(columns=METRIC_OUTPUT_COLS)

    breakdown = metric_df.loc[
        metric_df["metric_level"].isin(
            ["by_disease", "by_county", "by_disease_county"]
        )
    ].copy()

    if breakdown.empty:
        return pd.DataFrame(columns=METRIC_OUTPUT_COLS)

    if "WAPE" in breakdown.columns:
        breakdown = breakdown.sort_values("WAPE", ascending=False)

    return _compact_metric_columns(breakdown).head(top_n)


def _make_meta_model_comparison(metric_df: pd.DataFrame) -> pd.DataFrame:
    """
    整理每個 data_source/model_task 底下，各個 meta model 的 stacking overall WAPE，
    並附上該任務 best base model 的 WAPE 作為比較基準（wape_vs_best_base < 0 代表
    這個 meta model 的 stacking 比目前最好的 base model 還準）。

    單一 meta model 執行時這張表只會有一列；搭配 --compare-meta-models 時，
    同一個 data_source/model_task 會出現多列，方便直接比較。
    """
    if metric_df.empty or "metric_level" not in metric_df.columns:
        return pd.DataFrame()

    overall = metric_df.loc[metric_df["metric_level"].eq("overall")].copy()

    if overall.empty:
        return pd.DataFrame()

    group_cols = [
        c for c in ["data_source", "model_task", "model_scope"]
        if c in overall.columns
    ]

    if not group_cols:
        return pd.DataFrame()

    base_rows = overall.loc[
        overall["model_layer"].eq("base") & overall["WAPE"].notna()
    ].copy()

    if base_rows.empty:
        base_best = pd.DataFrame(columns=group_cols + ["best_base_model", "best_base_wape"])
    else:
        base_best = (
            base_rows.sort_values("WAPE")
            .groupby(group_cols, dropna=False, as_index=False)
            .first()[group_cols + ["model_name", "WAPE"]]
            .rename(columns={"model_name": "best_base_model", "WAPE": "best_base_wape"})
        )

    stacking_rows = overall.loc[overall["model_layer"].eq("stacking")].copy()

    if stacking_rows.empty:
        return pd.DataFrame()

    keep_cols = [
        c for c in [
            "data_source", "model_task", "model_scope",
            "meta_model", "meta_eval_type", "model_name",
            "n_obs", "WAPE", "MAE", "RMSE", "Bias",
        ]
        if c in stacking_rows.columns
    ]

    comparison = stacking_rows[keep_cols].merge(base_best, on=group_cols, how="left")
    comparison["wape_vs_best_base"] = comparison["WAPE"] - comparison["best_base_wape"]
    comparison["stacking_wins"] = comparison["wape_vs_best_base"] < 0

    sort_cols = [c for c in group_cols + ["WAPE"] if c in comparison.columns]

    if sort_cols:
        comparison = comparison.sort_values(sort_cols)

    return comparison.reset_index(drop=True)


def _make_run_summary(
    args,
    metric_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
) -> pd.DataFrame:
    best_model = None
    best_wape = None

    if not leaderboard_df.empty and "WAPE" in leaderboard_df.columns:
        best = leaderboard_df.sort_values("WAPE", ascending=True).iloc[0]
        best_model = best.get("model_method")
        best_wape = best.get("WAPE")

    return pd.DataFrame(
        [
            {
                "task": args.task,
                "run_mode": args.run_mode,
                "sources": ",".join(args.sources),
                "forecast_period": args.forecast_period if args.task == "forecast" else pd.NA,
                "end_week": args.end_week if args.task == "holdout" else pd.NA,
                "holdout_weeks": args.holdout_weeks if args.task == "holdout" else pd.NA,
                "plot_weeks": args.plot_weeks if args.task == "holdout" else pd.NA,
                "start_week": args.start_week,
                "recent_weeks": args.recent_weeks,
                "feature_set": args.feature_set,
                "feature_select": args.feature_select,
                "top_k": args.top_k,
                "covid_policy": args.covid_policy,
                "enable_grid_search": args.enable_grid_search,
                "grid_cv_splits": args.grid_cv_splits,
                "meta_model": args.meta_model,
                "compare_meta_models": (
                    ",".join(args.compare_meta_models)
                    if getattr(args, "compare_meta_models", None)
                    else None
                ),
                "use_gpu": args.use_gpu,
                "enable_sarimax": args.enable_sarimax,
                "metric_rows": len(metric_df),
                "leaderboard_rows": len(leaderboard_df),
                "best_model_by_wape": best_model,
                "best_wape": best_wape,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    )


def _write_outputs(
    forecast_df: pd.DataFrame,
    base_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    args,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leaderboard_df = _make_leaderboard(metric_df)
    breakdown_df = _make_breakdown(metric_df)
    predictions_df = _compact_prediction_columns(forecast_df)
    base_predictions_df = _compact_prediction_columns(base_df)
    summary_df = _make_run_summary(args, metric_df, leaderboard_df)
    meta_comparison_df = _make_meta_model_comparison(metric_df)

    leaderboard_df.to_csv(
        output_dir / "leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    breakdown_df.to_csv(
        output_dir / "breakdown.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base_predictions_df.to_csv(
        output_dir / "base_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        output_dir / "run_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    meta_comparison_df.to_csv(
        output_dir / "meta_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        leaderboard_df,
        breakdown_df,
        predictions_df,
        base_predictions_df,
        summary_df,
        meta_comparison_df,
    )


def _print_console_summary(
    leaderboard_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    base_predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    meta_comparison_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\n========== MODEL SUMMARY ==========")

    if not summary_df.empty:
        row = summary_df.iloc[0]
        print(f"task              : {row.get('task')}")
        print(f"run_mode          : {row.get('run_mode')}")
        print(f"sources           : {row.get('sources')}")
        print(f"covid_policy      : {row.get('covid_policy')}")
        print(f"enable_grid_search: {row.get('enable_grid_search')}")
        print(f"meta_model        : {row.get('meta_model')}")
        print(f"compare_meta_models: {row.get('compare_meta_models')}")
        print(f"best_model        : {row.get('best_model_by_wape')}")
        print(f"best_wape         : {row.get('best_wape')}")
        print(f"output_dir        : {output_dir}")

    print("\n========== LEADERBOARD ==========")

    if leaderboard_df.empty:
        print("[WARN] leaderboard is empty")
    else:
        show_cols = [
            "data_source",
            "model_method",
            "MAE",
            "RMSE",
            "MAPE",
            "sMAPE",
            "WAPE",
            "Bias",
            "y_true_sum",
            "y_pred_sum",
            "y_true_mean",
            "y_pred_mean",
            "disease",
            "county",
        ]
        show_cols = [c for c in show_cols if c in leaderboard_df.columns]
        print(leaderboard_df[show_cols].to_string(index=False))

    print("\n========== WORST BREAKDOWN TOP 10 ==========")

    if breakdown_df.empty:
        print("[WARN] breakdown is empty")
    else:
        show_cols = [
            "data_source",
            "model_method",
            "MAE",
            "RMSE",
            "MAPE",
            "sMAPE",
            "WAPE",
            "Bias",
            "y_true_sum",
            "y_pred_sum",
            "y_true_mean",
            "y_pred_mean",
            "disease",
            "county",
        ]
        show_cols = [c for c in show_cols if c in breakdown_df.columns]
        print(breakdown_df[show_cols].head(10).to_string(index=False))

    print("\n========== META MODEL COMPARISON ==========")

    if meta_comparison_df.empty:
        print("[WARN] meta model comparison is empty")
    else:
        show_cols = [
            "data_source",
            "model_task",
            "meta_model",
            "meta_eval_type",
            "WAPE",
            "best_base_model",
            "best_base_wape",
            "wape_vs_best_base",
            "stacking_wins",
        ]
        show_cols = [c for c in show_cols if c in meta_comparison_df.columns]
        print(meta_comparison_df[show_cols].to_string(index=False))

    print("\nwritten files:")
    print(f"  {output_dir / 'leaderboard.csv'}")
    print(f"  {output_dir / 'breakdown.csv'}")
    print(f"  {output_dir / 'predictions.csv'}")
    print(f"  {output_dir / 'base_predictions.csv'}")
    print(f"  {output_dir / 'run_summary.csv'}")
    print(f"  {output_dir / 'meta_model_comparison.csv'}")

    if (output_dir / "plots").exists():
        print(f"  {output_dir / 'plots'}/*.png")

    print("===================================\n")


def _resolve_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.task == "holdout":
        return Path(
            f"outputs/holdout_{args.holdout_weeks}w"
            f"_{args.run_mode}_meta_{args.meta_model}_{timestamp}"
        )

    return Path(f"outputs/forecast_{args.run_mode}_meta_{args.meta_model}_{timestamp}")


def main():
    args = parse_args()

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "forecast":
        forecast_df, base_df, metric_df, _selected_df = run_all(
            data_sources=args.sources,
            forecast_period=args.forecast_period,
            start_week=args.start_week,
            recent_weeks=args.recent_weeks,
            run_mode=args.run_mode,
            feature_set=args.feature_set,
            feature_select=args.feature_select,
            top_k=args.top_k,
            covid_policy=args.covid_policy,
            enable_grid_search=args.enable_grid_search,
            grid_cv_splits=args.grid_cv_splits,
            use_gpu=args.use_gpu,
            enable_sarimax=args.enable_sarimax,
            compare_meta_models=args.compare_meta_models,
            meta_model=args.meta_model,
        )

    elif args.task == "holdout":
        if args.end_week is None:
            raise ValueError("--task holdout requires --end-week, e.g. --end-week 202626")

        forecast_df, base_df, metric_df, _selected_df = run_holdout_all(
            data_sources=args.sources,
            end_week=args.end_week,
            holdout_weeks=args.holdout_weeks,
            plot_weeks=args.plot_weeks,
            start_week=args.start_week,
            recent_weeks=args.recent_weeks,
            run_mode=args.run_mode,
            feature_set=args.feature_set,
            feature_select=args.feature_select,
            top_k=args.top_k,
            covid_policy=args.covid_policy,
            enable_grid_search=args.enable_grid_search,
            grid_cv_splits=args.grid_cv_splits,
            use_gpu=args.use_gpu,
            enable_sarimax=args.enable_sarimax,
            meta_model=args.meta_model,
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Unknown task={args.task}")

    (
        leaderboard_df,
        breakdown_df,
        predictions_df,
        base_predictions_df,
        summary_df,
        meta_comparison_df,
    ) = _write_outputs(
        forecast_df=forecast_df,
        base_df=base_df,
        metric_df=metric_df,
        args=args,
        output_dir=output_dir,
    )

    if args.write_db:
        append_dataframe(metric_df, settings.metric_table)

    _print_console_summary(
        leaderboard_df=leaderboard_df,
        breakdown_df=breakdown_df,
        predictions_df=predictions_df,
        base_predictions_df=base_predictions_df,
        summary_df=summary_df,
        meta_comparison_df=meta_comparison_df,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()