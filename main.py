from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from etl_disease_model_predict.config import settings
from etl_disease_model_predict.db.postgres import append_dataframe
from etl_disease_model_predict.pipeline.train_predict import run_all


RUN_MODE_CHOICES = ["smoke", "fast", "full", "forecast"]
FEATURE_SET_CHOICES = ["base", "medium", "full"]
FEATURE_SELECT_CHOICES = ["none", "filter", "lgbm_topk"]
COVID_POLICY_CHOICES = ["include", "exclude", "flag"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run disease model evaluation pipeline."
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
        choices=["ridge", "elasticnet"],
        default="ridge",
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
        "--write-db",
        action="store_true",
        help="Write evaluation metric table to PostgreSQL.",
    )

    return parser.parse_args()


def _make_leaderboard(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame()

    if "metric_level" not in metric_df.columns:
        return metric_df.copy()

    leaderboard = metric_df.loc[
        metric_df["metric_level"].eq("overall")
    ].copy()

    if leaderboard.empty:
        return pd.DataFrame()

    sort_cols = []
    ascending = []

    for col in ["data_source", "model_task", "model_layer", "WAPE"]:
        if col in leaderboard.columns:
            sort_cols.append(col)
            ascending.append(True)

    if sort_cols:
        leaderboard = leaderboard.sort_values(sort_cols, ascending=ascending)

    rank_group_cols = [
        c for c in ["data_source", "model_task"]
        if c in leaderboard.columns
    ]

    if "WAPE" in leaderboard.columns:
        if rank_group_cols:
            leaderboard["rank_wape"] = (
                leaderboard
                .groupby(rank_group_cols)["WAPE"]
                .rank(method="dense", ascending=True)
                .astype(int)
            )
        else:
            leaderboard["rank_wape"] = (
                leaderboard["WAPE"]
                .rank(method="dense", ascending=True)
                .astype(int)
            )

    keep_cols = [
        "data_source",
        "model_task",
        "model_scope",
        "is_rods_ev_national",
        "is_offshore_collapsed",
        "weather_removed",
        "run_mode",
        "feature_set",
        "feature_select",
        "top_k",
        "covid_policy",
        "enable_grid_search",
        "model_layer",
        "model_name",
        "metric_level",
        "n_obs",
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
        "rank_wape",
        "created_at",
    ]

    keep_cols = [c for c in keep_cols if c in leaderboard.columns]

    return leaderboard[keep_cols]


def _make_breakdown(metric_df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    if metric_df.empty or "metric_level" not in metric_df.columns:
        return pd.DataFrame()

    breakdown = metric_df.loc[
        metric_df["metric_level"].isin(
            ["by_disease", "by_county", "by_disease_county"]
        )
    ].copy()

    if breakdown.empty:
        return pd.DataFrame()

    if "WAPE" in breakdown.columns:
        breakdown = breakdown.sort_values("WAPE", ascending=False)

    keep_cols = [
        "data_source",
        "model_task",
        "model_scope",
        "is_rods_ev_national",
        "is_offshore_collapsed",
        "weather_removed",
        "run_mode",
        "feature_set",
        "feature_select",
        "top_k",
        "covid_policy",
        "enable_grid_search",
        "model_layer",
        "model_name",
        "metric_level",
        "disease",
        "county",
        "n_obs",
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
        "created_at",
    ]

    keep_cols = [c for c in keep_cols if c in breakdown.columns]

    return breakdown[keep_cols].head(top_n)


def _make_used_features(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()

    if "feature" not in selected_df.columns:
        return pd.DataFrame()

    out = selected_df.copy()

    if "feature_type" not in out.columns:
        out["feature_type"] = "numeric"

    if "data_source" not in out.columns:
        out["data_source"] = "unknown"

    if "model_task" not in out.columns:
        out["model_task"] = "unknown"

    if "selected" in out.columns:
        selected_mask = (
            out["selected"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        out = out.loc[selected_mask].copy()

    if out.empty:
        return pd.DataFrame()

    sort_cols = [
        c for c in [
            "data_source",
            "model_task",
            "feature_type",
            "importance",
            "feature",
        ]
        if c in out.columns
    ]

    if "importance" in out.columns:
        out["importance"] = pd.to_numeric(out["importance"], errors="coerce")
        out = out.sort_values(
            sort_cols,
            ascending=[
                True if c != "importance" else False
                for c in sort_cols
            ],
        )
    else:
        out = out.sort_values(
            [c for c in ["data_source", "model_task", "feature"] if c in out.columns]
        )

    rank_group_cols = [
        c for c in ["data_source", "model_task"]
        if c in out.columns
    ]

    if rank_group_cols:
        out["feature_rank"] = (
            out.groupby(rank_group_cols)
            .cumcount()
            .add(1)
        )
    else:
        out["feature_rank"] = out.reset_index().index + 1

    keep_cols = [
        "data_source",
        "model_task",
        "model_scope",
        "is_rods_ev_national",
        "is_offshore_collapsed",
        "weather_removed",
        "run_mode",
        "feature_set",
        "feature_select",
        "top_k",
        "covid_policy",
        "enable_grid_search",
        "feature_rank",
        "feature",
        "feature_type",
        "selected",
        "selection_stage",
        "importance",
        "missing_rate",
        "zero_rate",
        "n_unique",
        "dtype",
        "created_at",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]

    return out[keep_cols]

def _make_forecast_results(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame()

    out = forecast_df.copy()

    keep_cols = [
        "data_source",
        "model_task",
        "model_scope",
        "is_rods_ev_national",
        "is_offshore_collapsed",
        "weather_removed",
        "disease",
        "county",
        "yearweek",
        "forecast_count",
        "model_name",
        "run_mode",
        "feature_set",
        "feature_select",
        "top_k",
        "covid_policy",
        "enable_grid_search",
        "created_at",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]

    sort_cols = [
        c for c in ["data_source", "model_task", "disease", "county", "yearweek"]
        if c in out.columns
    ]

    if sort_cols:
        out = out.sort_values(sort_cols)

    return out[keep_cols]


def _make_base_forecast_results(base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame()

    out = base_df.copy()

    keep_cols = [
        "data_source",
        "model_task",
        "model_scope",
        "is_rods_ev_national",
        "is_offshore_collapsed",
        "weather_removed",
        "disease",
        "county",
        "yearweek",
        "base_model",
        "forecast_count",
        "run_mode",
        "feature_set",
        "feature_select",
        "top_k",
        "covid_policy",
        "enable_grid_search",
        "created_at",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]

    sort_cols = [
        c for c in [
            "data_source",
            "model_task",
            "base_model",
            "disease",
            "county",
            "yearweek",
        ]
        if c in out.columns
    ]

    if sort_cols:
        out = out.sort_values(sort_cols)

    return out[keep_cols]

def _make_run_summary(
    args,
    metric_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
) -> pd.DataFrame:
    best_model = None
    best_wape = None

    if not leaderboard_df.empty and "WAPE" in leaderboard_df.columns:
        best = leaderboard_df.sort_values("WAPE", ascending=True).iloc[0]
        best_model = best.get("model_name")
        best_wape = best.get("WAPE")

    return pd.DataFrame(
        [
            {
                "run_mode": args.run_mode,
                "sources": ",".join(args.sources),
                "forecast_period": args.forecast_period,
                "start_week": args.start_week,
                "recent_weeks": args.recent_weeks,
                "feature_set": args.feature_set,
                "feature_select": args.feature_select,
                "top_k": args.top_k,
                "covid_policy": args.covid_policy,
                "enable_grid_search": args.enable_grid_search,
                "grid_cv_splits": args.grid_cv_splits,
                "meta_model": args.meta_model,
                "use_gpu": args.use_gpu,
                "enable_sarimax": args.enable_sarimax,
                "metric_rows": len(metric_df),
                "leaderboard_rows": len(leaderboard_df),
                "best_model_by_wape": best_model,
                "best_wape": best_wape,
            }
        ]
    )


def _write_outputs(
    forecast_df: pd.DataFrame,
    base_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    args,
    output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    leaderboard_df = _make_leaderboard(metric_df)
    breakdown_df = _make_breakdown(metric_df)
    used_features_df = _make_used_features(selected_df)
    forecast_results_df = _make_forecast_results(forecast_df)
    base_forecast_results_df = _make_base_forecast_results(base_df)
    summary_df = _make_run_summary(args, metric_df, leaderboard_df)

    leaderboard_df.to_csv(
        output_dir / "model_eval_leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    breakdown_df.to_csv(
        output_dir / "model_eval_breakdown.csv",
        index=False,
        encoding="utf-8-sig",
    )

    used_features_df.to_csv(
        output_dir / "model_used_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    forecast_results_df.to_csv(
        output_dir / "forecast_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base_forecast_results_df.to_csv(
        output_dir / "base_forecast_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        output_dir / "run_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        leaderboard_df,
        breakdown_df,
        used_features_df,
        forecast_results_df,
        base_forecast_results_df,
        summary_df,
    )


def _print_console_summary(
    leaderboard_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    used_features_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    print("\n========== MODEL EVALUATION SUMMARY ==========")

    if not summary_df.empty:
        row = summary_df.iloc[0]
        print(f"run_mode          : {row.get('run_mode')}")
        print(f"sources           : {row.get('sources')}")
        print(f"forecast_period   : {row.get('forecast_period')}")
        print(f"covid_policy      : {row.get('covid_policy')}")
        print(f"enable_grid_search: {row.get('enable_grid_search')}")
        print(f"best_model        : {row.get('best_model_by_wape')}")
        print(f"best_wape         : {row.get('best_wape')}")

    print("\n========== LEADERBOARD ==========")

    if leaderboard_df.empty:
        print("[WARN] leaderboard is empty")
    else:
        show_cols = [
            "data_source",
            "model_task",
            "model_scope",
            "model_layer",
            "model_name",
            "n_obs",
            "MAE",
            "RMSE",
            "WAPE",
            "Bias",
            "rank_wape",
        ]
        show_cols = [c for c in show_cols if c in leaderboard_df.columns]
        print(leaderboard_df[show_cols].to_string(index=False))

    print("\n========== WORST BREAKDOWN TOP 10 ==========")

    if breakdown_df.empty:
        print("[WARN] breakdown is empty")
    else:
        show_cols = [
            "data_source",
            "model_task",
            "model_name",
            "metric_level",
            "disease",
            "county",
            "n_obs",
            "WAPE",
            "Bias",
        ]
        show_cols = [c for c in show_cols if c in breakdown_df.columns]
        print(breakdown_df[show_cols].head(10).to_string(index=False))

    print("\n========== USED FEATURES TOP 30 ==========")

    if used_features_df.empty:
        print("[WARN] used features is empty")
    else:
        show_cols = [
            "data_source",
            "model_task",
            "feature_rank",
            "feature",
            "feature_type",
            "selection_stage",
            "importance",
        ]
        show_cols = [c for c in show_cols if c in used_features_df.columns]
        print(used_features_df[show_cols].head(30).to_string(index=False))

    print("\nwritten files:")
    print("  outputs/model_eval_leaderboard.csv")
    print("  outputs/model_eval_breakdown.csv")
    print("  outputs/model_used_features.csv")
    print("  outputs/forecast_results.csv")
    print("  outputs/base_forecast_results.csv")
    print("  outputs/run_summary.csv")
    print("============================================\n")


def main():
    args = parse_args()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_df, base_df, metric_df, selected_df = run_all(
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
        meta_model=args.meta_model,
    )

    (
        leaderboard_df,
        breakdown_df,
        used_features_df,
        forecast_results_df,
        base_forecast_results_df,
        summary_df,
    ) = _write_outputs(
        forecast_df=forecast_df,
        base_df=base_df,
        metric_df=metric_df,
        selected_df=selected_df,
        args=args,
        output_dir=output_dir,
    )

    if args.write_db:
        append_dataframe(metric_df, settings.metric_table)

    _print_console_summary(
        leaderboard_df=leaderboard_df,
        breakdown_df=breakdown_df,
        used_features_df=used_features_df,
        summary_df=summary_df,
    )


if __name__ == "__main__":
    main()