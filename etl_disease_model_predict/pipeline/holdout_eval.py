from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from etl_disease_model_predict.features.dataset import (
    build_feature_table,
    get_exog_columns,
)

from etl_disease_model_predict.pipeline.train_predict import (
    KEY_COLS,
    _resolve_config,
    _expand_model_tasks,
    _prepare_feature_for_model_task,
    _apply_covid_policy,
    _apply_recent_weeks,
    add_lag_rolling_features,
    _recursive_future_features,
    _feature_columns,
    _select_features,
    _fit_predict_global_oof,
    _fit_predict_local_oof,
    _fit_final_base_predictions,
    _metric_row,
    _add_directional_hit,
    resolve_task_metadata,
    resolve_allowed_registry,
)
from etl_disease_model_predict.modeling.combiner import (
    fit_combiner,
    predict_combiner,
    combiner_layer,
    combiner_model_name,
    ALL_META_METHODS,
)
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run last-N-weeks holdout validation and plot recent trend."
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=["nhi_er", "nhi_opd", "rods"],
        choices=["nhi_er", "nhi_opd", "rods"],
    )

    parser.add_argument(
        "--end-week",
        type=int,
        default=202626,
        help="Latest actual yearweek included in the dataset.",
    )

    parser.add_argument(
        "--holdout-weeks",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--plot-weeks",
        type=int,
        default=52,
    )

    parser.add_argument(
        "--start-week",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--recent-weeks",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run-mode",
        choices=["smoke", "fast", "full", "forecast"],
        default="full",
    )

    parser.add_argument(
        "--feature-set",
        choices=["base", "medium", "full"],
        default="full",
    )

    parser.add_argument(
        "--feature-select",
        choices=["none", "filter", "lgbm_topk"],
        default="lgbm_topk",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--covid-policy",
        choices=["include", "exclude", "flag"],
        default="exclude",
    )

    parser.add_argument(
        "--enable-grid-search",
        action="store_true",
    )

    parser.add_argument(
        "--grid-cv-splits",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--meta-model",
        choices=sorted(ALL_META_METHODS),
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
        "--output-dir",
        type=str,
        default="outputs/holdout_8w",
    )

    return parser.parse_args()


def setup_chinese_font() -> None:
    """
    設定圖表字型，讓中文（縣市名、標題等）能正確顯示。

    分兩層，第一層不依賴伺服器上「剛好有沒有裝」某個字型：
        1. 掃描專案內建字型目錄 assets/fonts/ 底下的 .ttf/.otf/.ttc，
           用 font_manager.addfont() 直接註冊給 matplotlib，
           不需要 sudo、不需要系統字型安裝，換到哪台 server 都一樣可用。
        2. 系統原本就有安裝的中文字型（依名稱比對）作為備援，
           萬一 assets/fonts/ 是空的，至少還有機會用到系統字型。

    若兩層都找不到任何支援中文的字型，會印出清楚的 [WARN]，
    避免中文顯示成缺字方框卻完全沒有任何提示。
    """
    bundled_families: list[str] = []

    if FONT_DIR.exists():
        for font_path in sorted(FONT_DIR.glob("*")):
            if font_path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
                continue

            try:
                font_manager.fontManager.addfont(str(font_path))
                family = font_manager.FontProperties(fname=str(font_path)).get_name()
                bundled_families.append(family)
            except Exception as exc:
                print(
                    f"[WARN] 無法載入字型檔 {font_path}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    fallback_system_families = [
        "Noto Sans CJK TC",
        "Noto Sans TC",
        "AR PL UKai TW",
        "AR PL UMing TW",
        "TW-Kai",
        "BiauKai",
        "DFKai-SB",
        "Microsoft JhengHei",
        "PingFang TC",
    ]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = (
        bundled_families + fallback_system_families + ["Times New Roman"]
    )
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

    available = {f.name for f in font_manager.fontManager.ttflist}
    resolved = [f for f in plt.rcParams["font.sans-serif"] if f in available]

    if not bundled_families and not any(
        f in available for f in fallback_system_families
    ):
        print(
            f"[WARN] 找不到任何中文字型（{FONT_DIR} 是空的，系統上也沒有裝對應名稱的字型），"
            "圖表中的中文（縣市名等）會顯示成缺字方框。"
            f"請把 .ttf/.otf/.ttc 字型檔放進 {FONT_DIR} 底下再重新執行。",
            flush=True,
        )
    else:
        print(f"[FONT] 實際套用字型優先順序: {resolved}", flush=True)


def _get_holdout_weeks(
    hist: pd.DataFrame,
    end_week: int,
    holdout_weeks: int,
) -> list[int]:
    weeks = (
        hist.loc[hist["yearweek"].astype(int) <= int(end_week), "yearweek"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(weeks) < holdout_weeks + 52:
        print(
            f"[WARN] available weeks={len(weeks)} may be short for "
            f"holdout_weeks={holdout_weeks}",
            flush=True,
        )

    if len(weeks) <= holdout_weeks:
        raise RuntimeError(
            f"Not enough actual weeks. available={len(weeks)}, "
            f"holdout_weeks={holdout_weeks}"
        )

    return weeks[-holdout_weeks:]


def _make_holdout_features(
    base_train: pd.DataFrame,
    holdout_raw: pd.DataFrame,
    holdout_weeks: int,
    feature_set: str,
) -> pd.DataFrame:
    parts = []

    for keys, hist in base_train.groupby(KEY_COLS, dropna=False):
        mask = (
            (holdout_raw["data_source"] == keys[0])
            & (holdout_raw["disease"] == keys[1])
            & (holdout_raw["county"] == keys[2])
        )

        fg_actual = (
            holdout_raw.loc[mask]
            .sort_values("yearweek")
            .head(holdout_weeks)
            .copy()
        )

        if len(fg_actual) != holdout_weeks:
            print(
                f"[WARN] incomplete holdout rows keys={keys}, "
                f"rows={len(fg_actual)}, expected={holdout_weeks}",
                flush=True,
            )
            continue

        actual_map = fg_actual[KEY_COLS + ["yearweek", "count"]].copy()
        actual_map = actual_map.rename(columns={"count": "actual_count"})

        fg_exog = fg_actual.copy()
        fg_exog["count"] = np.nan

        if "disease_rate" in fg_exog.columns:
            fg_exog["disease_rate"] = np.nan

        tmp = _recursive_future_features(
            history=hist,
            future_exog=fg_exog,
            forecast_period=holdout_weeks,
            feature_set=feature_set,
        )

        tmp = tmp.merge(
            actual_map,
            on=KEY_COLS + ["yearweek"],
            how="left",
        )

        parts.append(tmp)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def _build_holdout_metric_df(
    holdout_pred: pd.DataFrame,
    base_pred: pd.DataFrame,
    data_source: str,
    meta_model: str,
    context_base: dict,
) -> pd.DataFrame:
    """
    建立 holdout 的 metric_df。

    除了原本的 overall／by_disease／by_county／by_disease_county 四種聚合層級，
    這裡額外加上 by_lead_week：holdout 是從同一個預測起點往後遞迴展開多週，
    lead_week 代表「這是預測起點之後的第幾週」，可以直接看出準確度是否隨著
    預測距離拉遠而下降（呼應論文裡 nowcast 到 3-week forecast 誤差遞增的分析）。

    每個 metric row 也會同時算出 Pearson correlation 與 HitRate（逐期漲跌方向
    命中率），HitRate 需要先用 _add_directional_hit() 在最細粒度算好方向命中欄位，
    這裡才能不管怎麼分組聚合都得到正確結果。
    """
    rows = []

    eval_cols = KEY_COLS + ["yearweek", "lead_week", "actual_count"]

    for model_name in base_pred.columns:
        tmp = holdout_pred[eval_cols].copy()
        tmp["y_pred"] = base_pred[model_name].values
        tmp = tmp[tmp["y_pred"].notna()].copy()

        if tmp.empty:
            continue

        tmp = _add_directional_hit(
            tmp,
            key_cols=KEY_COLS,
            y_true_col="actual_count",
            y_pred_col="y_pred",
        )

        context = context_base.copy()
        context["data_source"] = data_source
        context["model_layer"] = "base"
        context["model_name"] = model_name
        context["prediction_type"] = "holdout"

        c = context.copy()
        c["metric_level"] = "overall"
        rows.append(_metric_row(tmp["actual_count"], tmp["y_pred"], c, hit=tmp["_hit"]))

        for disease, g in tmp.groupby("disease", dropna=False):
            c = context.copy()
            c["metric_level"] = "by_disease"
            c["disease"] = disease
            rows.append(_metric_row(g["actual_count"], g["y_pred"], c, hit=g["_hit"]))

        for county, g in tmp.groupby("county", dropna=False):
            c = context.copy()
            c["metric_level"] = "by_county"
            c["county"] = county
            rows.append(_metric_row(g["actual_count"], g["y_pred"], c, hit=g["_hit"]))

        for (disease, county), g in tmp.groupby(["disease", "county"], dropna=False):
            c = context.copy()
            c["metric_level"] = "by_disease_county"
            c["disease"] = disease
            c["county"] = county
            rows.append(_metric_row(g["actual_count"], g["y_pred"], c, hit=g["_hit"]))

        for lead_week, g in tmp.groupby("lead_week", dropna=False):
            c = context.copy()
            c["metric_level"] = "by_lead_week"
            c["lead_week"] = lead_week
            rows.append(_metric_row(g["actual_count"], g["y_pred"], c, hit=g["_hit"]))

    stack_tmp = holdout_pred.copy()
    stack_tmp = _add_directional_hit(
        stack_tmp,
        key_cols=KEY_COLS,
        y_true_col="actual_count",
        y_pred_col="forecast_count",
    )

    context = context_base.copy()
    context["data_source"] = data_source
    context["model_layer"] = combiner_layer(meta_model)
    context["model_name"] = combiner_model_name(meta_model)
    context["prediction_type"] = "holdout"

    c = context.copy()
    c["metric_level"] = "overall"
    rows.append(
        _metric_row(stack_tmp["actual_count"], stack_tmp["forecast_count"], c, hit=stack_tmp["_hit"])
    )

    for disease, g in stack_tmp.groupby("disease", dropna=False):
        c = context.copy()
        c["metric_level"] = "by_disease"
        c["disease"] = disease
        rows.append(_metric_row(g["actual_count"], g["forecast_count"], c, hit=g["_hit"]))

    for county, g in stack_tmp.groupby("county", dropna=False):
        c = context.copy()
        c["metric_level"] = "by_county"
        c["county"] = county
        rows.append(_metric_row(g["actual_count"], g["forecast_count"], c, hit=g["_hit"]))

    for (disease, county), g in stack_tmp.groupby(["disease", "county"], dropna=False):
        c = context.copy()
        c["metric_level"] = "by_disease_county"
        c["disease"] = disease
        c["county"] = county
        rows.append(_metric_row(g["actual_count"], g["forecast_count"], c, hit=g["_hit"]))

    for lead_week, g in stack_tmp.groupby("lead_week", dropna=False):
        c = context.copy()
        c["metric_level"] = "by_lead_week"
        c["lead_week"] = lead_week
        rows.append(_metric_row(g["actual_count"], g["forecast_count"], c, hit=g["_hit"]))

    return pd.DataFrame(rows)


def _plot_group(
    actual_recent: pd.DataFrame,
    pred_group: pd.DataFrame,
    plot_dir: Path,
    data_source: str,
    model_task: str,
    disease: str,
    county: str,
    holdout_start_week: int,
) -> None:
    actual_recent = actual_recent.sort_values("yearweek").copy()
    pred_group = pred_group.sort_values("yearweek").copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    # 使用冷色調莫蘭迪色系
    color_actual = "#5C7A95"  # 冷灰藍
    color_pred = "#6A8A82"    # 冷杉綠
    color_vline = "#A9B2C3"   # 淺灰藍 (輔助線)

    ax.plot(
        actual_recent["yearweek"].astype(str),
        actual_recent["count"],
        marker="o",
        color=color_actual,
        linewidth=1.8,
        label="實際值",
    )

    ax.plot(
        pred_group["yearweek"].astype(str),
        pred_group["forecast_count"],
        marker="o",
        color=color_pred,
        linewidth=2.0,
        linestyle="--",
        label="holdout預測",
    )

    x_labels = actual_recent["yearweek"].astype(str).tolist()

    if str(holdout_start_week) in x_labels:
        ax.axvline(
            x=x_labels.index(str(holdout_start_week)),
            color=color_vline,
            linestyle=":",
            linewidth=1.8,
            label="holdout起點",
        )

    title = f"{data_source} / {model_task} / {disease} / {county} 最近一年趨勢與8週holdout"
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("yearweek", fontsize=12)
    ax.set_ylabel("count", fontsize=12)
    
    # 學術發表圖表細節優化
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333")
    
    # 柔和的網格線
    ax.grid(True, alpha=0.4, color="#D3D9DF", linestyle="--")
    ax.legend(frameon=False, fontsize=11)

    step = max(1, len(x_labels) // 12)
    ax.set_xticks(range(0, len(x_labels), step))
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45)

    fig.tight_layout()

    safe_name = (
        f"{data_source}_{model_task}_{disease}_{county}"
        .replace("/", "_")
        .replace(" ", "_")
    )

    fig.savefig(plot_dir / f"{safe_name}.png", dpi=300) # 提升解析度至學術常用的 300 dpi
    plt.close(fig)


def run_task(args, task: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_source = task["data_source"]
    model_task = task["model_task"]

    cfg = _resolve_config(
        run_mode=args.run_mode,
        recent_weeks=args.recent_weeks,
        feature_set=args.feature_set,
        feature_select=args.feature_select,
        top_k=args.top_k,
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(
        f"[HOLDOUT] source={data_source}, model_task={model_task}, "
        f"end_week={args.end_week}, holdout_weeks={args.holdout_weeks}, "
        f"feature_set={cfg.feature_set}, feature_select={cfg.feature_select}, "
        f"top_k={cfg.top_k}, grid={args.enable_grid_search}",
        flush=True,
    )

    feature = build_feature_table(
        data_source=data_source,
        start_week=args.start_week,
        forecast_period=args.holdout_weeks,
    )

    feature, removed_weather_cols = _prepare_feature_for_model_task(
        feature=feature,
        data_source=data_source,
        model_task=model_task,
    )

    meta_info = {
        "model_task": model_task,
        **resolve_task_metadata(model_task),
        "weather_removed": len(removed_weather_cols) > 0,
    }

    feature = _apply_covid_policy(
        df=feature,
        covid_policy=args.covid_policy,
    )

    exog_cols = get_exog_columns(feature)

    hist = feature[
        (feature["is_future"] == False)
        & feature["count"].notna()
        & (feature["yearweek"].astype(int) <= int(args.end_week))
    ].copy()

    if hist.empty:
        raise RuntimeError(f"No historical rows for source={data_source}, task={model_task}")

    holdout_weeks = _get_holdout_weeks(
        hist=hist,
        end_week=args.end_week,
        holdout_weeks=args.holdout_weeks,
    )

    holdout_start_week = int(holdout_weeks[0])
    # lead_week：holdout 期間內每個 yearweek 對應第幾個 horizon（1 = 第一週）。
    # holdout_weeks 已經是依時間排序的清單，直接用位置對應即可。
    week_to_lead = {int(w): i + 1 for i, w in enumerate(holdout_weeks)}
    base_train = hist[
        ~hist["yearweek"].astype(int).isin(holdout_weeks)
    ].copy()

    base_train = _apply_recent_weeks(base_train, cfg.recent_weeks)

    holdout_raw = hist[
        hist["yearweek"].astype(int).isin(holdout_weeks)
    ].copy()

    train = (
        add_lag_rolling_features(base_train, feature_set=cfg.feature_set)
        .dropna(subset=["lag_1", "count"])
        .reset_index(drop=True)
    )

    if train.empty:
        raise RuntimeError(
            f"Training data became empty after lag generation: "
            f"source={data_source}, task={model_task}"
        )

    holdout_feature = _make_holdout_features(
        base_train=base_train,
        holdout_raw=holdout_raw,
        holdout_weeks=args.holdout_weeks,
        feature_set=cfg.feature_set,
    )

    if holdout_feature.empty:
        raise RuntimeError(
            f"Holdout feature is empty: source={data_source}, task={model_task}"
        )

    _, numeric_cols_all, categorical_cols = _feature_columns(train, exog_cols)

    selected_numeric_cols, selected_report = _select_features(
        train=train,
        numeric_cols=numeric_cols_all,
        feature_select=cfg.feature_select,
        top_k=cfg.top_k,
    )

    feature_cols = categorical_cols + selected_numeric_cols

    registry = resolve_allowed_registry(
        numeric_cols=selected_numeric_cols,
        categorical_cols=categorical_cols,
        model_task=model_task,
        cfg_model_names=cfg.model_names,
        use_gpu=args.use_gpu,
        enable_sarimax=args.enable_sarimax,
    )

    print(
        f"[MODEL_POOL] source={data_source}, "
        f"model_task={model_task}, "
        f"models={[spec.name for spec in registry]}",
        flush=True,
    )

    global_oof = _fit_predict_global_oof(
        train=train,
        feature_cols=feature_cols,
        y_col="count",
        registry=registry,
        n_splits=cfg.n_splits,
        min_train_weeks=cfg.min_train_weeks,
        enable_grid_search=args.enable_grid_search,
        grid_cv_splits=args.grid_cv_splits,
    )

    local_oof = _fit_predict_local_oof(
        train=train,
        exog_cols=selected_numeric_cols,
        registry=registry,
        n_splits=cfg.n_splits,
        min_train_weeks=cfg.min_train_weeks,
    )

    oof = pd.concat([global_oof, local_oof], axis=1)
    oof = oof.dropna(axis=1, how="all")

    valid_mask = oof.notna().all(axis=1)

    if valid_mask.sum() == 0:
        raise RuntimeError(f"OOF prediction is empty: source={data_source}, task={model_task}")

    if oof.shape[1] < 2:
        raise RuntimeError(
            f"Need at least 2 base models for stacking, got {oof.shape[1]}"
        )

    meta = fit_combiner(
        oof.loc[valid_mask],
        train.loc[valid_mask, "count"],
        method=args.meta_model,
    )

    holdout_base = _fit_final_base_predictions(
        train=train,
        future=holdout_feature,
        feature_cols=feature_cols,
        exog_cols=selected_numeric_cols,
        registry=registry,
        forecast_period=args.holdout_weeks,
        enable_grid_search=args.enable_grid_search,
        grid_cv_splits=args.grid_cv_splits,
    )

    holdout_base = holdout_base.reindex(columns=oof.columns)

    if holdout_base.isna().any().any():
        holdout_base = holdout_base.fillna(holdout_base.median(numeric_only=True))
        holdout_base = holdout_base.fillna(0)

    final_pred = predict_combiner(meta, holdout_base)

    holdout_pred = holdout_feature[KEY_COLS + ["yearweek", "actual_count"]].copy()
    holdout_pred["lead_week"] = holdout_pred["yearweek"].astype(int).map(week_to_lead)
    holdout_pred["forecast_count"] = final_pred
    holdout_pred["forecast_count_rounded"] = (
        np.clip(np.rint(holdout_pred["forecast_count"]), 0, None)
        .astype(int)
    )

    for key, value in meta_info.items():
        holdout_pred[key] = value

    holdout_pred["run_mode"] = args.run_mode
    holdout_pred["feature_set"] = cfg.feature_set
    holdout_pred["feature_select"] = cfg.feature_select
    holdout_pred["top_k"] = cfg.top_k
    holdout_pred["covid_policy"] = args.covid_policy
    holdout_pred["enable_grid_search"] = args.enable_grid_search
    holdout_pred["model_name"] = combiner_model_name(args.meta_model)
    holdout_pred["created_at"] = now

    base_rows = []

    for model_name in holdout_base.columns:
        tmp = holdout_feature[KEY_COLS + ["yearweek", "actual_count"]].copy()
        tmp["lead_week"] = tmp["yearweek"].astype(int).map(week_to_lead)
        tmp["base_model"] = model_name
        tmp["forecast_count"] = holdout_base[model_name].values
        tmp["forecast_count_rounded"] = (
            np.clip(np.rint(tmp["forecast_count"].fillna(0)), 0, None)
            .astype(int)
        )

        for key, value in meta_info.items():
            tmp[key] = value

        tmp["run_mode"] = args.run_mode
        tmp["feature_set"] = cfg.feature_set
        tmp["feature_select"] = cfg.feature_select
        tmp["top_k"] = cfg.top_k
        tmp["covid_policy"] = args.covid_policy
        tmp["enable_grid_search"] = args.enable_grid_search
        tmp["created_at"] = now
        base_rows.append(tmp)

    base_pred = (
        pd.concat(base_rows, ignore_index=True)
        if base_rows
        else pd.DataFrame()
    )

    context_base = {
        **meta_info,
        "disease": "ALL",
        "county": "ALL",
        "lead_week": np.nan,
        "meta_model": args.meta_model,
        "meta_eval_type": "holdout_backtest",
        "run_mode": args.run_mode,
        "feature_set": cfg.feature_set,
        "feature_select": cfg.feature_select,
        "top_k": cfg.top_k,
        "covid_policy": args.covid_policy,
        "enable_grid_search": args.enable_grid_search,
        "holdout_weeks": args.holdout_weeks,
        "holdout_start_week": holdout_start_week,
        "holdout_end_week": int(holdout_weeks[-1]),
        "created_at": now,
    }

    metric_df = _build_holdout_metric_df(
        holdout_pred=holdout_pred,
        base_pred=holdout_base,
        data_source=data_source,
        meta_model=args.meta_model,
        context_base=context_base,
    )

    selected_report = selected_report.copy()

    for key, value in meta_info.items():
        selected_report[key] = value

    selected_report["data_source"] = data_source
    selected_report["run_mode"] = args.run_mode
    selected_report["feature_set"] = cfg.feature_set
    selected_report["feature_select"] = cfg.feature_select
    selected_report["top_k"] = cfg.top_k
    selected_report["covid_policy"] = args.covid_policy
    selected_report["enable_grid_search"] = args.enable_grid_search
    selected_report["created_at"] = now

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    recent_weeks = (
        hist["yearweek"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )[-args.plot_weeks:]

    actual_recent = hist[
        hist["yearweek"].astype(int).isin(recent_weeks)
    ].copy()

    for keys, pred_group in holdout_pred.groupby(KEY_COLS, dropna=False):
        actual_group = actual_recent[
            (actual_recent["data_source"] == keys[0])
            & (actual_recent["disease"] == keys[1])
            & (actual_recent["county"] == keys[2])
        ].copy()

        if actual_group.empty:
            continue

        _plot_group(
            actual_recent=actual_group,
            pred_group=pred_group,
            plot_dir=plot_dir,
            data_source=str(keys[0]),
            model_task=model_task,
            disease=str(keys[1]),
            county=str(keys[2]),
            holdout_start_week=holdout_start_week,
        )

    return holdout_pred, base_pred, metric_df, selected_report


def main():
    args = parse_args()
    setup_chinese_font()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _expand_model_tasks(args.sources)

    all_holdout = []
    all_base = []
    all_metrics = []
    all_selected = []

    for task in tasks:
        h, b, m, s = run_task(args, task, output_dir)

        all_holdout.append(h)
        all_base.append(b)
        all_metrics.append(m)
        all_selected.append(s)

    holdout_df = pd.concat(all_holdout, ignore_index=True)
    base_df = pd.concat(all_base, ignore_index=True)
    metric_df = pd.concat(all_metrics, ignore_index=True)
    selected_df = pd.concat(all_selected, ignore_index=True)

    holdout_df = holdout_df.sort_values(
        ["data_source", "model_task", "disease", "county", "yearweek"]
    )

    base_df = base_df.sort_values(
        ["data_source", "model_task", "base_model", "disease", "county", "yearweek"]
    )

    metric_df = metric_df.sort_values(
        [
            "data_source",
            "model_task",
            "model_layer",
            "model_name",
            "metric_level",
            "WAPE",
        ],
        ascending=[True, True, True, True, True, True],
    )

    selected_df = selected_df.sort_values(
        [
            c for c in [
                "data_source",
                "model_task",
                "feature_type",
                "importance",
                "feature",
            ]
            if c in selected_df.columns
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
        ][:len([
            c for c in [
                "data_source",
                "model_task",
                "feature_type",
                "importance",
                "feature",
            ]
            if c in selected_df.columns
        ])],
    )

    holdout_df.to_csv(
        output_dir / "holdout_8w_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base_df.to_csv(
        output_dir / "holdout_8w_base_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_df.to_csv(
        output_dir / "holdout_8w_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_df.to_csv(
        output_dir / "holdout_8w_used_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        metric_df[
            (metric_df["metric_level"] == "overall")
        ]
        .sort_values(["data_source", "model_task", "WAPE"])
        .copy()
    )

    summary.to_csv(
        output_dir / "holdout_8w_leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== HOLDOUT 8W FINISHED ==========")
    print(f"output_dir: {output_dir}")
    print("files:")
    print("  holdout_8w_predictions.csv")
    print("  holdout_8w_base_predictions.csv")
    print("  holdout_8w_metrics.csv")
    print("  holdout_8w_used_features.csv")
    print("  holdout_8w_leaderboard.csv")
    print("  plots/*.png")
    print("\nleaderboard:")
    show_cols = [
        c for c in [
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
        ]
        if c in summary.columns
    ]
    print(summary[show_cols].to_string(index=False))
    print("=========================================\n")
from types import SimpleNamespace


def run_holdout_all(
    data_sources: list[str],
    end_week: int,
    holdout_weeks: int,
    plot_weeks: int,
    start_week: int | None,
    recent_weeks: int | None,
    run_mode: str,
    feature_set: str | None,
    feature_select: str | None,
    top_k: int | None,
    covid_policy: str,
    enable_grid_search: bool,
    grid_cv_splits: int,
    use_gpu: bool = False,
    enable_sarimax: bool = False,
    meta_model: str = "ridge",
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    給 main.py 呼叫的 holdout 統一入口。

    回傳格式與 train_predict.run_all() 一致：
        forecast_df, base_df, metric_df, selected_df
    """
    if output_dir is None:
        output_dir = Path(f"outputs/holdout_{holdout_weeks}w")

    output_dir.mkdir(parents=True, exist_ok=True)

    setup_chinese_font()

    args = SimpleNamespace(
        sources=data_sources,
        end_week=end_week,
        holdout_weeks=holdout_weeks,
        plot_weeks=plot_weeks,
        start_week=start_week,
        recent_weeks=recent_weeks,
        run_mode=run_mode,
        feature_set=feature_set,
        feature_select=feature_select,
        top_k=top_k,
        covid_policy=covid_policy,
        enable_grid_search=enable_grid_search,
        grid_cv_splits=grid_cv_splits,
        meta_model=meta_model,
        use_gpu=use_gpu,
        enable_sarimax=enable_sarimax,
        output_dir=str(output_dir),
    )

    tasks = _expand_model_tasks(data_sources)

    all_holdout = []
    all_base = []
    all_metrics = []
    all_selected = []

    for task in tasks:
        h, b, m, s = run_task(
            args=args,
            task=task,
            output_dir=output_dir,
        )

        all_holdout.append(h)
        all_base.append(b)
        all_metrics.append(m)
        all_selected.append(s)

    forecast_df = (
        pd.concat(all_holdout, ignore_index=True)
        if all_holdout
        else pd.DataFrame()
    )

    base_df = (
        pd.concat(all_base, ignore_index=True)
        if all_base
        else pd.DataFrame()
    )

    metric_df = (
        pd.concat(all_metrics, ignore_index=True)
        if all_metrics
        else pd.DataFrame()
    )

    selected_df = (
        pd.concat(all_selected, ignore_index=True)
        if all_selected
        else pd.DataFrame()
    )

    if not forecast_df.empty:
        forecast_df = forecast_df.sort_values(
            ["data_source", "model_task", "disease", "county", "yearweek"]
        )

    if not base_df.empty:
        base_df = base_df.sort_values(
            ["data_source", "model_task", "base_model", "disease", "county", "yearweek"]
        )

    if not metric_df.empty and "WAPE" in metric_df.columns:
        sort_cols = [
            c for c in [
                "data_source",
                "model_task",
                "model_layer",
                "model_name",
                "metric_level",
                "WAPE",
            ]
            if c in metric_df.columns
        ]
        metric_df = metric_df.sort_values(sort_cols)

    return forecast_df, base_df, metric_df, selected_df

if __name__ == "__main__":
    main()