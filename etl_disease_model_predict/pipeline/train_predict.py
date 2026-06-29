from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from etl_disease_model_predict.features.dataset import build_feature_table, get_exog_columns
from etl_disease_model_predict.modeling.base_models import (
    build_base_registry,
    fit_sarimax,
    new_model,
    predict_sarimax,
)
from etl_disease_model_predict.modeling.stacking import fit_meta_model, predict_meta, regression_metrics
from etl_disease_model_predict.utils.week import forecast_yearweeks, latest_closed_yearweek

KEY_COLS = ["data_source", "disease", "county"]
CATEGORICAL_COLS = ["county", "disease", "data_source"]


def add_lag_rolling_features(
    df: pd.DataFrame,
    lags: int = 8,
    rolling_windows: tuple[int, ...] = (4, 8, 13),
) -> pd.DataFrame:
    out = df.sort_values(KEY_COLS + ["yearweek"]).copy()
    grp = out.groupby(KEY_COLS, dropna=False)["count"]

    for lag in range(1, lags + 1):
        out[f"lag_{lag}"] = grp.shift(lag)

    for window in rolling_windows:
        out[f"roll{window}_mean"] = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        out[f"roll{window}_std"] = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())

    return out


def _feature_columns(df: pd.DataFrame, exog_cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    lag_cols = [c for c in df.columns if c.startswith("lag_") or (c.startswith("roll") and (c.endswith("_mean") or c.endswith("_std")))]
    numeric_cols = list(dict.fromkeys(lag_cols + [c for c in exog_cols if c in df.columns]))
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    feature_cols = categorical_cols + numeric_cols
    return feature_cols, numeric_cols, categorical_cols


def _rolling_splits(df: pd.DataFrame, n_splits: int = 5, min_train_weeks: int = 104) -> list[tuple[np.ndarray, np.ndarray]]:
    weeks = np.array(sorted(df["yearweek"].unique()))
    if len(weeks) < 12:
        raise RuntimeError(f"Not enough training weeks for rolling validation: {len(weeks)}")

    if len(weeks) < min_train_weeks + n_splits:
        n_splits = max(2, min(3, len(weeks) // 26))
        min_train_weeks = max(52, len(weeks) - n_splits * 8)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for tr_w_idx, va_w_idx in tscv.split(weeks):
        if len(tr_w_idx) < min_train_weeks:
            continue
        tr_weeks = set(weeks[tr_w_idx])
        va_weeks = set(weeks[va_w_idx])
        tr_idx = df.index[df["yearweek"].isin(tr_weeks)].to_numpy()
        va_idx = df.index[df["yearweek"].isin(va_weeks)].to_numpy()
        if len(tr_idx) and len(va_idx):
            splits.append((tr_idx, va_idx))

    if not splits:
        cut = int(len(weeks) * 0.8)
        if cut <= 0 or cut >= len(weeks):
            raise RuntimeError("Cannot create fallback train/validation split")
        splits = [
            (
                df.index[df["yearweek"].isin(set(weeks[:cut]))].to_numpy(),
                df.index[df["yearweek"].isin(set(weeks[cut:]))].to_numpy(),
            )
        ]
    return splits


def _recursive_future_features(history: pd.DataFrame, future_exog: pd.DataFrame, forecast_period: int) -> pd.DataFrame:
    hist = history.sort_values("yearweek").copy()
    future = future_exog.sort_values("yearweek").head(forecast_period).copy()
    count_history = hist["count"].astype(float).dropna().tolist()
    rows = []

    for _, row in future.iterrows():
        item = row.to_dict()
        for lag in range(1, 9):
            item[f"lag_{lag}"] = count_history[-lag] if len(count_history) >= lag else np.nan
        for window in (4, 8, 13):
            vals = count_history[-window:]
            item[f"roll{window}_mean"] = float(np.mean(vals)) if vals else np.nan
            item[f"roll{window}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(item)

        # 第一版以最近實際值遞補未來 lag，避免 future lag 全部缺失。
        # 後續可升級為逐期使用上一期 stacking forecast 回填。
        count_history.append(count_history[-1] if count_history else 0.0)

    return pd.DataFrame(rows)


def _fit_predict_global_oof(train: pd.DataFrame, feature_cols: list[str], y_col: str, registry) -> pd.DataFrame:
    oof = pd.DataFrame(index=train.index)
    splits = _rolling_splits(train)
    for spec in registry:
        if spec.scope != "global_panel" or not spec.enabled:
            continue
        oof[spec.name] = np.nan
        for tr_idx, va_idx in splits:
            model = new_model(spec)
            model.fit(train.loc[tr_idx, feature_cols], train.loc[tr_idx, y_col].astype(float))
            oof.loc[va_idx, spec.name] = model.predict(train.loc[va_idx, feature_cols])
    return oof


def _fit_predict_local_oof(train: pd.DataFrame, exog_cols: list[str], registry) -> pd.DataFrame:
    oof = pd.DataFrame(index=train.index)
    splits = _rolling_splits(train)

    for spec in registry:
        if spec.scope == "naive" and spec.enabled:
            oof[spec.name] = np.nan
            for _, g in train.groupby(KEY_COLS, dropna=False):
                g = g.sort_values("yearweek")
                for tr_idx, va_idx in splits:
                    tr = g.loc[g.index.intersection(tr_idx)]
                    va = g.loc[g.index.intersection(va_idx)]
                    if len(tr) == 0 or len(va) == 0:
                        continue
                    model = new_model(spec)
                    model.fit(pd.DataFrame(index=tr.index), tr["count"])
                    oof.loc[va.index, spec.name] = model.predict(pd.DataFrame(index=va.index))

        if spec.name == "sarimax" and spec.enabled:
            oof[spec.name] = np.nan
            for _, g in train.groupby(KEY_COLS, dropna=False):
                g = g.sort_values("yearweek")
                for tr_idx, va_idx in splits:
                    tr = g.loc[g.index.intersection(tr_idx)]
                    va = g.loc[g.index.intersection(va_idx)]
                    if len(tr) < 80 or len(va) == 0:
                        continue
                    try:
                        model = fit_sarimax(tr["count"], tr[exog_cols].fillna(0))
                        oof.loc[va.index, spec.name] = predict_sarimax(model, va[exog_cols].fillna(0), len(va))
                    except Exception as exc:
                        print(f"[WARN] sarimax OOF failed keys={tuple(g[KEY_COLS].iloc[0])}: {type(exc).__name__}: {exc}")
    return oof


def _fit_final_base_predictions(
    train: pd.DataFrame,
    future: pd.DataFrame,
    feature_cols: list[str],
    exog_cols: list[str],
    registry,
    forecast_period: int,
) -> pd.DataFrame:
    future_pred = pd.DataFrame(index=future.index)

    for spec in registry:
        if spec.scope == "global_panel" and spec.enabled:
            model = new_model(spec)
            model.fit(train[feature_cols], train["count"].astype(float))
            future_pred[spec.name] = model.predict(future[feature_cols])

    for spec in registry:
        if spec.scope == "naive" and spec.enabled:
            future_pred[spec.name] = np.nan
            for keys, g in train.groupby(KEY_COLS, dropna=False):
                mask = (future["data_source"] == keys[0]) & (future["disease"] == keys[1]) & (future["county"] == keys[2])
                fg = future.loc[mask].sort_values("yearweek").head(forecast_period)
                if fg.empty:
                    continue
                model = new_model(spec)
                model.fit(pd.DataFrame(index=g.index), g.sort_values("yearweek")["count"])
                future_pred.loc[fg.index, spec.name] = model.predict(pd.DataFrame(index=fg.index))

        if spec.name == "sarimax" and spec.enabled:
            future_pred[spec.name] = np.nan
            for keys, g in train.groupby(KEY_COLS, dropna=False):
                mask = (future["data_source"] == keys[0]) & (future["disease"] == keys[1]) & (future["county"] == keys[2])
                fg = future.loc[mask].sort_values("yearweek").head(forecast_period)
                g = g.sort_values("yearweek")
                if len(g) < 80 or fg.empty:
                    continue
                try:
                    model = fit_sarimax(g["count"], g[exog_cols].fillna(0))
                    future_pred.loc[fg.index, spec.name] = predict_sarimax(model, fg[exog_cols].fillna(0), len(fg))
                except Exception as exc:
                    print(f"[WARN] sarimax final failed keys={keys}: {type(exc).__name__}: {exc}")

    return future_pred


def run_source(
    data_source: str,
    forecast_period: int,
    start_week: int,
    use_gpu: bool = False,
    meta_model: str = "ridge",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"[RUN] source={data_source}, forecast_period={forecast_period}, start_week={start_week}")

    feature = build_feature_table(data_source, start_week=start_week, forecast_period=forecast_period)
    exog_cols = get_exog_columns(feature)
    forecast_weeks = forecast_yearweeks(forecast_period)
    train_cut_week = latest_closed_yearweek()

    base_train = feature[(feature["is_future"] == False) & feature["count"].notna() & (feature["yearweek"] <= train_cut_week)].copy()
    if base_train.empty:
        raise RuntimeError(f"No training data available for source={data_source} before yearweek={train_cut_week}")

    future_raw = feature[(feature["is_future"] == True) & feature["yearweek"].isin(forecast_weeks)].copy()
    if future_raw.empty:
        raise RuntimeError(f"No future rows available for source={data_source}; check dim_weekdate and forecast_period")

    train = add_lag_rolling_features(base_train).dropna(subset=["lag_1", "count"]).reset_index(drop=True)
    if train.empty:
        raise RuntimeError(f"Training data became empty after lag feature generation for source={data_source}")

    future_parts = []
    for keys, hist in base_train.groupby(KEY_COLS, dropna=False):
        mask = (future_raw["data_source"] == keys[0]) & (future_raw["disease"] == keys[1]) & (future_raw["county"] == keys[2])
        fg = future_raw.loc[mask].sort_values("yearweek").head(forecast_period)
        if len(fg) == forecast_period:
            future_parts.append(_recursive_future_features(hist, fg, forecast_period))
        else:
            print(f"[WARN] incomplete future rows keys={keys}, rows={len(fg)}, expected={forecast_period}")

    if not future_parts:
        raise RuntimeError(f"No complete future rows available for source={data_source}")

    future = pd.concat(future_parts, ignore_index=True)

    feature_cols, numeric_cols, categorical_cols = _feature_columns(train, exog_cols)
    registry = build_base_registry(numeric_cols=numeric_cols, categorical_cols=categorical_cols, use_gpu=use_gpu)

    print(f"[RUN] source={data_source} train_rows={len(train)}, future_rows={len(future)}, exog_cols={len(exog_cols)}")

    global_oof = _fit_predict_global_oof(train, feature_cols, "count", registry)
    local_oof = _fit_predict_local_oof(train, exog_cols, registry)
    oof = pd.concat([global_oof, local_oof], axis=1)

    valid_mask = oof.notna().sum(axis=1) >= 2
    if valid_mask.sum() == 0:
        raise RuntimeError(f"OOF prediction is empty for source={data_source}")

    meta = fit_meta_model(oof.loc[valid_mask], train.loc[valid_mask, "count"], method=meta_model)
    stacked_oof = predict_meta(meta, oof.loc[valid_mask])
    metric = regression_metrics(train.loc[valid_mask, "count"], stacked_oof)

    future_base = _fit_final_base_predictions(train, future, feature_cols, exog_cols, registry, forecast_period)
    final_pred = predict_meta(meta, future_base)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    id_cols = KEY_COLS + ["yearweek"]
    forecast_df = future[id_cols].copy()
    forecast_df["forecast_count"] = final_pred
    forecast_df["model_name"] = f"stacking_{meta_model}"
    forecast_df["created_at"] = now

    base_rows = []
    for model_name in future_base.columns:
        tmp = future[id_cols].copy()
        tmp["base_model"] = model_name
        tmp["forecast_count"] = np.clip(np.rint(future_base[model_name].fillna(0)), 0, None).astype(int)
        tmp["created_at"] = now
        base_rows.append(tmp)
    base_df = pd.concat(base_rows, ignore_index=True) if base_rows else pd.DataFrame()

    metric_df = pd.DataFrame([
        {
            **{"data_source": data_source, "model_name": f"stacking_{meta_model}", "oof_rows": int(valid_mask.sum())},
            **metric,
            "created_at": now,
        }
    ])
    return forecast_df, base_df, metric_df


def run_all(
    data_sources: list[str],
    forecast_period: int,
    start_week: int,
    use_gpu: bool = False,
    meta_model: str = "ridge",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecasts, bases, metrics = [], [], []
    for source in data_sources:
        f, b, m = run_source(source, forecast_period, start_week, use_gpu=use_gpu, meta_model=meta_model)
        forecasts.append(f)
        bases.append(b)
        metrics.append(m)
    return pd.concat(forecasts, ignore_index=True), pd.concat(bases, ignore_index=True), pd.concat(metrics, ignore_index=True)
