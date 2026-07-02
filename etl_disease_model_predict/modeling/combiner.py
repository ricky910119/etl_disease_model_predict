from __future__ import annotations

import pandas as pd

from etl_disease_model_predict.modeling.ensemble import (
    RULE_BASED_METHODS,
    DEFAULT_ENSEMBLE_TOPK,
    EnsembleModel,
    fit_ensemble,
    predict_ensemble,
    extract_ensemble_weights,
)
from etl_disease_model_predict.modeling.stacking import (
    REGRESSION_METHODS,
    MetaModel,
    fit_meta_model,
    predict_meta,
    extract_meta_weights,
)

# combiner.py 是 pipeline 層（train_predict.py / holdout_eval.py / main.py）
# 呼叫「base model 組合方法」時唯一該用的入口。
#
# stacking.py（迴歸型 meta model）與 ensemble.py（規則型 ensemble）彼此完全獨立，
# 互不 import，這裡是唯一同時知道兩邊存在、負責依 method 名稱路由的地方。
# 之後不論是新增迴歸型方法還是規則型方法，pipeline 層都不需要跟著改。

ALL_META_METHODS = REGRESSION_METHODS | RULE_BASED_METHODS

CombinerModel = MetaModel | EnsembleModel

def combiner_layer(method: str) -> str:
    """
    回傳這個組合方法在輸出（metric_df / forecast_df / base_df）裡該標記的 model_layer。

    REGRESSION_METHODS（ridge/elasticnet/lasso/huber/nonnegative_linear）-> "stacking"
    RULE_BASED_METHODS（simple_average/weighted_average/topk_average）  -> "ensemble"

    這是唯一決定這個標籤的地方，避免規則型 ensemble 在報表裡被誤標成 stacking。
    """
    method = str(method).lower().strip()

    if method in RULE_BASED_METHODS:
        return "ensemble"

    if method in REGRESSION_METHODS:
        return "stacking"

    raise ValueError(
        f"Unknown combiner method={method}. Allowed: {sorted(ALL_META_METHODS)}"
    )


def combiner_model_name(method: str) -> str:
    """回傳這個組合方法在輸出裡該用的 model_name，例如 stacking_ridge 或 ensemble_simple_average。"""
    method = str(method).lower().strip()
    return f"{combiner_layer(method)}_{method}"
    
def fit_combiner(
    x,
    y,
    method: str = "ridge",
    topk: int | None = None,
) -> CombinerModel:
    """依 method 名稱路由到 stacking.fit_meta_model() 或 ensemble.fit_ensemble()。"""
    method = str(method).lower().strip()

    if method in RULE_BASED_METHODS:
        return fit_ensemble(x, y, method=method, topk=topk or DEFAULT_ENSEMBLE_TOPK)

    if method in REGRESSION_METHODS:
        return fit_meta_model(x, y, method=method)

    raise ValueError(
        f"Unknown combiner method={method}. Allowed: {sorted(ALL_META_METHODS)}"
    )


def predict_combiner(model: CombinerModel, x) -> "pd.Series | pd.DataFrame":
    """依 model 實際型別路由到 stacking.predict_meta() 或 ensemble.predict_ensemble()。"""
    if isinstance(model, EnsembleModel):
        return predict_ensemble(model, x)

    return predict_meta(model, x)


def extract_combiner_weights(model: CombinerModel) -> pd.DataFrame:
    """依 model 實際型別路由到 stacking.extract_meta_weights() 或 ensemble.extract_ensemble_weights()。"""
    if isinstance(model, EnsembleModel):
        return extract_ensemble_weights(model)

    return extract_meta_weights(model)