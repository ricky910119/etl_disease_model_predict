from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:
    from pmdarima import auto_arima
except Exception:  # pragma: no cover
    auto_arima = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    scope: str
    factory: Callable[[], object]
    enabled: bool = True


class SeasonalNaiveRegressor:
    """
    以去年同週優先，否則以最近 season_length 週均值補足。
    """

    def __init__(self, season_length: int = 52):
        self.season_length = season_length
        self.y_: list[float] = []
        self.fallback_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.y_ = pd.Series(y).astype(float).dropna().tolist()
        self.fallback_ = (
            float(np.nanmean(self.y_[-self.season_length:]))
            if self.y_
            else 0.0
        )

        if np.isnan(self.fallback_):
            self.fallback_ = 0.0

        return self

    def predict(self, x: pd.DataFrame):
        preds = []
        history = list(self.y_)

        for _ in range(len(x)):
            pred = (
                history[-self.season_length]
                if len(history) >= self.season_length
                else self.fallback_
            )
            preds.append(pred)
            history.append(pred)

        return np.asarray(preds, dtype=float)


class ConstantSeriesForecaster:
    """
    當某個 county / disease / source 的時間序列完全固定時，
    不跑 SARIMAX，直接回傳常數預測。
    """

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, n_periods: int, X=None):
        return np.repeat(self.value, n_periods)

class KerasMLPRegressor(BaseEstimator, RegressorMixin):
    """
    TensorFlow / Keras tabular MLP regressor.

    用於目前的 panel tabular forecasting 架構：
        categorical one-hot + numeric lag/rolling/weather/calendar features

    sklearn pipeline:
        ColumnTransformer -> numpy -> KerasMLPRegressor
    """

    def __init__(
        self,
        hidden_dims=(256, 128),
        dropout: float = 0.10,
        lr: float = 0.001,
        batch_size: int = 1024,
        max_epochs: int = 80,
        patience: int = 10,
        validation_fraction: float = 0.15,
        use_gpu: bool = False,
        random_state: int = 9101,
        verbose: int = 0,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, input_dim: int):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.random_state)

        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs

        for hidden_dim in self.hidden_dims:
            x = tf.keras.layers.Dense(int(hidden_dim))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

            if self.dropout and self.dropout > 0:
                x = tf.keras.layers.Dropout(float(self.dropout))(x)

        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.lr)),
            loss="mse",
        )

        return model

    def fit(self, X, y):
        import tensorflow as tf

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")

        n_rows = X_arr.shape[0]

        if n_rows < 10:
            self.fallback_ = float(np.nanmean(y_arr)) if n_rows else 0.0
            self.model_ = None
            return self

        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(np.std(y_arr))

        if self.y_std_ == 0 or np.isnan(self.y_std_):
            self.y_std_ = 1.0

        y_scaled = (y_arr - self.y_mean_) / self.y_std_

        valid_size = int(n_rows * float(self.validation_fraction))
        valid_size = max(1, valid_size)
        valid_size = min(valid_size, max(1, n_rows - 1))

        train_end = n_rows - valid_size

        X_train = X_arr[:train_end]
        y_train = y_scaled[:train_end]
        X_valid = X_arr[train_end:]
        y_valid = y_scaled[train_end:]

        gpu_devices = tf.config.list_physical_devices("GPU")

        if self.use_gpu and gpu_devices:
            device_name = "/GPU:0"
        else:
            device_name = "/CPU:0"

        if self.use_gpu and not gpu_devices:
            print("[WARN] use_gpu=True but TensorFlow GPU is unavailable. KerasMLP uses CPU.")

        with tf.device(device_name):
            self.model_ = self._build_model(input_dim=X_arr.shape[1])

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=int(self.patience),
                    restore_best_weights=True,
                )
            ]

            history = self.model_.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                epochs=int(self.max_epochs),
                batch_size=int(self.batch_size),
                shuffle=True,
                verbose=int(self.verbose),
                callbacks=callbacks,
            )

        self.input_dim_ = X_arr.shape[1]
        self.best_valid_loss_ = float(np.min(history.history.get("val_loss", [np.nan])))

        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float32)

        if getattr(self, "model_", None) is None:
            return np.repeat(getattr(self, "fallback_", 0.0), X_arr.shape[0])

        pred = self.model_.predict(
            X_arr,
            batch_size=int(self.batch_size),
            verbose=0,
        ).reshape(-1)

        pred = pred * self.y_std_ + self.y_mean_

        return pred.astype(float)

def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _as_numpy_transformer() -> FunctionTransformer:
    return FunctionTransformer(
        lambda x: np.asarray(x),
        validate=False,
        feature_names_out="one-to-one",
    )


def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    numeric = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        _one_hot_encoder(),
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_ridge(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> BaseEstimator:
    return make_pipeline(
        make_preprocessor(numeric_cols, categorical_cols),
        Ridge(alpha=1.0),
    )


def make_elasticnet(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> BaseEstimator:
    return make_pipeline(
        make_preprocessor(numeric_cols, categorical_cols),
        ElasticNet(
            alpha=0.02,
            l1_ratio=0.15,
            max_iter=5000,
            random_state=9101,
        ),
    )


def make_xgboost(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from xgboost import XGBRegressor

        kwargs = {} if use_gpu else {}

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            _as_numpy_transformer(),
            XGBRegressor(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=9101,
                n_jobs=-1,
                **kwargs,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] XGBoost unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}"
        )
        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )


def make_lightgbm(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from lightgbm import LGBMRegressor

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            _as_numpy_transformer(),
            LGBMRegressor(
                n_estimators=450,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="regression",
                device_type="cpu" ,
                random_state=9101,
                n_jobs=-1,
                verbose=-1,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] LightGBM unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}"
        )
        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )


def make_catboost(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from catboost import CatBoostRegressor

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            _as_numpy_transformer(),
            CatBoostRegressor(
                iterations=350,
                depth=5,
                learning_rate=0.04,
                loss_function="RMSE",
                task_type="CPU",
                random_seed=9101,
                verbose=False,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] CatBoost unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}"
        )
        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )
def make_xgboost_poisson(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from xgboost import XGBRegressor

        kwargs = {}
        if use_gpu:
            kwargs["device"] = "cuda"

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            XGBRegressor(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="count:poisson",
                tree_method="hist",
                random_state=9101,
                n_jobs=-1,
                **kwargs,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] xgboost_poisson unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )
def make_lightgbm_poisson(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from lightgbm import LGBMRegressor

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            LGBMRegressor(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="poisson",
                device_type="gpu" if use_gpu else "cpu",
                random_state=9101,
                n_jobs=-1,
                verbose=-1,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] lightgbm_poisson unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )
def make_catboost_poisson(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        from catboost import CatBoostRegressor

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            CatBoostRegressor(
                iterations=500,
                depth=4,
                learning_rate=0.03,
                loss_function="Poisson",
                task_type="GPU" if use_gpu else "CPU",
                random_seed=9101,
                verbose=False,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] catboost_poisson unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )
                
def make_keras_mlp(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
):
    try:
        import tensorflow as tf  # noqa: F401

        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            _as_numpy_transformer(),
            KerasMLPRegressor(
                hidden_dims=(256, 128),
                dropout=0.10,
                lr=0.001,
                batch_size=1024,
                max_epochs=80,
                patience=10,
                validation_fraction=0.15,
                use_gpu=use_gpu,
                random_state=9101,
                verbose=0,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] TensorFlow/Keras unavailable, fallback DummyRegressor: "
            f"{type(exc).__name__}: {exc}"
        )
        return make_pipeline(
            make_preprocessor(numeric_cols, categorical_cols),
            DummyRegressor(strategy="mean"),
        )

def fit_sarimax(y: pd.Series, x: pd.DataFrame | None = None):
    y = pd.Series(y).astype(float).dropna()

    if y.empty:
        return ConstantSeriesForecaster(0.0)

    if y.nunique(dropna=True) <= 1:
        return ConstantSeriesForecaster(float(y.iloc[-1]))

    if auto_arima is None:
        raise ImportError("pmdarima is required for SARIMAX/ARIMA models")

    x = None if x is None or x.shape[1] == 0 else x.fillna(0)

    return auto_arima(
        y,
        X=x,
        seasonal=True,
        m=52,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
    )


def predict_sarimax(
    model,
    x_future: pd.DataFrame | None,
    steps: int,
) -> np.ndarray:
    if isinstance(model, ConstantSeriesForecaster):
        return np.asarray(model.predict(n_periods=steps))

    names = [
        c for c in getattr(model.arima_res_.model, "exog_names", [])
        if c != "const"
    ]

    if names and x_future is not None:
        aligned = x_future.reindex(columns=names).fillna(0)
        return np.asarray(model.predict(n_periods=steps, X=aligned))

    return np.asarray(model.predict(n_periods=steps))


def build_base_registry(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_gpu: bool = False,
    model_task: str = "default",
) -> list[ModelSpec]:
    """
    建立 base model pool。

    不改資料粒度，只依 model_task 決定模型池。

    EV task:
        使用 count-oriented models。
        不使用 keras_mlp。

    non-EV task:
        使用原本穩定模型。
        不使用 keras_mlp。
    """
    is_ev_task = model_task in {
        "nhi_ev_branch",
        "rods_ev_national",
    }

    is_rods_ev_national = model_task == "rods_ev_national"

    registry = [
        ModelSpec(
            "seasonal_naive",
            "naive",
            lambda: SeasonalNaiveRegressor(52),
        ),
        ModelSpec(
            "ridge",
            "global_panel",
            lambda: make_ridge(numeric_cols, categorical_cols),
        ),
        ModelSpec(
            "elasticnet",
            "global_panel",
            lambda: make_elasticnet(numeric_cols, categorical_cols),
        ),
        ModelSpec(
            "xgboost",
            "global_panel",
            lambda: make_xgboost(numeric_cols, categorical_cols, use_gpu=use_gpu),
        ),
        ModelSpec(
            "lightgbm",
            "global_panel",
            lambda: make_lightgbm(numeric_cols, categorical_cols, use_gpu=use_gpu),
        ),
        ModelSpec(
            "catboost",
            "global_panel",
            lambda: make_catboost(numeric_cols, categorical_cols, use_gpu=use_gpu),
        ),
    ]

    # EV 才加入 count-based boosting
    if is_ev_task:
        registry.extend(
            [
                ModelSpec(
                    "xgboost_poisson",
                    "global_panel",
                    lambda: make_xgboost_poisson(
                        numeric_cols,
                        categorical_cols,
                        use_gpu=use_gpu,
                    ),
                ),
                ModelSpec(
                    "lightgbm_poisson",
                    "global_panel",
                    lambda: make_lightgbm_poisson(
                        numeric_cols,
                        categorical_cols,
                        use_gpu=use_gpu,
                    ),
                ),
                ModelSpec(
                    "catboost_poisson",
                    "global_panel",
                    lambda: make_catboost_poisson(
                        numeric_cols,
                        categorical_cols,
                        use_gpu=use_gpu,
                    ),
                ),
            ]
        )

    # RODS EV 是全國單序列，不要放太多一般 regression boosting
    if is_rods_ev_national:
        keep_names = {
            "seasonal_naive",
            "ridge",
            "xgboost_poisson",
            "lightgbm_poisson",
            "catboost_poisson",
        }

        registry = [
            spec for spec in registry
            if spec.name in keep_names
        ]

    return registry


def new_model(spec: ModelSpec):
    return spec.factory()