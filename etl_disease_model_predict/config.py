from __future__ import annotations

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    """專案設定。

    本專案在疾管署內部環境執行時，資料庫連線統一走 eic_utils.conn。
    
    """

    forecast_period: int = int(os.getenv("FORECAST_PERIOD", "8"))
    train_start_yearweek: int | None = (
        int(os.getenv("TRAIN_START_YEARWEEK"))
        if os.getenv("TRAIN_START_YEARWEEK")
        else None
    )

    dim_dbname: str = os.getenv("DIM_DBNAME", "DIM_DATA")
    postgres_dbname: str = os.getenv("POSTGRES_DBNAME", "postgres")

    weather_table: str = "disease_forecast_data.weather_weekly_city"
    source_tables: dict[str, str] = field(default_factory=lambda: {
        "nhi_er": "disease_forecast_data.model_nhi_er_weekly_county",
        "nhi_opd": "disease_forecast_data.model_nhi_opd_weekly_county",
        "rods": "disease_forecast_data.model_rods_weekly_county",
    })

    forecast_table: str = "disease_forecast_data.model_forecast_weekly_county"
    base_prediction_table: str = "disease_forecast_data.model_base_prediction_weekly_county"
    metric_table: str = "disease_forecast_data.model_metric_weekly_county"


settings = Settings()
