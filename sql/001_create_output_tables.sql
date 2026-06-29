CREATE TABLE IF NOT EXISTS disease_forecast_data.model_forecast_weekly_county (
    id BIGSERIAL PRIMARY KEY,
    data_source TEXT NOT NULL,
    disease TEXT NOT NULL,
    county TEXT NOT NULL,
    yearweek INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    forecast_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS disease_forecast_data.model_base_prediction_weekly_county (
    id BIGSERIAL PRIMARY KEY,
    data_source TEXT NOT NULL,
    disease TEXT NOT NULL,
    county TEXT NOT NULL,
    yearweek INTEGER NOT NULL,
    base_model TEXT NOT NULL,
    forecast_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS disease_forecast_data.model_metric_weekly_county (
    id BIGSERIAL PRIMARY KEY,
    data_source TEXT NOT NULL,
    model_name TEXT NOT NULL,
    oof_rows INTEGER,
    mae DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    wape DOUBLE PRECISION,
    smape DOUBLE PRECISION,
    bias DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
