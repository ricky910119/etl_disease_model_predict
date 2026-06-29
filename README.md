# etl_disease_model_predict

疾病週資料模型預測專案。此版從已整理好的週層級資料表開始建模，將病例週資料、縣市週天氣資料與年週維度特徵整合後，建立 base model 與 stacking meta model，最後輸出未來週縣市別預測。

## 使用資料表

```text
disease_forecast_data.weather_weekly_city
disease_forecast_data.model_nhi_er_weekly_county
disease_forecast_data.model_nhi_opd_weekly_county
disease_forecast_data.model_rods_weekly_county
```

維度資料使用：

```text
DIM_DATA.public.dim_new_year_holiday
DIM_DATA.public.dim_weekdate
```

## 整體資料流

```text
raw tables
  ↓
daily staging
  ↓
weekly staging
  ↓
feature table
  ↓
model training
  ↓
base model prediction
  ↓
stacking meta model
  ↓
forecast result
```

本專案目前從 `weekly staging / model_*_weekly_county` 階段開始執行。

## 專案架構

```text
etl_disease_model_predict/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── sql/
│   └── 001_create_output_tables.sql
├── outputs/
│   └── .gitkeep
└── etl_disease_model_predict/
    ├── config.py
    ├── db/
    │   └── postgres.py
    ├── features/
    │   ├── dataset.py
    │   └── dim_features.py
    ├── modeling/
    │   ├── base_models.py
    │   └── stacking.py
    ├── pipeline/
    │   └── train_predict.py
    └── utils/
        └── week.py
```

## 本次修正版重點

此版已移除第一版不適合目前環境的設定：

```text
POSTGRES_URL
.env
SQLAlchemy create_engine
python-dotenv
psycopg2-binary requirement
```

資料庫連線改回既有 ETL 專案慣用方式：

```python
from eic_utils import conn
```

其中：

- 疾病與天氣 model table 使用 `dbname="postgres"`
- `dim_new_year_holiday`、`dim_weekdate` 使用 `dbname="DIM_DATA"`

## 建模策略

此版廢除舊版固定 `model_expand/model_data` 的寫法，改成：

```text
model registry
  ↓
rolling origin OOF prediction
  ↓
meta feature table
  ↓
stacking meta model
  ↓
final base model refit
  ↓
final forecast
```

模型增減集中在：

```text
etl_disease_model_predict/modeling/base_models.py
```

要新增或停用模型，只需要調整 `build_base_registry()`。

## 第一層 Base Models

| 模型 | scope | 說明 |
| --- | --- | --- |
| Seasonal Naive | local series | 疾病季節性 baseline，優先取去年同週 |
| SARIMAX | local series | 傳統時間序列模型，限制搜尋範圍避免過慢 |
| Ridge | global panel | 線性可解釋 baseline |
| ElasticNet | global panel | 線性稀疏化補充 |
| XGBoost | global panel | 主要非線性模型 |
| LightGBM | global panel | 主要非線性模型 |
| CatBoost | global panel | 類別與非線性補充 |

## Global Panel Model

第一版優先採用 global panel，而不是一開始就所有縣市全面獨立建模。

建模資料概念：

```text
y_count ~ county + disease + source + lag + rolling + weather + calendar
```

特徵包含：

- 類別特徵：`county`、`disease`、`data_source`
- 病例 lag：`lag_1` 到 `lag_8`
- rolling 特徵：`roll4_mean/std`、`roll8_mean/std`、`roll13_mean/std`
- 天氣週特徵：來自 `weather_weekly_city`
- 年週維度特徵：`cnt`、`voc`、`eve`、`ev_period`、`di_period`、`covid`

## Stacking 設計

Stacking 使用 out-of-fold prediction，不使用 in-sample prediction 訓練 meta model。

流程：

```text
1. rolling origin validation 產生 base model OOF prediction
2. base OOF prediction 組成 meta feature
3. 訓練 meta model
4. final base model 用全資料重訓
5. final base prediction 丟入 meta model
6. 輸出 final forecast
```

Meta model 第一版提供：

```text
ridge
elasticnet
```

預設使用 `ridge`。

## Future rows 修正

第一版有一個重要問題：如果直接用 target table merge weather，target table 不會有未來週，所以 future rows 會是空的。

此版已修正為：

```text
歷史 county/disease/source 組合
  ×
forecast yearweeks
  ↓
產生未來預測列
  ↓
merge weather_weekly_city + dim features
```

因此即使病例 target 表沒有未來週，也可以產生未來 `forecast_period` 週的預測列。

## 安裝

```bash
cd etl_disease_model_predict
pip install -r requirements.txt
```

內部環境需要已可使用：

```python
from eic_utils import conn
```

## 建立輸出表

此專案不再依賴 `POSTGRES_URL`，建表方式依你的環境選擇：

```bash
psql -d postgres -f sql/001_create_output_tables.sql
```

或用既有資料庫工具執行 `sql/001_create_output_tables.sql`。

## 執行預測

輸出 CSV：

```bash
python main.py
```

指定來源：

```bash
python main.py --sources nhi_er nhi_opd rods
```

指定預測週數：

```bash
python main.py --forecast-period 8
```

指定 meta model：

```bash
python main.py --meta-model elasticnet
```

啟用 GPU 參數：

```bash
python main.py --use-gpu
```

輸出 CSV 並寫入 PostgreSQL：

```bash
python main.py --write-db
```

## 輸出檔案

```text
outputs/forecast_result.csv
outputs/base_model_prediction.csv
outputs/model_metric.csv
```

## 輸出資料表

```text
disease_forecast_data.model_forecast_weekly_county
disease_forecast_data.model_base_prediction_weekly_county
disease_forecast_data.model_metric_weekly_county
```

## 欄位需求

疾病週資料表至少需要：

```text
yearweek
county 或 city
count 或 weekly_count 或 case_count 或 cnt
disease 或 disease_name
```

天氣週資料表至少需要：

```text
yearweek
county 或 city
```

若未來週尚無天氣資料，模型仍會執行，天氣欄位會由各模型 preprocessing 的 imputer 處理。

## Git 推送建議

修正後建議使用一般版本控制流程：

```bash
git status
git add .
git commit -m "Refactor disease model prediction pipeline"
git push origin main
```
