# AUS Housing ML Pipeline

End-to-end ML pipeline predicting Australian residential property price index changes (next-quarter QoQ %) by capital city.

**Dataset**: ABS 6416.0 (RPPI) + RBA cash rate + ABS CPI + ABS Labour Force  
**Model**: LightGBM / XGBoost (Optuna-tuned) vs Ridge baseline  
**Stack**: Python 3.11 · pandas · scikit-learn · XGBoost · LightGBM · FastAPI · Docker

---

## Why this dataset

Australia's housing affordability crisis makes next-quarter price prediction immediately relevant to investors, policymakers, and buyers. The multi-source data (interest rates, inflation, unemployment → prices) demonstrates a real ETL pipeline rather than a single-file CSV import.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
make install

# 2. Download data (ABS + RBA — no auth required)
make download-data

# 3. Train all models (Ridge + XGBoost + LightGBM)
make train

# 4. Start the prediction API
make docker-up
curl http://localhost:8000/health

# 5. Make a prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Sydney",
    "quarter": 2,
    "year": 2025,
    "rppi_current": 163.4,
    "rppi_lag1": 161.8,
    "rppi_lag2": 159.5,
    "rppi_lag4": 154.2,
    "rppi_qoq_pct_current": 1.0,
    "rppi_yoy_pct_current": 5.9,
    "rolling_mean_4q": 1.1,
    "rolling_std_4q": 0.4,
    "cash_rate": 4.35,
    "cash_rate_prev": 4.35,
    "cpi": 131.2,
    "cpi_prev_year": 124.5,
    "unemployment_rate": 4.1,
    "unemployment_rate_prev": 4.0
  }'
```

---

## Project structure

```
aus-housing-ml/
├── src/
│   ├── data/          # Download, parse, merge ABS/RBA data
│   ├── features/      # Lag features, macro features, seasonality, city encoding
│   ├── models/        # Ridge baseline, XGBoost, LightGBM, evaluator, registry
│   └── api/           # FastAPI prediction service
├── tests/             # pytest test suite (features, models, API)
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature construction walkthrough
│   └── 03_model_comparison.ipynb     # Metrics, SHAP, residual analysis
├── scripts/
│   ├── download_data.py   # Download all data sources
│   └── train_models.py    # Train and evaluate all models
├── data/              # gitignored (auto-generated)
├── models/            # gitignored (trained artifacts)
├── Dockerfile
└── docker-compose.yml
```

---

## Pipeline

```
ABS 6416.0 (RPPI)  ─┐
ABS 6401.0 (CPI)   ─┼─ parse_abs_excel() ─┐
ABS 6202.0 (LF)    ─┘                      ├─ merge_to_panel() ─ FeaturePipeline ─ train/val/test split
RBA F1 (cash rate) ──── parse_rba_csv() ───┘
                                                 │
                                   Ridge · XGBoost · LightGBM
                                                 │
                                    evaluator (MAE, RMSE, R², MAPE, directional accuracy)
                                                 │
                                      ModelRegistry → FastAPI → Docker
```

### No data leakage guarantees

- All lag features use `.groupby(city).shift(n)` — never raw `.shift()` (prevents cross-city contamination)
- Rolling stats use `.shift(1)` before the rolling window (current period excluded)
- Target = next-quarter QoQ % change via `.groupby(city).shift(-1)`
- Train/val/test split is strictly temporal (2003–2019 / 2020–2021 / 2022–present)
- `StandardScaler` fit only on train split, applied to val/test
- Optuna tuning uses only train+val; test set is untouched until final comparison

---

## API

**`POST /api/v1/predict`**

```json
{
  "predicted_qoq_pct_change": 1.24,
  "direction": "up",
  "confidence_interval": {"lower": 0.31, "upper": 2.17, "confidence": 0.9},
  "model_name": "lgbm",
  "model_version": "1.0"
}
```

**`GET /health`** · **`GET /model-info`**

Interactive docs: http://localhost:8000/docs

---

## Results (example — rerun to get actuals)

| Model   | MAE  | RMSE | R²   | Dir. Accuracy |
|---------|------|------|------|---------------|
| Ridge   | 0.81 | 1.12 | 0.61 | 66%           |
| XGBoost | 0.65 | 0.89 | 0.73 | 71%           |
| LightGBM| 0.61 | 0.84 | 0.75 | 73%           |

*Directional accuracy = fraction of quarters where the predicted direction (up/down) was correct.*

---

## Make targets

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make download-data` | Download ABS + RBA files |
| `make train` | Full train (50 Optuna trials per model) |
| `make train-fast` | Ridge only, quick validation |
| `make test` | Run pytest suite |
| `make test-cov` | Run with coverage report |
| `make docker-build` | Build Docker image |
| `make docker-up` | Start API container |
| `make notebook` | Launch Jupyter Lab |

---

## Data sources

| Source | Catalogue | Update frequency |
|--------|-----------|-----------------|
| [Residential Property Price Indexes](https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/residential-property-price-indexes-eight-capital-cities/latest-release) | ABS 6416.0 | Quarterly |
| [Consumer Price Index](https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/latest-release) | ABS 6401.0 | Quarterly |
| [Labour Force](https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia/latest-release) | ABS 6202.0 | Monthly |
| [Cash Rate Target](https://www.rba.gov.au/statistics/tables/csv/f1-data.csv) | RBA F1 | Daily |

All data is publicly available with no API key required.
