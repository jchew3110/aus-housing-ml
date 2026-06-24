# AUS Housing ML Pipeline

End-to-end ML pipeline predicting Australian residential property price index changes (next-quarter QoQ %) by capital city.

**Dataset**: ABS 6416.0 (RPPI) · RBA cash rate · ABS CPI · ABS Labour Force — 2011Q1–2021Q4  
**Models**: Ensemble (XGBoost + LightGBM) · XGBoost · LightGBM · Ridge baseline  
**Stack**: Python 3.11+ · pandas · scikit-learn · XGBoost · LightGBM · Optuna · FastAPI · Docker

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
make install

# 2. Download data (ABS + RBA — no auth required)
make download-data

# 3. Train all models (Ridge + XGBoost + LightGBM + Ensemble)
make train

# 4. Start the prediction API
make docker-up
curl http://localhost:8000/health

# 5. Make a prediction using raw time-series data (recommended)
curl -X POST http://localhost:8000/api/v1/predict/raw \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Sydney",
    "rppi_history": [
      {"year": 2020, "quarter": 3, "rppi_index": 155.2},
      {"year": 2020, "quarter": 4, "rppi_index": 158.0},
      {"year": 2021, "quarter": 1, "rppi_index": 163.4},
      {"year": 2021, "quarter": 2, "rppi_index": 170.1},
      {"year": 2021, "quarter": 3, "rppi_index": 177.8},
      {"year": 2021, "quarter": 4, "rppi_index": 181.5}
    ],
    "macro_history": [
      {"year": 2020, "quarter": 3, "cash_rate": 0.25, "cpi": 116.5, "unemployment_rate": 6.8},
      {"year": 2020, "quarter": 4, "cash_rate": 0.10, "cpi": 117.2, "unemployment_rate": 6.4},
      {"year": 2021, "quarter": 1, "cash_rate": 0.10, "cpi": 118.8, "unemployment_rate": 5.8},
      {"year": 2021, "quarter": 2, "cash_rate": 0.10, "cpi": 119.9, "unemployment_rate": 5.1},
      {"year": 2021, "quarter": 3, "cash_rate": 0.10, "cpi": 121.3, "unemployment_rate": 4.6},
      {"year": 2021, "quarter": 4, "cash_rate": 0.10, "cpi": 123.5, "unemployment_rate": 4.2}
    ]
  }'
```

---

## Project structure

```
aus-housing-ml/
├── src/
│   ├── data/          # Download, parse, merge ABS/RBA data
│   ├── features/      # Lag, rolling, momentum, seasonal, macro, city encoding
│   ├── models/        # Ridge, XGBoost, LightGBM, Ensemble, evaluator, registry
│   └── api/           # FastAPI prediction service
├── tests/             # pytest suite (107 tests)
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature construction walkthrough
│   └── 03_model_comparison.ipynb     # Metrics, feature importance, residual analysis
├── scripts/
│   ├── download_data.py   # Download all data sources
│   └── train_models.py    # Train and evaluate all models
├── .github/workflows/ci.yml  # GitHub Actions (lint + test)
├── data/                     # gitignored (auto-generated)
├── models/                   # gitignored (trained artifacts)
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
                            Ridge · XGBoost · LightGBM · Ensemble
                                                 │
                         evaluator (MAE, RMSE, R², MAPE, directional accuracy)
                                  walk-forward cross-validation
                                                 │
                              ModelRegistry → FastAPI → Docker
```

### Features (27 total)

| Group | Features |
|-------|----------|
| Price levels | `rppi_lag_1`, `rppi_lag_2`, `rppi_lag_4` |
| Price changes | `rppi_qoq_pct_lag1`, `rppi_yoy_pct_lag1` |
| Rolling stats | `rolling_mean_4q`, `rolling_std_4q` |
| Momentum | `momentum_streak_lag1`, `price_acceleration_lag1` |
| Cash rate | `cash_rate_lag1`, `cash_rate_delta_lag1`, `rate_regime` |
| Inflation | `cpi_lag1`, `cpi_yoy_pct_lag1` |
| Labour market | `unemp_lag1`, `unemp_delta_lag1` |
| Seasonal | `quarter`, `quarter_sin`, `quarter_cos` |
| City dummies | `city_Sydney` … `city_Canberra` (8) |

**`momentum_streak_lag1`**: signed count of consecutive up/down quarters. Captures trend persistence — a city on a 6-quarter run behaves differently from one at an inflection point.

**`price_acceleration_lag1`**: second derivative of price changes. Identifies whether momentum is building or fading.

**`rate_regime`**: cash rate bucketed into 4 ordinal levels (very low / low / normal / high). Captures the non-linear effect of interest rates — a move from 0.1% to 1.1% is very different from 5% to 6%.

### No data leakage guarantees

- All lag features use `.groupby(city).shift(n)` — never raw `.shift()` (prevents cross-city contamination)
- Rolling stats use `.shift(1)` before the rolling window (current period excluded)
- Target = next-quarter QoQ % change via `.groupby(city).shift(-1)`
- Train/val/test split is strictly temporal: 2011Q1–2018Q4 train · 2019Q1–2020Q4 val · 2021Q1–2021Q4 test
- `StandardScaler` fit only on train split, applied to val/test
- Optuna tuning uses only train + val; test set is untouched until final comparison

---

## API

Three prediction endpoints:

### `POST /api/v1/predict/raw` *(recommended)*

Accepts raw time-series data — the server computes all features automatically.
Provide at least 6 consecutive quarters of RPPI + macro observations.
Returns a prediction for the quarter after your last provided period.

### `POST /api/v1/predict`

Accepts a pre-computed feature vector (for callers who already maintain lag state).

### `POST /api/v1/predict/batch`

Accepts up to 100 feature-vector requests in a single call.

**Example response:**
```json
{
  "city": "Sydney",
  "year": 2022,
  "quarter": 1,
  "predicted_qoq_pct_change": 1.84,
  "direction": "up",
  "confidence_interval": {"lower": 0.62, "upper": 3.06, "confidence": 0.9},
  "model_name": "ensemble",
  "model_version": "1.0"
}
```

**`GET /health`** · **`GET /model-info`**

Interactive docs: http://localhost:8000/docs

---

## Results (2021Q1–2021Q4 test split, 32 observations)

| Model    | MAE  | RMSE | R²     | Dir. Accuracy |
|----------|------|------|--------|---------------|
| Ensemble | 4.46 | 4.91 | −4.62  | **100%**      |
| XGBoost  | 4.43 | 4.89 | −4.58  | **100%**      |
| LightGBM | 4.48 | 4.93 | −4.67  | **100%**      |
| Ridge    | 4.93 | 5.42 | −5.85  | 87.5%         |

*Validation split (2019Q1–2020Q4, 64 obs): XGBoost MAE=1.81, LightGBM MAE=1.78, Ridge MAE=2.09.*

**Note on test set performance**: The 2021 test period coincides with an extraordinary COVID-era housing boom driven by the RBA cash rate at a historical low of 0.1% — a monetary regime the models had never seen in training (2011–2018 max rate ~4.75%). All models systematically underestimate the magnitude of 2021 price surges, hence the strongly negative R². The 100% directional accuracy on test for tree models (correctly predicting "up" for all 32 city-quarters in 2021) is genuinely meaningful — the models identified the sustained upward regime even if they missed the amplitude. Point forecasts should be interpreted with wide confidence intervals.

---

## Make targets

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make download-data` | Download ABS + RBA files |
| `make train` | Full train (50 Optuna trials per model + ensemble) |
| `make train-fast` | Ridge only, quick validation |
| `make test` | Run pytest suite (107 tests) |
| `make test-cov` | Run with coverage report |
| `make docker-build` | Build Docker image |
| `make docker-up` | Start API container |
| `make notebook` | Launch Jupyter Lab |

---

## Data sources

| Source | Catalogue | Update frequency | Coverage |
|--------|-----------|-----------------|---------|
| [Residential Property Price Indexes](https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/residential-property-price-indexes-eight-capital-cities/latest-release) | ABS 6416.0 | Quarterly | 2003–2021 |
| [Consumer Price Index](https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/dec-2021/640101.xlsx) | ABS 6401.0 | Quarterly (pinned to Dec 2021) | 1948–2021 |
| [Labour Force](https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia/latest-release) | ABS 6202.0 | Monthly | 1978–present |
| [Cash Rate Target](https://www.rba.gov.au/statistics/tables/csv/f1-data.csv) | RBA F1 | Daily | 1990–present |

All data is publicly available with no API key required.

> **CPI note**: The ABS switched CPI to monthly from Nov 2022. This pipeline pins the download to the Dec 2021 quarterly release so the series aligns with the RPPI data range.
