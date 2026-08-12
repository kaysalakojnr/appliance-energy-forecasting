# Appliance Energy Forecasting

This repository contains the reproducible codebase for **Assignment 2: Time Series Coding Case Study and Report**. It models and forecasts household appliance energy consumption using benchmark forecasts, seasonal ARIMA/SARIMAX, feature-based machine learning, and the Chronos-Bolt time-series foundation model.

The project follows the repository organisation and reproducibility guidance supplied with the assignment: reusable code is kept in `src/`, notebooks are retained for exploration and explanation, scripts provide command-line entry points, generated outputs are separated from source code, and tests check the most important leakage and evaluation assumptions.

## Project aim

The target is short-term household appliance energy use. The analysis asks:

1. Which simple benchmark is strongest?
2. Does a seasonal ARIMA/SARIMAX model improve on that benchmark?
3. Do lag, rolling-window, time, sensor and weather features improve feature-based forecasting?
4. Does a zero-shot time-series foundation model improve on the simpler methods?
5. Which covariates are genuinely available at the forecast origin?
6. Which model provides the best practical balance of accuracy, interpretability, uncertainty, computational cost and deployability?

## Dataset

The project uses the UCI **Appliances Energy Prediction** dataset. The original data are sampled every 10 minutes and contain appliance energy consumption, lighting, indoor temperature/humidity measurements, outdoor weather variables and timestamps.

The raw CSV is **not committed to the repository**. The pipeline downloads it directly from UCI when required:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv
```

Preprocessing used in the assignment:

- parse `date` as a timestamp and sort chronologically;
- remove the random control columns `rv1` and `rv2`;
- verify duplicate timestamps, missing timestamps and missing cells;
- resample six 10-minute observations to an hourly mean;
- drop one incomplete terminal hour;
- retain **3,289 hourly observations and 26 variables**.

The target is:

```text
Appliances
```

## Forecasting task

The common forecasting design is:

```text
Forecast horizon: 24 hours
Test period:       final 14 days (336 hours)
Forecast origins:  14 non-overlapping rolling origins
Training window:   expanding
```

At each origin, models use only information available strictly before the forecast block, except the explicitly labelled **Regime B conditional forecast**, which quantifies the value of realised future covariates.

## Models

### 1. Benchmark models

- historical mean;
- naive/persistence;
- daily seasonal naive (lag 24);
- weekly seasonal naive (lag 168);
- drift.

The strongest benchmark was **weekly seasonal naive** with MAE **42.92 Wh**.

### 2. SARIMA / SARIMAX

The required non-seasonal AIC search evaluates all combinations:

```text
p = 0..6
d = 0..2
q = 0..6
```

This gives 147 non-seasonal candidates. The executed analysis then carries the three strongest converged base orders into a daily seasonal search over `(P,D,Q) in {0,1}^3`, with `s=24`.

The selected specification is:

```text
SARIMA(2,0,6)(0,1,1,24)
AIC = 32137.2
```

The repository also evaluates calendar and weather SARIMAX variants.

### 3. Feature-based models

The feature table contains:

- deterministic hour/day/week features and cyclical encodings;
- appliance lags including 24, 48, 72, 168 and 336 hours;
- rolling and same-hour historical statistics;
- lagged indoor temperature and humidity information;
- lagged lighting and outdoor weather information.

The main feature-based models are:

- `HistGradientBoostingRegressor`;
- `RandomForestRegressor`.

The notebooks additionally contain optional XGBoost/LightGBM comparisons when those packages are available.

### 4. Foundation model

The foundation-model experiment uses **Chronos-Bolt base** zero-shot:

```text
model:          amazon/chronos-bolt-base
context length: 512 hours
horizon:        24 hours
interval:       80% (0.1 to 0.9 quantiles)
```

No Chronos parameters are fitted to the appliance dataset.

## Headline results

All values below are evaluated on the same 336-hour holdout.

| Model | Family | MAE (Wh) | RMSE (Wh) | MASE | sMAPE (%) | Bias (Wh) |
|---|---|---:|---:|---:|---:|---:|
| Chronos zero-shot | Foundation | **33.58** | 67.53 | **0.629** | 26.01 | -18.05 |
| HGB Regime B* | ML / conditional | 36.55 | **61.51** | 0.685 | 30.55 | 3.00 |
| SARIMA | Statistical | 37.06 | 64.08 | 0.695 | 30.18 | -3.15 |
| SARIMAX weather | Statistical | 38.33 | 64.24 | 0.718 | 32.77 | -3.32 |
| Random Forest | ML | 39.37 | 64.06 | 0.738 | 31.70 | 2.22 |
| SARIMAX calendar | Statistical | 39.66 | 63.53 | 0.743 | 33.62 | 3.07 |
| Histogram Gradient Boosting | ML | 40.03 | 65.80 | 0.750 | 32.53 | 0.73 |
| Weekly seasonal naive | Benchmark | 42.92 | 79.67 | 0.804 | 32.67 | -12.62 |
| Daily seasonal naive | Benchmark | 48.39 | 85.79 | 0.907 | 35.57 | 1.67 |
| Mean | Benchmark | 50.01 | 73.99 | 0.937 | 46.24 | -3.06 |
| Naive | Benchmark | 143.38 | 170.62 | 2.687 | 87.07 | 114.67 |
| Drift | Benchmark | 143.94 | 171.26 | 2.698 | 87.23 | 115.31 |

`*` Regime B is a **conditional forecast** because some future covariates are supplied from the held-out period and would not normally be observed at forecast origin.

Chronos gives the best observed MAE, but its improvement over the leading alternatives is not statistically decisive on only 14 forecast origins. The report therefore recommends **SARIMA as the practical deployment compromise**, while Chronos is the accuracy-oriented alternative where the heavier model dependency is acceptable.

## Feature findings

The ablation analysis shows that more features do not automatically improve generalisation:

| Feature group alone | MAE (Wh) |
|---|---:|
| Calendar | 40.91 |
| Target rolling | 41.25 |
| Target lags | 41.75 |
| Indoor sensors | 44.21 |
| Lights | 46.18 |
| Weather | 52.62 |

The compact **calendar + rolling-target** subset reached **37.87 Wh MAE using 33 features**, outperforming the full 76-feature model at 40.03 Wh. This supports the conclusion that time-of-day and smoothed recent demand carry more useful signal than the additional environmental channels.

## Repository structure

```text
appliance-energy-forecasting/
│
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_preparation_and_eda.ipynb
│   ├── 02_forecasting_design.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarima_sarimax_models.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_feature_based_models.ipynb
│   ├── 07_foundation_model.ipynb
│   ├── 08_model_comparison.ipynb
│   └── 09_critical_analysis.ipynb
│
├── src/
│   └── appliance_energy/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── evaluation.py
│       ├── plotting.py
│       ├── pipeline.py
│       └── models/
│           ├── __init__.py
│           ├── benchmarks.py
│           ├── sarimax.py
│           ├── feature_models.py
│           └── foundation.py
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_pipeline.py
│   └── evaluate_models.py
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── metrics/
│   └── model_objects/
│
├── reports/
│   ├── Assignment_2_Report_8_PAGE_COMPACT.docx
│   └── figures/
│
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_evaluation.py
    └── test_benchmarks.py
```

## Installation

Python 3.11 is recommended.

### `venv`

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and the local package:

```bash
pip install -r requirements.txt
pip install -e .
```

Chronos downloads pretrained weights from Hugging Face on first use. The model is therefore the slowest/heaviest dependency in a completely fresh environment.

## Running the pipeline

The principal entry point is:

```bash
python scripts/run_pipeline.py
```

This performs the following steps:

1. download/load the raw UCI data;
2. resample to complete hourly means;
3. build the chronological rolling-origin design;
4. generate benchmark forecasts;
5. fit the selected SARIMA and two SARIMAX variants;
6. build leakage-safe Regime A and conditional Regime B features;
7. fit feature-based models;
8. run Chronos-Bolt zero-shot forecasting;
9. evaluate all forecasts on common metrics;
10. save forecasts, metrics and comparison figures.

Useful options:

```bash
# Run without the heavy foundation model
python scripts/run_pipeline.py --skip-foundation

# Also rerun the full p,d,q AIC search and seasonal candidate search
python scripts/run_pipeline.py --full-sarimax-grid
```

The default pipeline uses the selected SARIMA order from the completed analysis so that ordinary reproduction does not spend several minutes repeating model selection. The full search remains available with `--full-sarimax-grid`.

## Outputs

The pipeline writes:

```text
outputs/forecasts/all_forecasts.csv
outputs/metrics/model_comparison.csv
outputs/figures/forecast_comparison.png
outputs/figures/model_mae.png
```

The repository already includes the main figures and headline metric tables produced for the submitted report. Generated raw/interim data and model objects are ignored by Git.

## Evaluation metrics

All models use the same held-out period and the same MASE denominator calculated from training data only.

Reported point metrics are:

- MAE;
- RMSE;
- MASE;
- sMAPE;
- bias.

Probabilistic models additionally report interval coverage (PICP) and mean prediction-interval width (MPIW).

Every advanced model is compared with the **strongest benchmark**, not merely with other complex models.

## Data leakage and forecast realism

Temporal leakage is explicitly controlled:

- no random train/test splitting;
- lag/rolling target variables use only information at least 24 hours old for a 24-hour direct forecast;
- training rows are strictly earlier than each forecast origin;
- calendar variables are valid because they are deterministically known in advance;
- Regime A uses lagged sensor/weather values;
- Regime B deliberately uses realised future covariates and is labelled **conditional** rather than an operational forecast.

The Part 5 notebook also contains explicit leakage tests. The strongest check rebuilds a feature row from history truncated at that row's timestamp and verifies that it is identical to the row built when the full dataset is available.

## Tests

Run the unit tests with:

```bash
pytest
```

The tests check, among other things:

- incomplete hours are removed during resampling;
- benchmark forecast lengths and seasonal lags are correct;
- MASE is zero for a perfect forecast;
- feature construction does not change when future rows are removed;
- Regime A lag features do not use future target observations.

A small GitHub Actions workflow is included at `.github/workflows/tests.yml` so the core tests run automatically on pushes and pull requests.

## Notebooks

The notebooks are the executed analysis record and correspond to Parts 1–9 of the assignment. Reusable production-style functions have also been extracted into `src/appliance_energy/`; the notebooks are intentionally retained because they contain the full exploratory narrative, plots and diagnostics used in the report.

## Report

The submitted eight-page report is stored in:

```text
reports/Assignment_2_Report_8_PAGE_COMPACT.docx
```

It contains the final methodology, figures, common model comparison, critical discussion, leakage analysis and practical model recommendation.

## Reproducibility notes

- Random seed: `42` for stochastic machine-learning models.
- Raw UCI data are downloaded by script and are not committed.
- Large Chronos model weights are downloaded externally and are not committed.
- Paths are relative to the repository root.
- Generated forecasts/metrics are written under `outputs/`.
- The final SARIMA specification used throughout this repository is **SARIMA(2,0,6)(0,1,1,24)**.

## References

- Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house*. Energy and Buildings, 140, 81–97.
- Hyndman, R.J. and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Ansari, A.F. et al. (2024). *Chronos: Learning the Language of Time Series*. Transactions on Machine Learning Research.
