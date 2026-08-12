from __future__ import annotations

import warnings
import pandas as pd

from .config import (TARGET, TEST_STEPS, DAILY_PERIOD, METRICS_DIR, FORECAST_DIR,
                     FIGURE_DIR, PROCESSED_DIR, ensure_directories)
from .data import prepare_hourly
from .evaluation import make_origins, mase_scale, score_forecast_frame
from .features import build_feature_table
from .models.benchmarks import rolling_benchmark_forecasts
from .models.sarimax import (make_exogenous_frames, rolling_sarimax_forecast,
                             grid_search_nonseasonal, grid_search_seasonal)
from .models.feature_models import hgb_factory, random_forest_factory, rolling_ml_forecast
from .models.foundation import rolling_chronos_forecast
from .plotting import plot_model_mae, plot_forecast_comparison


def run_pipeline(skip_foundation: bool = False, full_sarimax_grid: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the reproducible assignment pipeline and save outputs."""
    ensure_directories()
    hourly = prepare_hourly()
    series = hourly[TARGET]
    origins = make_origins(series.index)
    train = series.iloc[:-TEST_STEPS]
    scale = mase_scale(train, DAILY_PERIOD)

    frames = [rolling_benchmark_forecasts(series, origins)]

    if full_sarimax_grid:
        nonseasonal = grid_search_nonseasonal(train)
        nonseasonal.to_csv(METRICS_DIR / "sarima_grid_nonseasonal.csv", index=False)
        top_orders = [tuple(map(int, row)) for row in nonseasonal.loc[nonseasonal["converged"], ["p", "d", "q"]].head(3).to_numpy()]
        seasonal = grid_search_seasonal(train, top_orders)
        seasonal.to_csv(METRICS_DIR / "sarima_grid_seasonal.csv", index=False)

    for name, exog in make_exogenous_frames(hourly).items():
        frames.append(rolling_sarimax_forecast(series, origins, name, exog))

    features_a = build_feature_table(hourly, regime="A")
    features_b = build_feature_table(hourly, regime="B")
    features_a.to_csv(PROCESSED_DIR / "features_regime_a.csv", index_label="date")
    features_b.to_csv(PROCESSED_DIR / "features_regime_b.csv", index_label="date")
    cutoff_times = [origin["first_forecast"] for origin in origins]

    frames.append(rolling_ml_forecast(features_a, cutoff_times, hgb_factory,
                                      "hist_gradient_boosting"))
    frames.append(rolling_ml_forecast(features_a, cutoff_times, random_forest_factory,
                                      "random_forest"))
    frames.append(rolling_ml_forecast(features_b, cutoff_times, hgb_factory,
                                      "hist_gradient_boosting_regimeB"))

    if not skip_foundation:
        try:
            frames.append(rolling_chronos_forecast(series, origins))
        except Exception as exc:
            warnings.warn(f"Foundation model skipped because it could not run: {exc}")

    forecasts = pd.concat(frames, ignore_index=True, sort=False)
    forecasts.to_csv(FORECAST_DIR / "all_forecasts.csv", index=False)
    scores = score_forecast_frame(forecasts, scale)
    scores.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    plot_model_mae(scores, FIGURE_DIR / "model_mae.png")
    plot_forecast_comparison(forecasts, FIGURE_DIR / "forecast_comparison.png")
    return forecasts, scores
