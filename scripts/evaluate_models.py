from pathlib import Path
import pandas as pd
from appliance_energy.config import FORECAST_DIR, METRICS_DIR, TARGET, TEST_STEPS, DAILY_PERIOD
from appliance_energy.data import load_hourly
from appliance_energy.evaluation import mase_scale, score_forecast_frame

if __name__ == "__main__":
    path = FORECAST_DIR / "all_forecasts.csv"
    if not path.exists():
        raise SystemExit("Run scripts/run_pipeline.py first.")
    forecasts = pd.read_csv(path, parse_dates=["timestamp"])
    hourly = load_hourly()
    scale = mase_scale(hourly[TARGET].iloc[:-TEST_STEPS], DAILY_PERIOD)
    scores = score_forecast_frame(forecasts, scale)
    scores.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(scores.round(3).to_string(index=False))
