from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

RAW_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00374/energydata_complete.csv")
RAW_FILE = RAW_DIR / "energydata_complete.csv"
HOURLY_FILE = PROCESSED_DIR / "energy_hourly.csv"

TARGET = "Appliances"
NOISE_COLS = ["rv1", "rv2"]
OBS_PER_HOUR = 6
HORIZON = 24
TEST_DAYS = 14
TEST_STEPS = HORIZON * TEST_DAYS
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168
RANDOM_STATE = 42

INDOOR_TEMP = [f"T{i}" for i in range(1, 10)]
INDOOR_RH = [f"RH_{i}" for i in range(1, 10)]
WEATHER_COLS = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
SARIMAX_WEATHER_COLS = ["T_out", "RH_out", "Windspeed", "Tdewpoint", "Press_mm_hg"]

SARIMA_ORDER = (2, 0, 6)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 24)
CHRONOS_MODEL = "amazon/chronos-bolt-base"
CHRONOS_CONTEXT = 512
CHRONOS_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def ensure_directories():
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURE_DIR,
                 FORECAST_DIR, METRICS_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
