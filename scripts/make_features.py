from appliance_energy.data import load_hourly
from appliance_energy.features import build_feature_table
from appliance_energy.config import PROCESSED_DIR

if __name__ == "__main__":
    hourly = load_hourly()
    for regime in ("A", "B"):
        table = build_feature_table(hourly, regime=regime)
        path = PROCESSED_DIR / f"features_regime_{regime.lower()}.csv"
        table.to_csv(path, index_label="date")
        print(f"Regime {regime}: {table.shape} -> {path}")
