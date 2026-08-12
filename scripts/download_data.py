from appliance_energy.data import download_raw, prepare_hourly

if __name__ == "__main__":
    raw = download_raw()
    hourly = prepare_hourly(raw)
    print(f"Raw: {raw}")
    print(f"Hourly shape: {hourly.shape}")
