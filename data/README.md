# Data directory

Raw UCI data are intentionally not committed. Run:

```bash
python scripts/download_data.py
```

The script downloads `energydata_complete.csv` into `data/raw/` and writes the complete-hour dataset to `data/processed/energy_hourly.csv`. These generated files are ignored by Git so the repository remains lightweight and reproducible.
