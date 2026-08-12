from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .config import RAW_URL, RAW_FILE, HOURLY_FILE, TARGET, NOISE_COLS, OBS_PER_HOUR, ensure_directories


def download_raw(destination: Path = RAW_FILE, url: str = RAW_URL, force: bool = False) -> Path:
    """Download the UCI CSV if it is not already available locally."""
    ensure_directories()
    destination = Path(destination)
    if destination.exists() and not force:
        return destination
    urlretrieve(url, destination)
    return destination


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load raw data with a sorted DatetimeIndex."""
    path = Path(path) if path is not None else download_raw()
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date").sort_index()


def audit_raw(frame: pd.DataFrame) -> dict:
    """Return simple quality-control counts for the 10-minute data."""
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="10min")
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "missing_timestamps": int(len(expected.difference(frame.index))),
        "missing_cells": int(frame.isna().sum().sum()),
    }


def to_hourly(frame: pd.DataFrame, drop_noise: bool = True) -> pd.DataFrame:
    """Aggregate six 10-minute readings to complete hourly means only."""
    work = frame.copy()
    if drop_noise:
        work = work.drop(columns=[c for c in NOISE_COLS if c in work.columns])
    counts = work.resample("h").size()
    hourly = work.resample("h").mean().loc[counts >= OBS_PER_HOUR]
    return hourly.dropna(subset=[TARGET])


def prepare_hourly(raw_path: Path | None = None, output_path: Path = HOURLY_FILE) -> pd.DataFrame:
    """Download/load, audit, hourly-resample and save the modelling dataset."""
    ensure_directories()
    raw = load_raw(raw_path)
    hourly = to_hourly(raw)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path, index_label="date")
    return hourly


def load_hourly(path: Path = HOURLY_FILE, rebuild: bool = True) -> pd.DataFrame:
    """Load the prepared hourly dataset, rebuilding it if requested."""
    path = Path(path)
    if not path.exists():
        if not rebuild:
            raise FileNotFoundError(path)
        return prepare_hourly(output_path=path)
    frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return frame.asfreq("h")
