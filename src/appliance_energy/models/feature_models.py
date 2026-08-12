from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from ..config import TARGET, HORIZON, RANDOM_STATE


def hgb_factory():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, random_state=RANDOM_STATE)


def random_forest_factory():
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=0.4,
        random_state=RANDOM_STATE, n_jobs=-1)


def rolling_ml_forecast(table: pd.DataFrame, origin_timestamps: list[pd.Timestamp],
                        model_factory, name: str, target: str = TARGET,
                        horizon: int = HORIZON,
                        feature_subset: list[str] | None = None) -> pd.DataFrame:
    columns = feature_subset or [c for c in table.columns if c != target]
    records = []
    for origin_id, cutoff in enumerate(origin_timestamps, start=1):
        train = table.loc[table.index < cutoff]
        test = table.loc[cutoff:].iloc[:horizon]
        if len(test) != horizon:
            raise ValueError(f"Origin {cutoff} has only {len(test)} test rows")
        model = model_factory()
        model.fit(train[columns], train[target])
        predicted = model.predict(test[columns])
        records.append(pd.DataFrame({
            "origin_id": origin_id, "timestamp": test.index,
            "step": np.arange(1, horizon + 1), "actual": test[target].to_numpy(),
            "prediction": predicted, "model": name,
        }))
    return pd.concat(records, ignore_index=True)
