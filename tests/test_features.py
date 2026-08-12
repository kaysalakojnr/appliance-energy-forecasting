import pandas as pd
from appliance_energy.features import build_feature_table, add_target_lags
from conftest import synthetic_hourly

def test_lag_24_is_exactly_24_hours_old():
    frame = synthetic_hourly()
    lags = add_target_lags(frame)
    t = frame.index[250]
    assert lags.loc[t, "lag_24"] == frame.loc[frame.index[226], "Appliances"]

def test_regime_a_row_is_unchanged_when_future_is_removed():
    frame = synthetic_hourly(600)
    full = build_feature_table(frame, regime="A")
    t = full.index[-30]
    truncated = build_feature_table(frame.loc[:t], regime="A")
    pd.testing.assert_series_equal(full.loc[t], truncated.loc[t], check_names=True)

def test_regime_b_has_future_covariates_but_a_does_not():
    frame = synthetic_hourly()
    a = build_feature_table(frame, regime="A")
    b = build_feature_table(frame, regime="B")
    assert not any(c.endswith("_future") for c in a.columns)
    assert any(c.endswith("_future") for c in b.columns)
