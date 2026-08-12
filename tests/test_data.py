import numpy as np
import pandas as pd
from appliance_energy.data import to_hourly

def test_hourly_resample_drops_incomplete_terminal_hour():
    idx = pd.date_range("2016-01-01", periods=17, freq="10min")  # 2 complete hours + 5 readings
    frame = pd.DataFrame({"Appliances": np.arange(17), "rv1": 1, "rv2": 2}, index=idx)
    hourly = to_hourly(frame)
    assert hourly.shape == (2, 1)
    assert "rv1" not in hourly.columns and "rv2" not in hourly.columns
    assert not hourly["Appliances"].isna().any()
