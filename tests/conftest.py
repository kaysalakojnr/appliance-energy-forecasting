import numpy as np
import pandas as pd

def synthetic_hourly(n=500):
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    frame = pd.DataFrame(index=idx)
    frame["Appliances"] = 80 + 20*np.sin(2*np.pi*idx.hour/24) + np.arange(n)*0.01
    frame["lights"] = 5.0
    for i in range(1, 10):
        frame[f"T{i}"] = 20 + i*0.1 + np.sin(np.arange(n)/48)
        frame[f"RH_{i}"] = 40 + i*0.2 + np.cos(np.arange(n)/48)
    frame["T_out"] = 10 + np.sin(np.arange(n)/72)
    frame["Press_mm_hg"] = 755.0
    frame["RH_out"] = 60.0
    frame["Windspeed"] = 3.0
    frame["Visibility"] = 40.0
    frame["Tdewpoint"] = 5.0
    return frame
