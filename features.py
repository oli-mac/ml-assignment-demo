"""Feature engineering shared by the deployed app.

These functions must stay identical to the ones in the two notebooks
(regression: `add_features`, classification: `add_features`), because the saved
pipelines expect the engineered columns to exist.
"""
import numpy as np
import pandas as pd


def gas_turbine_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["TEMP_DROP"] = d["TIT"] - d["TAT"]
    d["PRESSURE_RATIO"] = d["CDP"] * 1000 / d["AP"]
    d["POWER_PER_EXHAUST_P"] = d["TEY"] / d["GTEP"]
    d["AIR_DENSITY_IDX"] = d["AP"] / (d["AT"] + 273.15)
    return d


def intrusion_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["bytes_ratio"] = np.log1p(d["sbytes"]) - np.log1p(d["dbytes"])
    d["pkts_ratio"] = np.log1p(d["spkts"]) - np.log1p(d["dpkts"])
    d["bytes_per_sec"] = (d["sbytes"] + d["dbytes"]) / (d["dur"] + 1e-6)
    d["is_one_way"] = (d["dbytes"] == 0).astype(int)
    d["service_known"] = (d["service"] != "-").astype(int)
    return d
