"""Loading of the two public datasets used in the demo.

Marine: UCI "Condition Based Maintenance of Naval Propulsion Plants"
        (Coraddu et al., 2014) - numerical simulator of a CODLAG frigate
        gas-turbine propulsion plant, 11 934 steady-state points.
Energy: Open Power System Data, time_series 2020-10-06 - hourly Finnish load,
        ENTSO-E day-ahead load forecast and onshore wind generation 2015-2020.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

UCI_URL = "https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip"
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

MARINE_COLS = ["lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48", "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf", "kMc", "kMt"]
# T1 and P1 are constant in the dataset; lp is a 1:1 function of v.
MARINE_SENSORS = ["GTT", "GTn", "GGn", "Ts", "Tp", "T48", "T2", "P48", "P2", "Pexh", "TIC", "mf"]
MARINE_FEATURES = ["v"] + MARINE_SENSORS
MARINE_TARGETS = ["kMc", "kMt"]
SENSOR_LABELS = {
    "v": "Ship speed [kn]", "GTT": "GT shaft torque [kN m]", "GTn": "GT speed [rpm]",
    "GGn": "Gas generator speed [rpm]", "Ts": "Stbd propeller torque [kN]", "Tp": "Port propeller torque [kN]",
    "T48": "HP turbine exit temp [C]", "T2": "Compressor outlet temp [C]", "P48": "HP turbine exit pressure [bar]",
    "P2": "Compressor outlet pressure [bar]", "Pexh": "Exhaust pressure [bar]", "TIC": "Turbine injection control [%]",
    "mf": "Fuel flow [kg/s]", "kMc": "Compressor decay coefficient", "kMt": "Turbine decay coefficient",
}


def load_marine() -> pd.DataFrame:
    path = RAW / "UCI CBM Dataset" / "data.txt"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(UCI_URL, timeout=120)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(RAW)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=MARINE_COLS)
    df["kMc"] = df["kMc"].round(3)
    df["kMt"] = df["kMt"].round(3)
    return df


def load_energy() -> pd.DataFrame:
    """Hourly Finnish load / day-ahead load forecast / wind [MW], UTC index."""
    small = RAW / "opsd_fi.csv"
    if not small.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(OPSD_URL, usecols=lambda c: c == "utc_timestamp" or c.startswith("FI_"))
        df = df.dropna(subset=["FI_load_actual_entsoe_transparency"])
        df.to_csv(small, index=False)
    df = pd.read_csv(small, parse_dates=["utc_timestamp"]).set_index("utc_timestamp").sort_index()
    df = df.rename(columns={
        "FI_load_actual_entsoe_transparency": "load",
        "FI_load_forecast_entsoe_transparency": "load_da_forecast",
        "FI_wind_onshore_generation_actual": "wind",
    })
    df = df.asfreq("h")
    df["load_da_forecast"] = df["load_da_forecast"].ffill(limit=3)
    return df
