"""Model loading and prediction helpers for the Streamlit app."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "capitalbikeshare-complete.csv"

WEATHER_TYPES = ["Clear", "Clouds", "Mist", "Rain", "Drizzle", "Fog", "Snow", "Thunderstorm", "Haze", "Smoke", "Squall"]
RARE_FOR_CLASSIFIER = ["Squall", "Smoke", "Haze"]


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load the three saved models; train them first if the .pkl files are missing."""
    needed = ["regression_random_forest.pkl", "final_gradient_boosting.pkl", "clustering_kmeans.pkl"]
    if not all((MODELS / f).exists() for f in needed):
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).with_name("train_models.py"))], check=True)
    return (joblib.load(MODELS / needed[0]), joblib.load(MODELS / needed[1]),
            joblib.load(MODELS / "clustering_kmeans.pkl"), joblib.load(MODELS / "classification_meta.pkl"))


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df[["rain_1h", "snow_1h"]] = df[["rain_1h", "snow_1h"]].fillna(0)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["is_weekend"] = (df["datetime"].dt.dayofweek >= 5).astype(int)
    return df


def derive_calendar(inp: dict) -> dict:
    """Add hour / month / year / working-day flags from the chosen date."""
    d = pd.Timestamp(inp["date"])
    out = dict(inp)
    out.update(hour=int(inp["hour"]), month=d.month, year=d.year, day_of_week=d.dayofweek,
               is_weekend=int(d.dayofweek >= 5), datetime=d.replace(hour=int(inp["hour"])),
               workingday=int(d.dayofweek < 5 and not inp["holiday"]), holiday=int(inp["holiday"]))
    return out


# ----------------------------- regression ---------------------------------- #
def predict_regression(reg: dict, s: dict) -> float:
    feels = float(reg["feels_helper"].predict(pd.DataFrame(
        [[s["temp"], s["humidity"], s["wind_speed"]]], columns=["temp", "humidity", "wind_speed"]))[0])
    h = s["hour"]
    row = dict(holiday=s["holiday"], workingday=s["workingday"], temp=s["temp"], feels_like=feels,
               temp_min=s["temp"] - 1.0, temp_max=s["temp"] + 1.0, pressure=s["pressure"],
               humidity=float(np.clip(s["humidity"], *reg["caps"]["humidity"])),
               wind_speed=float(np.clip(s["wind_speed"], *reg["caps"]["wind_speed"])),
               wind_deg=s["wind_deg"], rain_1h=s["rain_1h"], snow_1h=s["snow_1h"],
               clouds_all=float(np.clip(s["clouds_all"], *reg["caps"]["clouds_all"])),
               is_rush_hour=int((7 <= h <= 9) or (17 <= h <= 19)))
    for c in reg["weather_dummies"]:
        row[c] = int(c == f"weather_{s['weather']}")
    X = pd.DataFrame([row])[reg["features"]].astype(float)
    X[reg["numeric"]] = reg["scaler"].transform(X[reg["numeric"]])
    return max(0.0, float(reg["model"].predict(X)[0]))


# --------------------------- classification -------------------------------- #
def predict_classification(pipe, s: dict) -> float:
    """Returns P(High demand)."""
    h = s["hour"]
    w = "Other" if s["weather"] in RARE_FOR_CLASSIFIER else s["weather"]
    row = dict(holiday=s["holiday"], workingday=s["workingday"], temp=s["temp"], pressure=s["pressure"],
               humidity=s["humidity"], wind_speed=s["wind_speed"], wind_deg=s["wind_deg"],
               rain_1h=np.log1p(s["rain_1h"]), snow_1h=np.log1p(s["snow_1h"]), clouds_all=s["clouds_all"],
               weather_main=w, hour=h, day_of_week=s["day_of_week"], month=s["month"], year=s["year"],
               is_weekend=s["is_weekend"],
               is_rush_hour=int(h in (7, 8, 9, 16, 17, 18, 19) and s["workingday"] == 1),
               hour_sin=np.sin(2 * np.pi * h / 24), hour_cos=np.cos(2 * np.pi * h / 24))
    return float(pipe.predict_proba(pd.DataFrame([row]))[0, 1])


# ------------------------------ clustering --------------------------------- #
def predict_cluster(clu: dict, reg: dict, s: dict):
    feels = float(reg["feels_helper"].predict(pd.DataFrame(
        [[s["temp"], s["humidity"], s["wind_speed"]]], columns=["temp", "humidity", "wind_speed"]))[0])
    h, m = s["hour"], s["month"]
    rain = min(s["rain_1h"], clu["caps"]["rain_1h"]); snow = min(s["snow_1h"], clu["caps"]["snow_1h"])
    row = dict(holiday=s["holiday"], workingday=s["workingday"], temp=s["temp"], temp_feels_diff=feels - s["temp"],
               pressure=s["pressure"], humidity=s["humidity"], wind_speed=min(s["wind_speed"], clu["caps"]["wind_speed"]),
               wind_sin=np.sin(np.deg2rad(s["wind_deg"])), wind_cos=np.cos(np.deg2rad(s["wind_deg"])),
               rain_1h=rain, snow_1h=snow, clouds_all=s["clouds_all"], is_precip=int(rain > 0 or snow > 0),
               hour_sin=np.sin(2 * np.pi * h / 24), hour_cos=np.cos(2 * np.pi * h / 24),
               month_sin=np.sin(2 * np.pi * m / 12), month_cos=np.cos(2 * np.pi * m / 12))
    X = pd.DataFrame([row])[clu["features"]]
    Xs = pd.DataFrame(clu["scaler"].transform(X), columns=clu["features"])
    label = int(clu["kmeans"].predict(Xs)[0])
    dist = clu["kmeans"].transform(Xs)[0]
    closeness = (1 / dist) / (1 / dist).sum()
    xy = clu["pca"].transform(Xs)[0]
    return label, closeness, xy
