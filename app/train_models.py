"""
Re-creates the final model of each capstone track exactly as in the notebooks
and saves it to ../models/ so the Streamlit app can load it.

    python app/train_models.py

Regression     -> Random Forest Regressor (best R2 in regression.ipynb)
Classification -> Tuned Gradient Boosting pipeline (best F1 in classification.ipynb)
Clustering     -> K-Means, k = 4 (clustering.ipynb)
All pipelines use random_state = 42, and scalers/encoders are fitted on the TRAIN split only.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "capitalbikeshare-complete.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# --------------------------------------------------------------------------- #
# 1. REGRESSION  (Random Forest, n_estimators=100)
# --------------------------------------------------------------------------- #
REG_NUMERIC = ["temp", "feels_like", "temp_min", "temp_max", "pressure",
               "humidity", "wind_speed", "wind_deg", "rain_1h", "snow_1h", "clouds_all"]


def train_regression(raw: pd.DataFrame) -> dict:
    df = raw.copy()
    df[["rain_1h", "snow_1h"]] = df[["rain_1h", "snow_1h"]].fillna(0)

    # IQR capping of humidity, wind_speed, clouds_all
    caps = {}
    for c in ["humidity", "wind_speed", "clouds_all"]:
        q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        caps[c] = (float(lo), float(hi))
        df[c] = df[c].clip(lo, hi)

    h = df["datetime"].dt.hour
    df["is_rush_hour"] = (((h >= 7) & (h <= 9)) | ((h >= 17) & (h <= 19))).astype(int)

    enc = pd.get_dummies(df, columns=["weather_main"], prefix="weather", drop_first=True)
    feats = [c for c in enc.columns if c not in ["datetime", "count", "is_rush_hour"]] + ["is_rush_hour"]
    X, y = enc[feats], enc["count"]

    bins = pd.qcut(y, q=4, labels=False)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=RANDOM_STATE, stratify=bins)

    scaler = StandardScaler()
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[REG_NUMERIC] = scaler.fit_transform(X_tr[REG_NUMERIC])
    X_te[REG_NUMERIC] = scaler.transform(X_te[REG_NUMERIC])

    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {"R2": r2_score(y_te, pred), "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
               "MAE": mean_absolute_error(y_te, pred)}
    print("Regression  :", {k: round(v, 4) for k, v in metrics.items()}, "(notebook: R2 0.4852)")

    # tiny helper so the GUI can estimate feels_like from temp / humidity / wind
    helper = LinearRegression().fit(df[["temp", "humidity", "wind_speed"]], df["feels_like"])

    return dict(model=model, scaler=scaler, features=feats, numeric=REG_NUMERIC, caps=caps,
                weather_dummies=[c for c in feats if c.startswith("weather_")],
                feels_helper=helper, metrics=metrics,
                importances=dict(zip(feats, model.feature_importances_)),
                y_test=y_te.to_numpy()[:1500], y_pred=pred[:1500])


# --------------------------------------------------------------------------- #
# 2. CLASSIFICATION  (tuned Gradient Boosting inside a preprocessing Pipeline)
# --------------------------------------------------------------------------- #
def cls_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering identical to classification.ipynb (Section 5)."""
    df = df.copy()
    dt = df["datetime"].dt
    df["hour"], df["day_of_week"], df["month"], df["year"] = dt.hour, dt.dayofweek, dt.month, dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (df["hour"].isin([7, 8, 9, 16, 17, 18, 19]) & (df["workingday"] == 1)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


CLS_COLS = ["holiday", "workingday", "temp", "pressure", "humidity", "wind_speed", "wind_deg",
            "rain_1h", "snow_1h", "clouds_all", "weather_main", "hour", "day_of_week", "month",
            "year", "is_weekend", "is_rush_hour", "hour_sin", "hour_cos"]


def train_classification(raw: pd.DataFrame) -> dict:
    df = raw.copy()
    median_count = float(df["count"].median())
    df["demand"] = (df["count"] > median_count).astype(int)
    df[["rain_1h", "snow_1h"]] = df[["rain_1h", "snow_1h"]].fillna(0)
    df["rain_1h"], df["snow_1h"] = np.log1p(df["rain_1h"]), np.log1p(df["snow_1h"])
    df["weather_main"] = df["weather_main"].replace(["Squall", "Smoke", "Haze"], "Other")
    df = cls_features(df)

    X, y = df[CLS_COLS], df["demand"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)

    cat, binc = ["weather_main"], ["holiday", "workingday", "is_weekend", "is_rush_hour"]
    num = [c for c in X.columns if c not in cat + binc]
    prep = ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                              ("bin", "passthrough", binc)])
    pipe = Pipeline([("prep", prep),
                     ("model", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=7,
                                                          subsample=0.8, random_state=RANDOM_STATE))])
    pipe.fit(X_tr, y_tr)
    f1 = f1_score(y_te, pipe.predict(X_te), average="weighted")
    print("Classification: F1 =", round(f1, 4), "(notebook: 0.9458)")

    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = dict(zip(names, pipe.named_steps["model"].feature_importances_))
    joblib.dump(pipe, MODELS / "final_gradient_boosting.pkl", compress=3)
    return dict(median=median_count, f1=f1, importances=imp)


# --------------------------------------------------------------------------- #
# 3. CLUSTERING  (K-Means, k = 4, on weather + cyclical time features)
# --------------------------------------------------------------------------- #
CLU_FEATURES = ['holiday', 'workingday', 'temp', 'temp_feels_diff', 'pressure', 'humidity', 'wind_speed',
                'wind_sin', 'wind_cos', 'rain_1h', 'snow_1h', 'clouds_all', 'is_precip',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos']


def clu_features(d: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering identical to clustering.ipynb (Section 4)."""
    d = d.copy()
    hr, mo = d["datetime"].dt.hour, d["datetime"].dt.month
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
    d["wind_sin"], d["wind_cos"] = np.sin(np.deg2rad(d["wind_deg"])), np.cos(np.deg2rad(d["wind_deg"]))
    d["is_precip"] = ((d["rain_1h"] > 0) | (d["snow_1h"] > 0)).astype(int)
    d["temp_feels_diff"] = d["feels_like"] - d["temp"]
    return d


def name_cluster(p: pd.Series) -> tuple[str, str]:
    """Human-readable label from the centroid profile (same story as notebook Section 9)."""
    if p["is_precip"] > .5 and p["temp"] < 4:
        return "Snowy & freezing", "❄️"
    if p["is_precip"] > .5:
        return "Rainy & mild", "🌧️"
    if p["temp"] >= 17:
        return "Warm & dry", "☀️"
    return "Cold & dry", "🧥"


def train_clustering(raw: pd.DataFrame) -> dict:
    df = raw.copy()
    median_count = df["count"].median()
    df["count_class"] = (df["count"] > median_count).astype(int)
    d = df.drop_duplicates().sort_values("datetime").reset_index(drop=True)
    d[["rain_1h", "snow_1h"]] = d[["rain_1h", "snow_1h"]].fillna(0)
    caps = {}
    for c in ["wind_speed", "rain_1h", "snow_1h"]:
        caps[c] = float(d[c].quantile(.999))
        d[c] = d[c].clip(upper=caps[c])
    d = clu_features(d)

    X, y = d[CLU_FEATURES], d["count_class"]
    X_tr, X_te, _, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=CLU_FEATURES, index=X_tr.index)
    samp_idx = X_tr_s.sample(8000, random_state=RANDOM_STATE).index
    X_samp = X_tr_s.loc[samp_idx]

    km = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit(X_samp)
    sil = silhouette_score(X_samp, km.labels_)
    print("Clustering  : silhouette =", round(sil, 3), "(notebook: 0.132)")

    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_tr_s)
    prof = d.loc[samp_idx, CLU_FEATURES + ["count", "count_class"]].assign(Cluster=km.labels_)
    summary = prof.groupby("Cluster").agg(temp=("temp", "mean"), humidity=("humidity", "mean"),
                                          wind_speed=("wind_speed", "mean"), clouds=("clouds_all", "mean"),
                                          is_precip=("is_precip", "mean"), mean_demand=("count", "mean"),
                                          median_demand=("count", "median"),
                                          low_demand_pct=("count_class", lambda s: (1 - s.mean()) * 100),
                                          size=("count", "size"))
    summary["share_pct"] = summary["size"] / summary["size"].sum() * 100
    labels = {int(k): name_cluster(r) for k, r in summary.iterrows()}
    summary["name"] = [labels[int(k)][0] for k in summary.index]
    print(summary[["name", "temp", "mean_demand", "share_pct"]].round(1))

    P = pca.transform(X_samp)
    return dict(scaler=scaler, kmeans=km, pca=pca, features=CLU_FEATURES, caps=caps, labels=labels,
                summary=summary, silhouette=sil, map_xy=P[:2000], map_lab=km.labels_[:2000])


if __name__ == "__main__":
    raw = load_raw()
    reg = train_regression(raw)
    joblib.dump(reg, MODELS / "regression_random_forest.pkl", compress=3)
    cls = train_classification(raw)
    joblib.dump(cls, MODELS / "classification_meta.pkl", compress=3)
    clu = train_clustering(raw)
    joblib.dump(clu, MODELS / "clustering_kmeans.pkl", compress=3)
    print("Saved models ->", MODELS)
