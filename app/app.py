"""
Urban Bike Demand Forecasting - ML Capstone (23CSE301)
Streamlit GUI: live predictions from the Regression, Classification and Clustering models.

Run:  streamlit run app/app.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import results_data as RD
from utils import (WEATHER_TYPES, derive_calendar, load_data, load_models, predict_classification,
                   predict_cluster, predict_regression)

st.set_page_config(page_title="Bike Demand Studio", page_icon="", layout="wide",
                   initial_sidebar_state="expanded")

TEAL, ORANGE, BLUE, SLATE, RED = "#0F766E", "#F59E0B", "#2563EB", "#64748B", "#DC2626"
CLUSTER_COLORS = ["#2563EB", "#F59E0B", "#0F766E", "#7C3AED"]  # colour-blind friendly

# ----------------------------------------------------------------------------- #
# Styling
# ----------------------------------------------------------------------------- #
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#0B1F33 0%,#102A43 100%);}
[data-testid="stSidebar"] * {color: #E6EEF5 !important;}
[data-testid="stSidebar"] h2 {font-size: 1.65rem; font-weight: 700; letter-spacing: .02em;}
[data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.15);}
.st-key-classification_best_model [data-testid="stMetricValue"] {font-size: 1.65rem;}
.hero {background: linear-gradient(120deg,#0B1F33 0%,#0F766E 100%); color: #fff; padding: 30px 36px;
       border-radius: 18px; margin-bottom: 20px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2.1rem; color:#fff;}
.hero p {margin: 0; font-size: 1.02rem; opacity: .92; max-width: 860px;}
.pill {display:inline-block; padding: 3px 12px; margin: 10px 8px 0 0; border-radius: 999px;
       background: rgba(255,255,255,.16); font-size: .8rem;}
.card {background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding: 18px 22px;
       box-shadow: 0 1px 3px rgba(15,23,42,.05); min-height: 172px;}
.card h4 {margin: 0 0 4px 0; font-size: .85rem; letter-spacing:.04em; text-transform: uppercase; color:#64748B;}
.big {font-size: 2.3rem; font-weight: 700; line-height: 1.15; margin: 2px 0;}
.sub {color:#64748B; font-size:.88rem;}
.tag {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;}
.step {background:#fff; border-left: 5px solid #0F766E; border-radius: 10px; padding: 12px 16px;
       margin-bottom: 10px; border-top:1px solid #E2E8F0; border-right:1px solid #E2E8F0; border-bottom:1px solid #E2E8F0;}
div[data-testid="stMetric"] {background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding: 12px 16px;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

reg, cls_pipe, clu, cls_meta = load_models()


def card(title, body, sub="", color="#0F172A"):
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="big" style="color:{color}">{body}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def style_fig(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      font=dict(family="sans-serif", size=13), title_font_size=15,
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ----------------------------------------------------------------------------- #
# Sidebar navigation
# ----------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## UrbanPulse")
    
    page = st.radio("Navigate", [" Overview", " Live Predictor", " Model Results", " Data Explorer"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset**  \nCapital Bikeshare, hourly  \n2018-01 → 2021-08  \n33,379 records · 16 columns")
    st.markdown("**Models in this app**  \n• Random Forest (regression)  \n• Tuned Gradient Boosting (classification)  \n• K-Means k=4 (clustering)")
    st.markdown("---")
    st.caption("")


# ----------------------------------------------------------------------------- #
# 1. OVERVIEW
# ----------------------------------------------------------------------------- #
def page_overview():
    st.markdown("""
    <div class="hero">
      <h1>Urban Bike Demand Forecasting</h1>
      <p>Explore hourly bike rental demand through <b>regression</b>, <b>classify hours</b> into low- and high-demand periods, and <b>discover weather patterns</b> using clustering.</p>
      <span class="pill">Regression</span><span class="pill">Classification</span><span class="pill">Clustering</span>
     
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Regression · best R²", "0.485", "Random Forest · RMSE 241.9 rides", TEAL)
    with c2: card("Classification · best F1", "0.946", "Tuned Gradient Boosting · AUC 0.986", BLUE)
    with c3: card("Clustering · K-Means", "k = 4", "Silhouette 0.132 · DB 1.81", ORANGE)
    with c4: card("Algorithms compared", "22", "10 regression · 10 classification · 2 clustering", "#7C3AED")

    st.markdown("&nbsp;")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Problem statement")
        st.write("Predict the bike rental demand (`count`) for an urban bike-sharing system using weather conditions, "
                 "temperature, humidity, wind, cloud cover, day type and time-related patterns. The objective is to "
                 "estimate the number of rentals accurately and identify the model with the best predictive performance.")
        st.subheader("Three tracks")
        st.markdown("""
        <div class="step"><b> Regression</b> – predict the exact hourly rental <i>count</i>. Ten algorithms compared; Random Forest wins (R² 0.485).</div>
        <div class="step"><b> Classification</b> – label an hour <i>Low</i> or <i>High</i> demand (split at the median of 236 rides). Ten algorithms; tuned Gradient Boosting wins (F1 0.946).</div>
        <div class="step"><b> Clustering</b> – without looking at demand, group hours into weather regimes: rainy &amp; mild, cold &amp; dry, warm &amp; dry, snowy &amp; freezing.</div>
        """, unsafe_allow_html=True)
    with right:
        st.subheader("Pipeline")
        st.markdown("""
        <div class="step"><b>1 · Audit &amp; EDA</b> Dataset shape, data types, missing values, distributions and correlations</div>
        <div class="step"><b>2 · Clean</b> Handle missing precipitation values, inspect outliers and process rare weather categories</div>
        <div class="step"><b>3 · Feature Engineering</b> – Create useful time-based features to capture hourly, daily and seasonal bike rental patterns.</div>
        <div class="step"><b>4 · Split &amp; scale</b> 80/20 train-test split, appropriate encoding and scaling fitted on training data only</div>
        <div class="step"><b>5 · Model &amp; Evaluate</b> Train and compare algorithms, tune selected models and evaluate performance using appropriate metrics and cross-validation</div>
        <div class="step"><b>6 · Deploy</b> – Load saved models into the interactive UrbanPulse application</div>
        """, unsafe_allow_html=True)
    st.info(" Open **Live Predictor** in the sidebar, choose a date and weather, and see all three models respond.")


# ----------------------------------------------------------------------------- #
# 2. LIVE PREDICTOR
# ----------------------------------------------------------------------------- #
PRESETS = {
    " Sunny summer commute": dict(date=dt.date(2021, 6, 15), hour=8, temp=26.0, humidity=45, wind_speed=2.5, wind_deg=200,
                                     clouds_all=5, pressure=1016, rain_1h=0.0, snow_1h=0.0, weather="Clear", holiday=False),
    " Rainy autumn evening": dict(date=dt.date(2020, 10, 14), hour=18, temp=13.0, humidity=90, wind_speed=4.0, wind_deg=90,
                                    clouds_all=100, pressure=1008, rain_1h=3.0, snow_1h=0.0, weather="Rain", holiday=False),
    " Snowy winter night": dict(date=dt.date(2021, 1, 20), hour=23, temp=-2.0, humidity=88, wind_speed=3.0, wind_deg=330,
                                  clouds_all=95, pressure=1022, rain_1h=0.0, snow_1h=2.0, weather="Snow", holiday=False),
    " Holiday afternoon": dict(date=dt.date(2019, 7, 4), hour=14, temp=30.0, humidity=55, wind_speed=3.0, wind_deg=180,
                                 clouds_all=20, pressure=1014, rain_1h=0.0, snow_1h=0.0, weather="Clear", holiday=True),
}
DEFAULTS = PRESETS[" Sunny summer commute"]


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[f"in_{k}"] = v


def page_predictor():
    st.markdown("##  Live Predictor")
    st.caption("Describe an hour in the city. All three trained models respond instantly.")
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(f"in_{k}", v)

    with st.container(border=True):
        st.markdown("**Quick scenarios**")
        cols = st.columns(len(PRESETS))
        for col, name in zip(cols, PRESETS):
            col.button(name, on_click=apply_preset, args=(name,), width="stretch")

        a, b, c = st.columns(3)
        with a:
            st.markdown("#####  When")
            date = st.date_input("Date", key="in_date", min_value=dt.date(2018, 1, 1), max_value=dt.date(2026, 12, 31))
            hour = st.slider("Hour of day", 0, 23, key="in_hour", format="%d:00")
            holiday = st.checkbox("Public holiday", key="in_holiday")
        with b:
            st.markdown("#####  Atmosphere")
            temp = st.slider("Temperature (°C)", -15.0, 40.0, step=0.5, key="in_temp")
            humidity = st.slider("Humidity (%)", 10, 100, key="in_humidity")
            pressure = st.slider("Pressure (hPa)", 980, 1045, key="in_pressure")
            weather = st.selectbox("Weather condition", WEATHER_TYPES, key="in_weather")
        with c:
            st.markdown("#####  Wind, cloud & precipitation")
            wind_speed = st.slider("Wind speed (m/s)", 0.0, 15.0, step=0.5, key="in_wind_speed")
            wind_deg = st.slider("Wind direction (°)", 0, 360, key="in_wind_deg")
            clouds = st.slider("Cloud cover (%)", 0, 100, key="in_clouds_all")
            r1, r2 = st.columns(2)
            rain = r1.number_input("Rain (mm/h)", 0.0, 50.0, step=0.5, key="in_rain_1h")
            snow = r2.number_input("Snow (mm/h)", 0.0, 30.0, step=0.5, key="in_snow_1h")

    s = derive_calendar(dict(date=date, hour=hour, holiday=holiday, temp=temp, humidity=humidity, pressure=pressure,
                             weather=weather, wind_speed=wind_speed, wind_deg=wind_deg, clouds_all=clouds,
                             rain_1h=rain, snow_1h=snow))
    day_type = "Holiday" if s["holiday"] else ("Weekend" if s["is_weekend"] else "Working day")
    st.caption(f" {pd.Timestamp(date):%A, %d %B %Y} · {hour:02d}:00 · detected day type: **{day_type}**")

    # ---------------- predictions ----------------
    pred_rides = predict_regression(reg, s)
    p_high = predict_classification(cls_pipe, s)
    label, closeness, xy = predict_cluster(clu, reg, s)
    cname, cemoji = clu["labels"][label]
    is_high = p_high >= 0.5

    st.markdown("&nbsp;")
    r1c, r2c, r3c = st.columns(3)
    with r1c:
        card(" Regression · expected rentals", f"{pred_rides:,.0f} <span style='font-size:1rem'>bikes / hour</span>",
             f"Random Forest · typical error ±{reg['metrics']['MAE']:.0f} bikes", TEAL)
    with r2c:
        tagc = ("#DCFCE7", "#166534") if is_high else ("#FEE2E2", "#991B1B")
        card(" Classification · demand level",
             f"{'HIGH' if is_high else 'LOW'} <span class='tag' style='background:{tagc[0]};color:{tagc[1]}'>"
             f"{max(p_high, 1 - p_high) * 100:.0f}% confident</span>",
             f"Gradient Boosting · threshold = {cls_meta['median']:.0f} rides/hour", BLUE)
    with r3c:
        card(" Clustering · weather regime", f"{cemoji} {cname}",
             f"K-Means · cluster {label} · typical demand {clu['summary'].loc[label, 'mean_demand']:.0f} bikes/h",
             CLUSTER_COLORS[label])

    st.markdown("&nbsp;")
    t1, t2, t3, t4 = st.tabs([" Gauges & 24-hour outlook", " What drives the prediction?",
                              " Weather regime map", " Model inputs"])

    # ---- tab 1
    with t1:
        g1, g2 = st.columns(2)
        with g1:
            top = float(max(900, pred_rides * 1.3))
            fig = go.Figure(go.Indicator(mode="gauge+number", value=pred_rides, number=dict(suffix=" bikes", font=dict(color="#0F172A")),
                                         title=dict(text="Expected rentals this hour"),
                                         gauge=dict(axis=dict(range=[0, top]), bar=dict(color=TEAL),
                                                    steps=[dict(range=[0, cls_meta["median"]], color="#E2E8F0"),
                                                           dict(range=[cls_meta["median"], top], color="#CCFBF1")],
                                                    threshold=dict(line=dict(color=ORANGE, width=4), value=cls_meta["median"]))))
            fig.update_layout(margin=dict(t=90, b=10)); st.plotly_chart(style_fig(fig, 320).update_layout(margin=dict(l=10,r=10,t=90,b=10)), width="stretch")
            st.caption("Orange line = median demand (236 rides/hour), the Low/High cut-off used in classification.")
        with g2:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=p_high * 100, number=dict(suffix="%", font=dict(color="#0F172A")),
                                         title=dict(text="Probability of HIGH demand"),
                                         gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=BLUE),
                                                    steps=[dict(range=[0, 50], color="#E2E8F0"), dict(range=[50, 100], color="#DBEAFE")],
                                                    threshold=dict(line=dict(color=ORANGE, width=4), value=50))))
            fig.update_layout(margin=dict(t=90, b=10)); st.plotly_chart(style_fig(fig, 320).update_layout(margin=dict(l=10,r=10,t=90,b=10)), width="stretch")

        hours = list(range(24))
        sc = []
        for h in hours:
            sh = dict(s, hour=h)
            sc.append((predict_regression(reg, sh), predict_classification(cls_pipe, sh)))
        sc = np.array(sc)
        o1, o2 = st.columns(2)
        with o1:
            fig = go.Figure(go.Scatter(x=hours, y=sc[:, 0], mode="lines+markers", line=dict(color=TEAL, width=3, shape="hv"),
                                       name="Expected rentals"))
            fig.add_vline(x=hour, line_dash="dash", line_color=ORANGE)
            fig.update_layout(title="Regression: expected rentals across the day", xaxis_title="Hour of day",
                              yaxis_title="Predicted rentals")
            st.plotly_chart(style_fig(fig), width="stretch")
            st.caption("The regression model sees time only through the rush-hour flag (7-9 h, 17-19 h), so it steps up at commute times.")
        with o2:
            fig = go.Figure(go.Scatter(x=hours, y=sc[:, 1] * 100, mode="lines+markers", fill="tozeroy",
                                       line=dict(color=BLUE, width=3), name="P(High)"))
            fig.add_hline(y=50, line_dash="dot", line_color=SLATE)
            fig.add_vline(x=hour, line_dash="dash", line_color=ORANGE)
            fig.update_layout(title="Classification: probability of high demand across the day", xaxis_title="Hour of day",
                              yaxis_title="P(High demand) %", yaxis_range=[0, 100])
            st.plotly_chart(style_fig(fig), width="stretch")
            st.caption("Hour, cyclical hour and temperature drive this model, so the daily commute rhythm shows clearly.")

        temps = np.arange(-15, 41, 2.0)
        tr = [predict_regression(reg, dict(s, temp=t)) for t in temps]
        tc = [predict_classification(cls_pipe, dict(s, temp=t)) * 100 for t in temps]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temps, y=tr, name="Expected rentals (left)", line=dict(color=TEAL, width=3)))
        fig.add_trace(go.Scatter(x=temps, y=tc, name="P(High) % (right)", yaxis="y2", line=dict(color=BLUE, width=3, dash="dot")))
        fig.add_vline(x=temp, line_dash="dash", line_color=ORANGE)
        fig.update_layout(title="Temperature sensitivity (everything else held fixed)", xaxis_title="Temperature (°C)",
                          yaxis=dict(title="Expected rentals"), yaxis2=dict(title="P(High) %", overlaying="y", side="right", range=[0, 100]))
        st.plotly_chart(style_fig(fig), width="stretch")

    # ---- tab 2
    with t2:
        i1, i2 = st.columns(2)
        with i1:
            imp = pd.Series(reg["importances"]).sort_values().tail(12)
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[TEAL],
                         title="Regression · Random Forest feature importance")
            fig.update_layout(xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(style_fig(fig, 420), width="stretch")
        with i2:
            imp = pd.Series(cls_meta["importances"]).sort_values().tail(12)
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[BLUE],
                         title="Classification · Gradient Boosting feature importance")
            fig.update_layout(xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(style_fig(fig, 420), width="stretch")
        st.markdown("**Reading the charts.** For regression, temperature-related columns and the rush-hour flag lead. "
                    "For classification, `hour` dominates (~0.42) followed by `temp` and `hour_cos`: *when* matters more than weather.")
        yt, yp = reg["y_test"], reg["y_pred"]
        fig = go.Figure(go.Scatter(x=yt, y=yp, mode="markers", marker=dict(color=TEAL, opacity=.35, size=5), name="Test hours"))
        lim = float(max(yt.max(), yp.max()))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=ORANGE, dash="dash"), name="Perfect prediction"))
        fig.update_layout(title="Regression quality: predicted vs actual (test sample)", xaxis_title="Actual rentals", yaxis_title="Predicted rentals")
        st.plotly_chart(style_fig(fig, 400), width="stretch")

    # ---- tab 3
    with t3:
        m1, m2 = st.columns([1.5, 1])
        with m1:
            P, L = clu["map_xy"], clu["map_lab"]
            fig = go.Figure()
            for k in range(4):
                nm, em = clu["labels"][k]
                fig.add_trace(go.Scatter(x=P[L == k, 0], y=P[L == k, 1], mode="markers", name=f"{em} {nm}",
                                         marker=dict(size=5, opacity=.45, color=CLUSTER_COLORS[k])))
            fig.add_trace(go.Scatter(x=[xy[0]], y=[xy[1]], mode="markers", name="Your scenario",
                                     marker=dict(size=18, color="#0F172A", symbol="star", line=dict(color="white", width=2))))
            ev = clu["pca"].explained_variance_ratio_ * 100
            fig.update_layout(title="PCA 2-D map of weather regimes", xaxis_title=f"PC1 ({ev[0]:.1f}% variance)",
                              yaxis_title=f"PC2 ({ev[1]:.1f}% variance)")
            st.plotly_chart(style_fig(fig, 460), width="stretch")
        with m2:
            st.markdown(f"#### {cemoji} {cname}")
            row = clu["summary"].loc[label]
            st.markdown(f"- Typical temperature: **{row['temp']:.1f} °C**\n- Humidity: **{row['humidity']:.0f}%**\n"
                        f"- Hours with rain or snow: **{row['is_precip'] * 100:.0f}%**\n"
                        f"- Mean demand: **{row['mean_demand']:.0f}** bikes/hour\n"
                        f"- Low-demand hours: **{row['low_demand_pct']:.0f}%**\n- Share of all hours: **{row['share_pct']:.1f}%**")
            st.markdown("**Closeness to each regime**")
            for k in np.argsort(-closeness):
                nm, em = clu["labels"][int(k)]
                st.progress(float(closeness[k]), text=f"{em} {nm} · {closeness[k] * 100:.0f}%")
        st.caption("Clusters are built from weather and time only; demand was never used for fitting, only to interpret the groups afterwards.")

    # ---- tab 4
    with t4:
        st.write("Values sent to the models after feature engineering:")
        feels = float(reg["feels_helper"].predict(pd.DataFrame([[temp, humidity, wind_speed]], columns=["temp", "humidity", "wind_speed"]))[0])
        info = {"datetime": f"{s['datetime']:%Y-%m-%d %H:00}", "day type": day_type, "is_rush_hour (regression)": int((7 <= hour <= 9) or (17 <= hour <= 19)),
                "is_rush_hour (classification)": int(hour in (7, 8, 9, 16, 17, 18, 19) and s["workingday"] == 1),
                "hour_sin / hour_cos": f"{np.sin(2*np.pi*hour/24):.3f} / {np.cos(2*np.pi*hour/24):.3f}",
                "estimated feels-like (°C)": round(feels, 1), "weather": weather,
                "weather used by classifier": "Other" if weather in ("Squall", "Smoke", "Haze") else weather}
        st.dataframe(pd.DataFrame(info.items(), columns=["Feature", "Value"]).astype(str), hide_index=True, width="stretch")
        st.caption("feels-like, min and max temperature are not asked from the user: feels-like is estimated from temperature, humidity and wind; min/max = temperature ∓ 1 °C.")


# ----------------------------------------------------------------------------- #
# 3. MODEL RESULTS
# ----------------------------------------------------------------------------- #
def page_results():
    st.markdown("##  Model Results")
    st.caption("All metrics are on the same held-out 20% test split (random_state = 42) as in the notebooks.")
    t1, t2, t3 = st.tabs([" Regression (10)", " Classification (10 + 2 tuned)", " Clustering (2)"])

    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", "Random Forest"); c2.metric("R²", "0.4852"); c3.metric("5-fold CV R²", "0.4749 ± 0.0166")
        df = RD.REGRESSION
        fig = px.bar(df.sort_values("R²"), x="R²", y="Model", orientation="h", color="R²",
                     color_continuous_scale=["#CBD5E1", TEAL], title="R² by algorithm")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig, 420), width="stretch")
        st.dataframe(df.style.format({"R²": "{:.4f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}).background_gradient(subset=["R²"], cmap="Greens"),
                     hide_index=True, width="stretch")
        st.markdown("**Takeaways.** Non-linear vs. linear models: R² scores are approximately 0.49 vs. 0.29. \n"
                    "Regularisation: Linear, Ridge, Lasso, and ElasticNet have nearly identical results.\n "
                    "Best model: Random Forest, with R² = 0.4852."
                    "Worst model: Decision Tree, with R² = 0.0384."
                    )

    with t2:
        c1, c2, c3 = st.columns(3)
        with c1.container(key="classification_best_model"):
            st.metric("Best model", "Gradient Boosting(tuned)")
        c2.metric("F1 / Accuracy", "0.9458"); c3.metric("ROC-AUC", "0.9856")
        df = RD.CLASSIFICATION
        m = df.melt(id_vars="Algorithm", value_vars=["Accuracy", "F1", "ROC-AUC"], var_name="Metric")
        fig = px.bar(m, x="Algorithm", y="value", color="Metric", barmode="group", color_discrete_sequence=[BLUE, TEAL, ORANGE],
                     title="Accuracy, F1 and ROC-AUC per algorithm")
        fig.update_layout(yaxis_range=[0.7, 1.0], xaxis_tickangle=-35)
        st.plotly_chart(style_fig(fig, 440), width="stretch")
        st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns[3:]}).background_gradient(subset=["F1"], cmap="Blues"),
                     hide_index=True, width="stretch")
        st.markdown(
                    "**Takeaways:** Tuning improved Gradient Boosting’s F1 score from 0.911 to 0.946, making it the best-performing model. Tree-based models performed well, while Naive Bayes had the lowest accuracy among the 12 models. "
                    )

    with t3:
        c1, c2, c3 = st.columns(3)
        c1.metric("Chosen model", "K-Means, k = 4"); c2.metric("Silhouette", "0.132"); c3.metric("Stability (ARI)", "0.903")
        df = RD.CLUSTERING
        st.dataframe(df.style.format({"Silhouette ↑": "{:.3f}", "Davies-Bouldin ↓": "{:.3f}", "Calinski-Harabasz ↑": "{:.1f}", "Stability ARI": "{:.3f}"}),
                     hide_index=True, width="stretch")
        summ = clu["summary"].copy()
        summ["Regime"] = summ["name"]
        out = summ[["Regime", "temp", "humidity", "is_precip", "mean_demand", "low_demand_pct", "share_pct"]].rename(columns={
            "temp": "Temp °C", "humidity": "Humidity %", "is_precip": "Rain/snow share", "mean_demand": "Mean demand",
            "low_demand_pct": "Low-demand hours %", "share_pct": "Share of hours %"})
        out["Rain/snow share"] = out["Rain/snow share"] * 100
        fig = px.bar(out, x="Regime", y="Mean demand", color="Regime", color_discrete_sequence=CLUSTER_COLORS,
                     title="Mean hourly demand by weather regime", text_auto=".0f")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 380), width="stretch")
        st.dataframe(out.style.format("{:.1f}", subset=out.columns[1:]), hide_index=True, width="stretch")
        st.markdown(
                    "**Takeaways:** Warm and dry conditions had the highest bike demand, while snowy and freezing conditions had the lowest. Ward linkage was selected because it grouped weather conditions more meaningfully. The low Silhouette scores indicate that the clusters overlap and are not clearly separated.")


# ----------------------------------------------------------------------------- #
# 4. DATA EXPLORER
# ----------------------------------------------------------------------------- #
def page_explorer():
    st.markdown("##  Data Explorer")
    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}"); c2.metric("Mean rentals / hour", f"{df['count'].mean():.0f}")
    c3.metric("Median", f"{df['count'].median():.0f}"); c4.metric("Peak hour", f"{df['count'].max():,}")

    yrs = st.multiselect("Years", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
    d = df[df["year"].isin(yrs)]
    a, b = st.columns(2)
    with a:
        h = d.assign(Day=np.where(d["is_weekend"] == 1, "Weekend", "Weekday")).groupby(["hour", "Day"])["count"].mean().reset_index()
        fig = px.line(h, x="hour", y="count", color="Day", markers=True, color_discrete_sequence=[TEAL, ORANGE],
                      title="Average rentals by hour: commuter peaks on weekdays")
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Average rentals")
        st.plotly_chart(style_fig(fig), width="stretch")
    with b:
        m = d.groupby("month")["count"].mean().reset_index()
        fig = px.bar(m, x="month", y="count", color_discrete_sequence=[BLUE], title="Average rentals by month: strong seasonality")
        fig.update_layout(xaxis_title="Month", yaxis_title="Average rentals", xaxis=dict(dtick=1))
        st.plotly_chart(style_fig(fig), width="stretch")
    a, b = st.columns(2)
    with a:
        order = d.groupby("weather_main")["count"].median().sort_values(ascending=False).index
        fig = px.box(d, x="weather_main", y="count", category_orders={"weather_main": list(order)}, color_discrete_sequence=[TEAL],
                     title="Rentals by weather condition")
        fig.update_layout(xaxis_title="Weather", yaxis_title="Rentals per hour")
        st.plotly_chart(style_fig(fig), width="stretch")
    with b:
        s = d.sample(min(5000, len(d)), random_state=42)
        fig = px.scatter(s, x="temp", y="count", opacity=.35, color_discrete_sequence=[ORANGE], title="Temperature vs rentals (5,000-row sample)")
        fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Rentals per hour")
        st.plotly_chart(style_fig(fig), width="stretch")
    corr = d[["count", "temp", "humidity", "wind_speed", "clouds_all", "pressure", "rain_1h", "snow_1h", "hour"]].corr()
    fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap")
    st.plotly_chart(style_fig(fig, 480), width="stretch")
    with st.expander("Preview raw data"):
        st.dataframe(d.drop(columns=["is_weekend"]).head(200), width="stretch")


{" Overview": page_overview, " Live Predictor": page_predictor, " Model Results": page_results,
 " Data Explorer": page_explorer}[page]()
