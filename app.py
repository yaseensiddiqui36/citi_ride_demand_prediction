import sys
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from datetime import timedelta

from src.inference import (
    get_feature_store,
    load_model_from_registry,
    get_model_predictions,
)
from src.data_utils import transform_ts_data_info_features
from src.plot_utils import plot_prediction
import src.config as config

st.set_page_config(layout="wide")

# Initialization
current_date = pd.Timestamp("2025-04-30 00:00:00", tz="America/New_York")
st.title("🚲 NYC Citi Bike Demand Predictor")
st.header(f"Predictions for: {current_date.strftime('%Y-%m-%d %H:%M:%S')}" )

# Progress bar steps
progress_bar = st.sidebar.header("Prediction Pipeline Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 4

with st.spinner("Fetching latest features and generating predictions..."):
    fs = get_feature_store()
    fg = fs.get_feature_group(name="citibike_hourly_features", version=1)
    hist = fg.read()
    hist["pickup_hour"] = pd.to_datetime(hist["pickup_hour"], utc=True)

    latest_hr = pd.Timestamp("2025-04-30 00:00:00", tz="UTC")
    window_size = 24 * 28
    fetch_from = latest_hr - pd.Timedelta(hours=window_size + 1)
    fetch_to = latest_hr

    fv = fs.get_feature_view(name="citibike_hourly_features_view", version=1)
    ts = (
        fv.get_batch_data(start_time=fetch_from, end_time=fetch_to)
        .loc[lambda df: df.pickup_hour.between(fetch_from, fetch_to)]
        .sort_values(["pickup_location_id", "pickup_hour"])
    )

    features = transform_ts_data_info_features(ts, feature_col="rides", window_size=window_size, step_size=1)
    features["target"] = 0  # dummy column

    model = load_model_from_registry()
    predictions = get_model_predictions(model, features)
    predictions = predictions.rename(columns={"predicted_demand": "predicted_rides"})
    predictions["pickup_hour"] = latest_hr + pd.Timedelta(hours=1)
    progress_bar.progress(1 / N_STEPS)
    st.sidebar.success("Predictions ready")

# Metrics and top-10 view
st.subheader("📊 Demand Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Rides", f"{predictions['predicted_rides'].mean():.0f}")
col2.metric("Peak Rides", f"{predictions['predicted_rides'].max():.0f}")
col3.metric("Minimum Rides", f"{predictions['predicted_rides'].min():.0f}")

st.subheader("🔟 Top 10 Most Demanded Locations")
st.dataframe(predictions.sort_values("predicted_rides", ascending=False).head(10))

progress_bar.progress(2 / N_STEPS)

# Dropdown for selecting location
unique_locations = predictions["pickup_location_id"].unique().tolist()
selected_location = st.selectbox("Select a pickup location to view prediction:", sorted(unique_locations))

if selected_location:
    selected_feature = features[features["pickup_location_id"] == selected_location]
    selected_prediction = predictions[predictions["pickup_location_id"] == selected_location]

    if not selected_prediction.empty and not selected_feature.empty:
        fig = plot_prediction(
            features=selected_feature,
            prediction=selected_prediction.rename(columns={"predicted_rides": "predicted_demand"}),
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    else:
        st.warning(f"⚠️ No prediction available for location {selected_location} at the current timestamp.")

# Optional: Evaluate predictions vs actuals for selected month/year
st.subheader("📅 Compare Predictions with Actuals")
selected_year = st.selectbox("Select Year", sorted(hist["pickup_hour"].dt.year.unique(), reverse=True))
selected_month = st.selectbox("Select Month", sorted(hist["pickup_hour"].dt.month.unique()))

# Filter data for selected month/year
mask = (hist["pickup_hour"].dt.year == selected_year) & (hist["pickup_hour"].dt.month == selected_month)
actuals = hist.loc[mask]

if selected_location:
    location_actuals = actuals[actuals["pickup_location_id"] == selected_location]
    if not location_actuals.empty:
        st.line_chart(location_actuals.set_index("pickup_hour")["rides"], use_container_width=True)
        st.caption(f"📉 Actual ride count for location {selected_location} during {selected_month:02d}/{selected_year}")
    else:
        st.warning("⚠️ No actual data available for selected location and time period.")

progress_bar.progress(3 / N_STEPS)

# ──────────────────────────────────────────────────────────
# 📊 Model Monitoring: MAE Over Time
# ──────────────────────────────────────────────────────────
st.subheader("📉 Model Monitoring: Mean Absolute Error (MAE)")
past_hours = st.slider("Select time range for MAE calculation (hours):", min_value=12, max_value=24*28, value=72)

@st.cache_data(show_spinner=False)
def fetch_hourly_rides(hours_back):
    current_hour = (pd.Timestamp("2025-04-30 00:00:00", tz="UTC") - timedelta(hours=hours_back)).floor("h")
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
    query = fg.select_all().filter(fg.pickup_hour >= current_hour)
    return query.read()

@st.cache_data(show_spinner=False)
def fetch_recent_predictions(hours_back):
    current_hour = (pd.Timestamp("2025-04-30 00:00:00", tz="UTC") - timedelta(hours=hours_back)).floor("h")
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    query = fg.select_all().filter(fg.pickup_hour >= current_hour)
    return query.read()

try:
    df_actual = fetch_hourly_rides(past_hours)
    df_pred = fetch_recent_predictions(past_hours)
    merged_df = pd.merge(df_actual, df_pred, on=["pickup_location_id", "pickup_hour"])
    merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

    mae_by_hour = (
        merged_df.groupby("pickup_hour")["absolute_error"]
        .mean()
        .reset_index()
        .rename(columns={"absolute_error": "MAE"})
    )

    fig = px.line(
        mae_by_hour,
        x="pickup_hour",
        y="MAE",
        title="Mean Absolute Error (MAE) by Pickup Hour",
        labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
        markers=True,
    )
    st.plotly_chart(fig)
    st.caption(f"📈 Average MAE across selected range: {mae_by_hour['MAE'].mean():.2f}")
except Exception as e:
    st.warning(f"🚫 Unable to calculate MAE: {str(e)}")

progress_bar.progress(4 / N_STEPS)
