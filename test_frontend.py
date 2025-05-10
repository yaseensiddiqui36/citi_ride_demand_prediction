import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False

if "selected_location" not in st.session_state:
    st.session_state.selected_location = None


def get_top_locations(prediction_data, geo_df, top_n=10):
    """Fetch the top N locations by predicted demand and include zone names."""
    merged_df = prediction_data.merge(
        geo_df[["LocationID", "zone"]], left_on="pickup_location_id", right_on="LocationID", how="left"
    )
    top_locations = merged_df.nlargest(top_n, "predicted_demand")
    return top_locations[["pickup_location_id", "zone", "predicted_demand"]]


def plot_selected_location(prediction_data, selected_location):
    """Plot the predicted demand over time for the selected location."""
    if selected_location:
        location_data = prediction_data[prediction_data["zone"] == selected_location]
        if not location_data.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(location_data["pickup_location_id"], location_data["predicted_demand"], marker="o", linestyle="-")
            ax.set_title(f"Predicted Demand for {selected_location}")
            ax.set_xlabel("Pickup Location ID")
            ax.set_ylabel("Predicted Demand")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.write("No data available for the selected location.")


# Streamlit UI Elements
st.title("New York Yellow Taxi Cab Demand Next Hour")
current_date = pd.Timestamp.now(tz="America/New_York")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

progress_bar = st.sidebar.progress(0)
N_STEPS = 4

with st.spinner("Downloading shape file for taxi zones"):
    geo_df = gpd.read_file(DATA_DIR / "taxi_zones" / "taxi_zones.shp").to_crs("epsg:4326")
    st.sidebar.write("Shape file downloaded")
    progress_bar.progress(1 / N_STEPS)

with st.spinner("Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Inference features fetched")
    progress_bar.progress(2 / N_STEPS)

with st.spinner("Fetching predictions"):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Predictions fetched")
    progress_bar.progress(3 / N_STEPS)

# Merge prediction data with zone names
predictions_df = pd.DataFrame(predictions)
top_locations_df = get_top_locations(predictions_df, geo_df)

# Add dropdown menu for location selection
location_names = top_locations_df["zone"].dropna().unique().tolist()
selected_location = st.selectbox("Select a location:", location_names)

with st.spinner("Plotting demand map"):
    st.subheader("Taxi Ride Predictions Map")
    map_obj = create_taxi_map(DATA_DIR / "taxi_zones" / "taxi_zones.shp", predictions)
    
    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

    st.subheader("Prediction Statistics")
    st.dataframe(top_locations_df)

    # Plot line graph for selected location
    st.subheader("Location-Based Prediction Trends")
    plot_selected_location(predictions_df, selected_location)
