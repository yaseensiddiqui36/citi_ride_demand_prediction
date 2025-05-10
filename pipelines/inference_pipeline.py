# from datetime import datetime, timedelta

# import pandas as pd

# import src.config as config
# from src.inference import (
#     get_feature_store,
#     get_model_predictions,
#     load_model_from_registry,
# )

# # Get the current datetime64[us, Etc/UTC]
# # for number in range(22, 24 * 29):
# # current_date = pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=number)
# current_date = pd.Timestamp.now(tz="Etc/UTC")
# feature_store = get_feature_store()

# # read time-series data from the feature store
# fetch_data_to = current_date - timedelta(hours=1)
# fetch_data_from = current_date - timedelta(days=1 * 29)
# print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
# feature_view = feature_store.get_feature_view(
#     name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
# )

# ts_data = feature_view.get_batch_data(
#     start_time=(fetch_data_from - timedelta(days=1)),
#     end_time=(fetch_data_to + timedelta(days=1)),
# )
# ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
# ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)
# ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)

# from src.data_utils import transform_ts_data_info_features

# features = transform_ts_data_info_features(ts_data, window_size=24 * 28, step_size=23)

# model = load_model_from_registry()

# predictions = get_model_predictions(model, features)
# predictions["pickup_hour"] = current_date.ceil("h")
# print(predictions)

# feature_group = get_feature_store().get_or_create_feature_group(
#     name=config.FEATURE_GROUP_MODEL_PREDICTION,
#     version=1,
#     description="Predictions from LGBM Model",
#     primary_key=["pickup_location_id", "pickup_hour"],
#     event_time="pickup_hour",
# )

# feature_group.insert(predictions, write_options={"wait_for_job": False})




from datetime import timedelta
import pandas as pd
from hsfs.feature import Feature
import joblib
import os

import src.config as config
from src.inference import (
    get_feature_store,
    load_model_from_registry,
    get_model_predictions,
)
from src.data_utils import transform_ts_data_info_features


def main():
    # ── 1️⃣  Connect to your Hopsworks feature store
    fs = get_feature_store()

    # ── 2️⃣  Read your historical hourly FG and find the latest hour
    hourly_fg = fs.get_feature_group(
        name    = config.FEATURE_GROUP_NAME,
        version = config.FEATURE_GROUP_VERSION,
    )
    hist      = hourly_fg.read()
    latest_hr = pd.to_datetime(hist["pickup_hour"].max())

    # ── 3️⃣  Define the sliding‐window slice
    window_size = 24 * 28    # 672 hours
    fetch_from  = latest_hr - timedelta(hours=window_size + 1)
    fetch_to    = latest_hr
    print(f"Building features from {fetch_from} → {fetch_to}")

    # ── 4️⃣  Pull exactly that range from your Feature View
    fv = fs.get_feature_view(
        name    = config.FEATURE_VIEW_NAME,
        version = config.FEATURE_VIEW_VERSION,
    )
    ts_data = (
        fv.get_batch_data(start_time=fetch_from, end_time=fetch_to)
          .loc[lambda df: df.pickup_hour.between(fetch_from, fetch_to)]
          .sort_values(["pickup_location_id","pickup_hour"])
    )

    # ── 5️⃣  Turn it into sliding‐window features
    features = transform_ts_data_info_features(
        ts_data,
        feature_col = "rides",
        window_size = window_size,
        step_size   = 1,
    )

    # ── 6️⃣  🎯 Insert a dummy "target" column so your pipeline sees 676 inputs
    features["target"] = 0

    # ── 7️⃣  Load your full sklearn Pipeline (with featurizer + LightGBM)
    model = load_model_from_registry()

    # ── 8️⃣  Get the raw predictions (this returns a column "predicted_demand")
    preds = get_model_predictions(model, features)

    # ── 9️⃣  Rename to match your FG schema
    # preds = preds.rename(columns={"predicted_demand": "predicted_rides"})

    # ── 🔟  Stamp on the next‐hour timestamp
    preds["pickup_hour"] = latest_hr + timedelta(hours=1)

    # ── 1️⃣1️⃣  Create (or fetch) your prediction FG v2
    pred_fg = fs.get_or_create_feature_group(
        name         = config.FEATURE_GROUP_MODEL_PREDICTION,
        version      = config.FEATURE_GROUP_MODEL_PREDICTION_VERSION,
        description  = "Next-hour demand predictions from LGBM model",
        primary_key  = ["pickup_location_id", "pickup_hour"],
        event_time   = "pickup_hour",
        online_enabled=False,
        features     = [
            Feature("pickup_location_id", "string"),
            Feature("pickup_hour",        "timestamp"),
            Feature("predicted_demand",    "double"),
        ],
    )

    # ── 1️⃣2️⃣  Cast to the FG schema and insert
    preds["pickup_location_id"] = preds["pickup_location_id"].astype(str)
    preds["predicted_demand"]    = preds["predicted_demand"].astype("float64")

    pred_fg.insert(preds, write_options={"wait_for_job": False})
    print("✅ Inference complete — predictions up to", preds["pickup_hour"].iloc[0])


if __name__ == "__main__":
    main()
