# import logging
# import os
# import sys
# from datetime import datetime, timedelta, timezone

# import hopsworks
# import pandas as pd

# import src.config as config
# from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,  # Set the logging level
#     format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
#     handlers=[
#         logging.StreamHandler(sys.stdout),  # Output logs to stdout
#     ],
# )
# logger = logging.getLogger(__name__)


# # Step 1: Get the current date and time (timezone-aware)
# current_date = pd.to_datetime(datetime.now(timezone.utc)).ceil("h")
# logger.info(f"Current date and time (UTC): {current_date}")

# # Step 2: Define the data fetching range
# fetch_data_to = current_date
# fetch_data_from = current_date - timedelta(days=28)
# logger.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# # Step 3: Fetch raw data
# logger.info("Fetching raw data...")
# rides = fetch_batch_raw_data(fetch_data_from, fetch_data_to)
# logger.info(f"Raw data fetched. Number of records: {len(rides)}")

# # Step 4: Transform raw data into time-series data
# logger.info("Transforming raw data into time-series data...")
# ts_data = transform_raw_data_into_ts_data(rides)
# logger.info(
#     f"Transformation complete. Number of records in time-series data: {len(ts_data)}"
# )

# # Step 5: Connect to the Hopsworks project
# logger.info("Connecting to Hopsworks project...")
# project = hopsworks.login(
#     project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
# )
# logger.info("Connected to Hopsworks project.")

# # Step 6: Connect to the feature store
# logger.info("Connecting to the feature store...")
# feature_store = project.get_feature_store()
# logger.info("Connected to the feature store.")

# # Step 7: Connect to or create the feature group
# logger.info(
#     f"Connecting to the feature group: {config.FEATURE_GROUP_NAME} (version {config.FEATURE_GROUP_VERSION})..."
# )
# feature_group = feature_store.get_feature_group(
#     name=config.FEATURE_GROUP_NAME,
#     version=config.FEATURE_GROUP_VERSION,
# )
# logger.info("Feature group ready.")

# # Step 8: Insert data into the feature group
# logger.info("Inserting data into the feature group...")
# feature_group.insert(ts_data, write_options={"wait_for_job": False})
# logger.info("Data insertion completed.")










#!/usr/bin/env python3
"""
Feature-engineering pipeline:
▪ Fetch Citi Bike CSVs that are already stored locally
▪ Aggregate to hourly counts for the three chosen stations
▪ Write the hourly table to the Hopsworks feature-store
"""

import logging
import os
import sys
from datetime import timedelta

import hopsworks
import pandas as pd

import src.config as config
from src.data_utils import (
    transform_raw_data_into_ts_data,
    load_and_process_citibike_data
)

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    # ────────────────────────────────────────────────────────────────────────────
    # 1 Define the window we want (last 28 days ending 1 Apr 2025 00 UTC)
    # ────────────────────────────────────────────────────────────────────────────
    # current_date    = pd.Timestamp("2025-04-01 00:00:00", tz="UTC")
    # fetch_data_from = current_date - timedelta(days=28)
    # fetch_data_to   = current_date
    # logger.info(f"Current date  : {current_date}")
    # logger.info(f"Fetch window  : {fetch_data_from}  →  {fetch_data_to}")

    # # ────────────────────────────────────────────────────────────────────────────
    # # 2 Load raw CSVs from local (abort gracefully if missing)
    # # ────────────────────────────────────────────────────────────────────────────
    # all_raw = []
    # for year, months in [
    #     (2023, list(range(1, 13))),
    #     (2024, list(range(1, 13))),
    #     (2025, [1, 2, 3]),
    # ]:
    #     try:
    #         df_year = load_and_process_citibike_data_local(
    #             year      = year,
    #             months    = months,
    #             base_path = config.LOCAL_CITIBIKE_DATA_PATH,
    #         )
    #         logger.info(f"  • {year}: loaded {len(df_year):,} rows")
    #         all_raw.append(df_year)
    #     except FileNotFoundError as e:
    #         logger.warning(f"No data for {year} (months={months}): {e}")

    # if not all_raw:
    #     logger.warning("No raw data found for any year → exiting with success")
    #     sys.exit(0)

    # raw_rides = pd.concat(all_raw, ignore_index=True)
    # logger.info(f"✅ Total raw rows loaded: {len(raw_rides):,}")


    ############################################################

    raw_rides = load_and_process_citibike_data()    


    ############################################################

    # ────────────────────────────────────────────────────────────────────────────
    # 3 Aggregate to hourly-location counts
    # ────────────────────────────────────────────────────────────────────────────
    logger.info("Aggregating to hourly counts …")
    ts_data = transform_raw_data_into_ts_data(raw_rides)
    logger.info(f"Time-series rows: {len(ts_data):,}")

    # ────────────────────────────────────────────────────────────────────────────
    # 4 Connect to Hopsworks
    # ────────────────────────────────────────────────────────────────────────────
    logger.info("Logging in to Hopsworks …")
    project = hopsworks.login(
        project       = config.HOPSWORKS_PROJECT_NAME,
        api_key_value = config.HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    # ────────────────────────────────────────────────────────────────────────────
    # 5 Get-or-create the Feature-Group
    # ────────────────────────────────────────────────────────────────────────────
    from hsfs.feature import Feature

    fg_schema = [
        Feature("pickup_hour",        "timestamp"),
        Feature("pickup_location_id", "string"),
        Feature("rides",              "int"),
    ]

    hourly_fg = fs.get_or_create_feature_group(
        name          = config.FEATURE_GROUP_NAME,
        version       = config.FEATURE_GROUP_VERSION,
        description   = "Hourly aggregated Citi Bike rides per location",
        primary_key   = ["pickup_hour", "pickup_location_id"],
        event_time    = "pickup_hour",
        online_enabled=False,
        features      = fg_schema,
    )
    logger.info("Feature-group ready → id %s", hourly_fg.id)

    # ────────────────────────────────────────────────────────────────────────────
    # 6 Cast & insert into Hopsworks
    # ────────────────────────────────────────────────────────────────────────────
    ts_data["pickup_location_id"] = ts_data["pickup_location_id"].astype(str)
    ts_data["rides"]              = ts_data["rides"].astype("int64")

    logger.info("Writing rows to the feature-store …")
    hourly_fg.insert(ts_data, write_options={"wait_for_job": False})
    logger.info("✅  Data insertion triggered")

if __name__ == "__main__":
    main()
