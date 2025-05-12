import logging
import sys
import os
from datetime import datetime
import pandas as pd
import argparse

import src.config.citibike_config as config
from src.utils.aws_feature_store_util import AWSFeatureStore
from src.utils.citibike_data_util import load_and_process_citibike_data
from src.utils.citibike_feature_engineering_util import process_data_for_modeling
from src.utils.feature_store_util import SQLFeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)





def run_feature_pipeline(year_start=2024, month_start=1, year_end=2024, month_end=3):
    """
    Run the complete feature pipeline for Citi Bike data:
    1. Load data for the specified date range
    2. Process and engineer features
    3. Save processed features locally and register with Feast feature store

    Args:
        year_start (int): Starting year
        month_start (int): Starting month (1-12)
        year_end (int): Ending year
        month_end (int): Ending month (1-12)

    Returns:
        dict: Dictionary containing the three feature sets
    """
    logger.info(f"Starting feature pipeline for {year_start}-{month_start:02d} to {year_end}-{month_end:02d}")

    # Step 1: Load data for each month in the date range
    all_rides = []

    # Handle single year case
    if year_start == year_end:
        logger.info(f"Processing data for {year_start}, months {month_start} to {month_end}")
        for month in range(month_start, month_end + 1):
            try:
                # Load data for the current month
                logger.info(f"Loading data for {year_start}-{month:02d}...")
                month_rides = load_and_process_citibike_data(year_start, month)
                all_rides.append(month_rides)
                logger.info(f"Successfully loaded data for {year_start}-{month:02d}, records: {len(month_rides)}")
            except Exception as e:
                logger.error(f"Error loading data for {year_start}-{month:02d}: {str(e)}")
    else:
        # Handle multi-year case
        for year in range(year_start, year_end + 1):
            # Determine month range for current year
            if year == year_start:
                start_m = month_start
                end_m = 12
            elif year == year_end:
                start_m = 1
                end_m = month_end
            else:
                start_m = 1
                end_m = 12

            logger.info(f"Processing data for {year}, months {start_m} to {end_m}")
            for month in range(start_m, end_m + 1):
                try:
                    # Load data for the current month
                    logger.info(f"Loading data for {year}-{month:02d}...")
                    month_rides = load_and_process_citibike_data(year, month)
                    all_rides.append(month_rides)
                    logger.info(f"Successfully loaded data for {year}-{month:02d}, records: {len(month_rides)}")
                except Exception as e:
                    logger.error(f"Error loading data for {year}-{month:02d}: {str(e)}")

    # Combine all data
    if not all_rides:
        raise Exception("No data could be loaded for the specified date range.")

    logger.info("Combining all data...")
    rides = pd.concat(all_rides, ignore_index=True)
    logger.info(f"Combined data. Total records: {len(rides)}")

    # Step 2: Apply feature engineering using the process_data_for_modeling function
    logger.info("Applying feature engineering process...")
    feature_sets = process_data_for_modeling(rides, top_n=config.TOP_N_STATIONS)

    baseline_features = feature_sets['baseline_features']
    full_features = feature_sets['full_features']
    reduced_features = feature_sets['reduced_features']

    logger.info(f"Feature engineering complete.")
    logger.info(f"Baseline features: {len(baseline_features.columns)} columns")
    logger.info(f"Full features: {len(full_features.columns)} columns")
    logger.info(f"Reduced features: {len(reduced_features.columns)} columns")

    # Step 3: Save all feature sets locally
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Create permanent feature files (always overwritten with latest)
    baseline_path = config.TRANSFORMED_DATA_DIR / "citibike_baseline_features.parquet"
    full_path = config.TRANSFORMED_DATA_DIR / "citibike_full_features.parquet"
    reduced_path = config.TRANSFORMED_DATA_DIR / "citibike_reduced_features.parquet"

    # Also save timestamped versions for historical tracking
    baseline_ts_path = config.TRANSFORMED_DATA_DIR / f"citibike_baseline_features_{current_timestamp}.parquet"
    full_ts_path = config.TRANSFORMED_DATA_DIR / f"citibike_full_features_{current_timestamp}.parquet"
    reduced_ts_path = config.TRANSFORMED_DATA_DIR / f"citibike_reduced_features_{current_timestamp}.parquet"

    # Save to both permanent and timestamped locations
    baseline_features.to_parquet(baseline_path)
    baseline_features.to_parquet(baseline_ts_path)
    logger.info(f"Baseline features saved to {baseline_path} and {baseline_ts_path}")

    full_features.to_parquet(full_path)
    full_features.to_parquet(full_ts_path)
    logger.info(f"Full features saved to {full_path} and {full_ts_path}")

    reduced_features.to_parquet(reduced_path)
    reduced_features.to_parquet(reduced_ts_path)
    logger.info(f"Reduced features saved to {reduced_path} and {reduced_ts_path}")

    # Step 4: Save all feature to Sagemaker feature store
    try:
        logger.info("Setting up AWS Feature Store...")
        # Initialize AWS Feature Store with appropriate region
        store = AWSFeatureStore(region_name="us-east-1")

        # Store each feature set
        logger.info("Storing baseline features...")
        store.insert_features(
            feature_group="citibike_baseline_features",
            df=baseline_features,
            entity_column="start_station_id",
            timestamp_column="pickup_hour",
            description="Simple lag model features (24h and 168h)"
        )

        logger.info("Storing full features...")
        store.insert_features(
            feature_group="citibike_full_features",
            df=full_features,
            entity_column="start_station_id",
            timestamp_column="pickup_hour",
            description="Complete feature set with all lag features"
        )

        logger.info("Storing reduced features...")
        store.insert_features(
            feature_group="citibike_reduced_features",
            df=reduced_features,
            entity_column="start_station_id",
            timestamp_column="pickup_hour",
            description="Reduced feature set with top 10 features"
        )

        logger.info("All features successfully stored in AWS Feature Store")
    except Exception as e:
        logger.error(f"Error storing features in AWS Feature Store: {str(e)}")

    # Return all feature sets
    return {
        'baseline_features': baseline_features,
        'full_features': full_features,
        'reduced_features': reduced_features
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run Citibike feature engineering pipeline")
    parser.add_argument("--year-start", type=int, default=2024, help="Start year")
    parser.add_argument("--month-start", type=int, default=1, help="Start month (1-12)")
    parser.add_argument("--year-end", type=int, default=2024, help="End year")
    parser.add_argument("--month-end", type=int, default=3, help="End month (1-12)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the pipeline with the specified parameters
    run_feature_pipeline(
        year_start=args.year_start,
        month_start=args.month_start,
        year_end=args.year_end,
        month_end=args.month_end
    )
