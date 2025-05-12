import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Import utility functions
from src.utils.mlflow_util import get_latest_full_model_from_mlflow
from src.utils.s3_upload_util import load_features_from_s3, save_forecast_to_s3
from src.utils.forecast_utils import generate_future_forecast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def run_inference_pipeline(forecast_days=90):
    """
    Run the Citibank demand inference pipeline to generate hourly predictions.

    Args:
        forecast_days (int): Number of days to forecast (default: 90)

    Returns:
        pd.DataFrame: DataFrame containing the hourly predictions
    """
    try:
        logger.info(f"Starting Citibank demand inference pipeline...")

        # Step 1: Get the latest full feature model from MLflow
        logger.info("Loading latest full feature model from MLflow...")
        model = get_latest_full_model_from_mlflow(model_name="citibike-full-model")

        # Step 2: Load historical feature data from S3
        logger.info("Loading historical feature data from S3...")
        feature_bucket = os.getenv("TARGET_S3_BUCKET")
        feature_path = "citibike/features/full/data.csv"
        historical_data = load_features_from_s3(bucket=feature_bucket, key=feature_path)

        # Step 3: Calculate forecast period
        current_date = datetime.now()
        start_date = current_date + timedelta(hours=1)
        end_date = start_date + timedelta(days=forecast_days)

        window_size = int(os.getenv("WINDOW_SIZE", 24 * 28))  # Default: 28 days of hourly data

        # Step 4: Generate forecast
        logger.info(f"Generating hourly predictions from {start_date} to {end_date}...")
        forecast_df = generate_future_forecast(
            model=model,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size
        )

        # Step 5: Save forecast to S3
        logger.info("Saving forecast to S3...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forecast_path = f"forecasts/citibike_hourly_forecast_{timestamp}.csv"
        latest_path = "forecasts/citibike_hourly_forecast_latest.csv"

        # Save both timestamped and latest versions
        save_forecast_to_s3(forecast_df, bucket=feature_bucket, key=forecast_path)
        save_forecast_to_s3(forecast_df, bucket=feature_bucket, key=latest_path)

        logger.info(f"Inference pipeline completed successfully!")
        logger.info(f"Forecast saved to s3://{feature_bucket}/{forecast_path}")

        return forecast_df

    except Exception as e:
        logger.error(f"Error in inference pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    run_inference_pipeline()
