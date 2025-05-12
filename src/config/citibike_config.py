import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Hopsworks credentials (if using Hopsworks)
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# Feature group settings
FEATURE_GROUP_NAME = "citibike_hourly_feature_group"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "citibike_hourly_feature_view"
FEATURE_VIEW_VERSION = 1

# Model settings
MODEL_NAME = "citibike_demand_predictor_next_hour"
MODEL_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTION = "citibike_hourly_model_prediction"

# Citi Bike specific settings
# Number of top stations to analyze
TOP_N_STATIONS = 3

# Feature engineering settings
LAG_HOURS = 24 * 28  # 28 days of hourly data
IMPORTANT_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 24*7, 24*14, 24*21, 24*28]
ROLLING_WINDOWS = [24, 24*7, 24*28]  # 1 day, 1 week, 4 weeks

# Weather API settings (if using weather data)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NYC_LAT = 40.7128
NYC_LON = -74.0060

import os
from dotenv import load_dotenv

# Load environment variables from .env

# Now you can access them
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
sagemaker_role = os.getenv("SAGEMAKER_ROLE_ARN")