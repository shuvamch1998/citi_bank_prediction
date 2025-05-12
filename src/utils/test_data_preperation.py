import pandas as pd
import numpy as np
import logging

from pandas.core.dtypes.common import is_datetime64_any_dtype, is_object_dtype, is_numeric_dtype
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session
import boto3
import os
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def fetch_features_from_aws_feature_store():
    """Fetch feature data from S3 bucket in CSV format using boto3."""
    logger.info("Fetching features from S3 bucket...")

    # Create boto3 session
    boto_session = boto3.Session(region_name=os.getenv("AWS_DEFAULT_REGION"))
    s3_client = boto_session.client("s3")

    # Get bucket name from environment
    bucket_name = os.getenv("TARGET_S3_BUCKET", "migration-citi-bike")

    # Define paths to the CSV files (based on our Athena export structure)
    feature_paths = {
        'baseline_features': "citibike/features/baseline/data.csv",
        'full_features': "citibike/features/full/data.csv",
        'reduced_features': "citibike/features/reduced/data.csv"
    }

    feature_sets = {}

    for set_name, path in feature_paths.items():
        logger.info(f"Fetching {set_name} from S3...")
        try:
            # Use boto3 to get the file content
            import io

            # Get the object from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=path)

            # Read the CSV data from the response
            csv_content = response['Body'].read()

            # Parse CSV with pandas
            df = pd.read_csv(io.BytesIO(csv_content))

            # Convert string columns to appropriate types
            for col in df.columns:
                # Convert datetime columns
                if col in ['pickup_hour', 'event_time'] or 'time' in col.lower():
                    df[col] = pd.to_datetime(df[col])

            feature_sets[set_name] = df
            logger.info(f"Loaded {len(df)} records for {set_name}")

        except Exception as e:
            logger.error(f"Error fetching {set_name} from S3: {str(e)}")
            # Try to use local data as fallback
            try:
                # First look for local CSV file
                local_csv_path = f"./data/{set_name}.csv"
                if os.path.exists(local_csv_path):
                    feature_sets[set_name] = pd.read_csv(local_csv_path)
                    # Convert datetime columns
                    for col in feature_sets[set_name].columns:
                        if col in ['pickup_hour', 'event_time'] or 'time' in col.lower():
                            feature_sets[set_name][col] = pd.to_datetime(feature_sets[set_name][col])
                    logger.info(f"Loaded {set_name} from local CSV: {local_csv_path}")
                else:
                    # Try parquet as fallback
                    local_parquet_path = f"./data/transformed/citibike_{set_name}.parquet"
                    feature_sets[set_name] = pd.read_parquet(local_parquet_path)
                    logger.info(f"Loaded {set_name} from local parquet: {local_parquet_path}")
            except Exception as local_e:
                logger.error(f"Local fallback also failed: {str(local_e)}")
                raise

    # Check if we have all required feature sets
    required_sets = ['baseline_features', 'full_features', 'reduced_features']
    missing_sets = [s for s in required_sets if s not in feature_sets]
    if missing_sets:
        raise ValueError(f"Missing required feature sets: {missing_sets}")

    return feature_sets

def transform_ts_data_info_features_and_target (df, target_col="rides", window_size=12, step_size=1):
    """
    Enhanced version that preserves all original features while adding
    sliding window features for the target column.

    Parameters:
        df (pd.DataFrame): The input DataFrame
        target_col (str): The column to predict
        window_size (int): The number of historical values to use as features
        step_size (int): The number of rows to slide the window by

    Returns:
        tuple: (features DataFrame, targets Series)
    """
    # Get all unique location IDs
    location_id_col = "start_station_id" if "start_station_id" in df.columns else "pickup_location_id"
    location_ids = df[location_id_col].unique()

    # List to store transformed data for each location
    transformed_data = []

    # Columns to exclude from feature set
    exclude_columns = [target_col, 'entity_id', 'event_time', 'record_insertion_time',
                       'write_time', 'api_invocation_time', 'is_deleted']

    # Get feature columns (all columns except excluded ones)
    original_feature_columns = [col for col in df.columns if col not in exclude_columns]

    # Loop through each location ID and transform the data
    for location_id in location_ids:
        try:
            # Filter and sort the data for this location
            location_data = df[df[location_id_col] == location_id].sort_values('pickup_hour').reset_index(drop=True)

            # Skip if we don't have enough data for this location
            if len(location_data) <= window_size:
                logger.warning(f"Not enough data for location_id {location_id}. Skipping.")
                continue

            # Create the tabular data using a sliding window approach
            rows = []
            for i in range(0, len(location_data) - window_size, step_size):
                # Get window slice and next row
                window_slice = location_data.iloc[i:i + window_size]
                next_row = location_data.iloc[i + window_size]

                # Extract target value from next row
                target_value = next_row[target_col]

                # Create a row by combining:
                # 1. All features from the most recent row in the window
                # 2. Historical target values as lag features
                row_dict = {}

                # Add most recent features
                for col in original_feature_columns:
                    row_dict[col] = window_slice.iloc[-1][col]

                # Add historical target values as lag features
                target_values = window_slice[target_col].values
                for j in range(window_size):
                    lag_name = f"{target_col}_t-{window_size - j}"
                    row_dict[lag_name] = target_values[j]

                # Add target
                row_dict["target"] = target_value

                rows.append(row_dict)

            # Create DataFrame for this location
            if rows:
                transformed_df = pd.DataFrame(rows)
                transformed_data.append(transformed_df)

        except Exception as e:
            logger.warning(f"Error processing location_id {location_id}: {str(e)}")

    # Combine all transformed data into a single DataFrame
    if not transformed_data:
        raise ValueError(
            "No data could be transformed. Check if input DataFrame is empty or window size is too large."
        )

    final_df = pd.concat(transformed_data, ignore_index=True)

    # Separate features and target
    features = final_df.drop('target', axis=1)
    targets = final_df['target']

    return features, targets


def prepare_train_test_split(df, window_size=None, step_size=None):
    """
    Prepare training and testing data using time-based split.
    Uses ALL available features from the dataframe.

    Args:
        df (pd.DataFrame): Feature DataFrame
        window_size, step_size: Parameters kept for API compatibility but not used

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # These are not features, they're metadata or internal fields
    exclude_columns = ['entity_id', 'event_time', 'record_insertion_time',
                       'write_time', 'api_invocation_time', 'is_deleted']

    # Create a proper feature set (all columns except target and excluded)
    feature_columns = [col for col in df.columns if col not in exclude_columns and col != 'rides']

    # Sort by time
    df = df.sort_values('pickup_hour')

    # Use time-based split (80% train, 20% test)
    # FIX: Use numpy's sort function instead of the sort method
    import numpy as np
    unique_times = np.sort(df['pickup_hour'].unique())

    split_idx = int(len(unique_times) * 0.8)
    split_time = unique_times[split_idx]

    # Split data
    train_mask = df['pickup_hour'] < split_time
    test_mask = df['pickup_hour'] >= split_time

    # Create train/test datasets that include ALL features
    X_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, 'rides']

    X_test = df.loc[test_mask, feature_columns]
    y_test = df.loc[test_mask, 'rides']

    logger.info(f"Train set: {len(X_train)} samples (before {split_time})")
    logger.info(f"Test set: {len(X_test)} samples (after {split_time})")
    logger.info(f"Using {len(feature_columns)} features")

    return X_train, X_test, y_train, y_test


def preprocess_features_for_lightgbm(X_train, X_test):
    """
    Preprocess features to be compatible with LightGBM.
    - Convert datetime columns to numeric features
    - Convert categorical/object columns to numeric using label encoding

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features

    Returns:
        tuple: (X_train_processed, X_test_processed)
    """
    # Make copies to avoid modifying the original DataFrames
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    # Keep track of encoders for categorical columns
    encoders = {}

    # Process each column
    for column in X_train.columns:
        # Handle datetime columns
        if is_datetime64_any_dtype(X_train[column]):
            # Extract useful datetime components
            X_train_processed[f"{column}_hour"] = X_train[column].dt.hour
            X_train_processed[f"{column}_day"] = X_train[column].dt.day
            X_train_processed[f"{column}_month"] = X_train[column].dt.month
            X_train_processed[f"{column}_year"] = X_train[column].dt.year
            X_train_processed[f"{column}_dayofweek"] = X_train[column].dt.dayofweek

            # Also convert to timestamp for a continuous representation
            X_train_processed[f"{column}_timestamp"] = X_train[column].astype(np.int64) // 10 ** 9

            # Do the same for test data
            X_test_processed[f"{column}_hour"] = X_test[column].dt.hour
            X_test_processed[f"{column}_day"] = X_test[column].dt.day
            X_test_processed[f"{column}_month"] = X_test[column].dt.month
            X_test_processed[f"{column}_year"] = X_test[column].dt.year
            X_test_processed[f"{column}_dayofweek"] = X_test[column].dt.dayofweek
            X_test_processed[f"{column}_timestamp"] = X_test[column].astype(np.int64) // 10 ** 9

            # Drop the original datetime column
            X_train_processed = X_train_processed.drop(column, axis=1)
            X_test_processed = X_test_processed.drop(column, axis=1)

        # Handle object/string columns using label encoding
        elif is_object_dtype(X_train[column]):
            encoder = LabelEncoder()

            # Fit on training data
            X_train_processed[column] = encoder.fit_transform(X_train[column])

            # Transform test data using the same encoder
            # Handle unseen categories in test data
            X_test_unique = X_test[column].unique()
            train_categories = set(encoder.classes_)

            # Check for unseen categories
            for cat in X_test_unique:
                if cat not in train_categories:
                    # Handle unseen category - assign a default value
                    X_test_processed.loc[X_test[column] == cat, column] = -1

            # Transform values that exist in the training data
            mask_seen = X_test[column].isin(train_categories)
            if mask_seen.any():
                X_test_processed.loc[mask_seen, column] = encoder.transform(X_test.loc[mask_seen, column])

            # Store encoder for later use if needed
            encoders[column] = encoder

    # Verify all columns are now numeric
    non_numeric_cols = [col for col in X_train_processed.columns
                        if not is_numeric_dtype(X_train_processed[col])]

    if non_numeric_cols:
        raise ValueError(f"Non-numeric columns remain after preprocessing: {non_numeric_cols}")

    return X_train_processed, X_test_processed, encoders