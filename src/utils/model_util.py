import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import os
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

from src.utils.test_data_preperation import preprocess_features_for_lightgbm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaselinePredictor(BaseEstimator, RegressorMixin):
    """Simple baseline model that predicts based on historical values."""

    def __init__(self):
        self.mean = None

    def fit(self, X, y):
        # Store the mean of the target for fallback
        self.mean = y.mean()
        return self

    def predict(self, X):
        # Compute average of the last day and last week values if available
        predictions = np.zeros(len(X))

        # Use the average of available lag features
        feature_cols = [col for col in X.columns if col.startswith('rides_t-')]

        if not feature_cols:
            # Fallback to the mean if no lag features
            return np.full(len(X), self.mean)

        # Calculate the average of all lag features for each row
        for i, row in X.iterrows():
            values = [row[col] for col in feature_cols]
            predictions[i] = np.mean(values)

        return predictions


def get_pipeline(**hyper_params):
    """
    Returns a pipeline with optional parameters for LGBMRegressor.

    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with LGBMRegressor.
    """
    pipeline = make_pipeline(
        lgb.LGBMRegressor(**hyper_params),  # Pass optional parameters here
    )
    return pipeline


def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train a simple baseline model using average of historical values.

    Args:
        X_train, y_train, X_test, y_test: Train and test data

    Returns:
        tuple: (trained model, test MAE, processed X_test)
    """
    # Preprocess features for consistency with other models
    X_train_processed, X_test_processed, _ = preprocess_features_for_lightgbm(X_train, X_test)

    # Train model
    baseline_model = BaselinePredictor()
    baseline_model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = baseline_model.predict(X_test_processed)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Baseline model MAE: {mae:.4f}")

    return baseline_model, mae, X_test_processed


def train_full_feature_model(X_train, y_train, X_test, y_test):
    """
    Train a model using all lag features.

    Args:
        X_train, y_train, X_test, y_test: Train and test data

    Returns:
        tuple: (trained model, test MAE, processed X_test)
    """
    # Preprocess features for LightGBM
    X_train_processed, X_test_processed, _ = preprocess_features_for_lightgbm(X_train, X_test)

    # Use the pipeline with hyperparameters
    model = get_pipeline(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=31,
        random_state=42
    )

    # Train model on processed data
    model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = model.predict(X_test_processed)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Full feature model MAE: {mae:.4f}")

    return model, mae, X_test_processed


def train_reduced_feature_model(X_train, y_train, X_test, y_test):
    """
    Train a model using reduced feature set.

    Args:
        X_train, y_train, X_test, y_test: Train and test data

    Returns:
        tuple: (trained model, test MAE, processed X_test)
    """
    # Preprocess features for LightGBM
    X_train_processed, X_test_processed, _ = preprocess_features_for_lightgbm(X_train, X_test)

    # This data is already reduced, so we can use a simpler lightgbm model
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=25,
        random_state=42
    )

    # Train model on processed data
    model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = model.predict(X_test_processed)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Reduced feature model MAE: {mae:.4f}")

    return model, mae, X_test_processed


def save_best_model(model, model_name="best_model"):
    """
    Save the best model to local file and potentially S3.

    Args:
        model: The trained model to save
        model_name (str): Name for the saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save locally
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    return model_path


def load_model(model_path):
    """
    Load model from file.

    Args:
        model_path (str): Path to the saved model

    Returns:
        The loaded model
    """
    return joblib.load(model_path)


def save_metrics_to_csv(results, output_path="model_metrics.csv", s3_bucket=None):
    """
    Save model metrics to a CSV file and optionally to S3.

    Args:
        results (dict): Dictionary with model results including metrics
        output_path (str): Path to save the CSV file
        s3_bucket (str, optional): S3 bucket name to save the CSV file

    Returns:
        str: Path to the saved CSV file
    """
    import pandas as pd
    import os
    from datetime import datetime
    import logging
    import boto3

    logger = logging.getLogger(__name__)

    # Create a DataFrame with metrics
    metrics_data = []

    for model_name, model_results in results.items():
        # Get base metrics
        metrics = {
            'model_name': model_name,
            'mae': model_results.get('mae', None),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add additional metrics if available
        if 'additional_metrics' in model_results:
            for metric_name, metric_value in model_results['additional_metrics'].items():
                metrics[metric_name] = metric_value

        metrics_data.append(metrics)

    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics_data)

    # Create directories if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save to local CSV
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Model metrics saved to {output_path}")

    # Upload to S3 if bucket is specified
    if s3_bucket:
        try:
            s3_client = boto3.client('s3')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_path = f"metrics/model_metrics_{timestamp}.csv"

            s3_client.upload_file(
                output_path,
                s3_bucket,
                s3_path
            )
            logger.info(f"Model metrics uploaded to S3: s3://{s3_bucket}/{s3_path}")
        except Exception as e:
            logger.error(f"Error uploading metrics to S3: {e}")

    return output_path


def append_to_metrics_history(results, history_path="model_metrics_history.csv"):
    """
    Append the current run's metrics to a historical metrics file.

    Args:
        results (dict): Dictionary with model results
        history_path (str): Path to the metrics history CSV file

    Returns:
        pd.DataFrame: Updated metrics history DataFrame
    """
    import pandas as pd
    import os
    from datetime import datetime
    import logging

    logger = logging.getLogger(__name__)

    # Create metrics data for current run
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_data = []

    for model_name, model_results in results.items():
        metrics = {
            'model_name': model_name,
            'mae': model_results.get('mae', None),
            'run_timestamp': run_timestamp,
            'run_id': model_results.get('run_id', None)
        }
        metrics_data.append(metrics)

    current_metrics = pd.DataFrame(metrics_data)

    # Check if history file exists
    if os.path.exists(history_path):
        # Load existing history and append new metrics
        try:
            history_df = pd.read_csv(history_path)
            updated_history = pd.concat([history_df, current_metrics], ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading existing metrics history: {e}")
            updated_history = current_metrics
    else:
        # Create new history file
        updated_history = current_metrics

    # Save updated history
    updated_history.to_csv(history_path, index=False)
    logger.info(f"Metrics history updated in {history_path}")

    return updated_history