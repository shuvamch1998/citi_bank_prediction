import logging
import os
from datetime import datetime

import boto3
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from src.utils.forecast_utils import generate_feb_may_2025_forecast
from src.utils.mlflow_util import log_model_to_mlflow, set_mlflow_tracking, register_best_model
from src.utils.model_util import train_reduced_feature_model, train_full_feature_model, save_best_model, \
    train_baseline_model, save_metrics_to_csv, append_to_metrics_history
from src.utils.plot_utils import plot_model_comparison, save_plot_to_s3, plot_future_forecast, plot_forecast_summary
from src.utils.test_data_preperation import fetch_features_from_aws_feature_store, prepare_train_test_split

# Import our modules


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def run_training_pipeline():
    """
    Main function to run the training pipeline with DagsHub MLflow integration.
    """
    try:
        # Set up MLflow tracking with DagsHub
        mlflow = set_mlflow_tracking()

        # Fetch features from AWS Feature Store
        feature_sets = fetch_features_from_aws_feature_store()

        # Train models on each feature set
        results = {}

        # Use different window sizes for different models
        window_sizes = {
            'baseline': 2,  # Simple baseline with just two lag features
            'full': int(os.getenv("WINDOW_SIZE", 24 * 28)),  # 28 days of hourly data by default
            'reduced': 10  # Top 10 features
        }

        # 1. Baseline model
        logger.info("Training baseline model...")
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            feature_sets['baseline_features'],
            window_size=window_sizes['baseline'],
            step_size=1
        )
        baseline_model, baseline_mae, X_test_baseline_processed = train_baseline_model(
            X_train, y_train, X_test, y_test
        )
        results['baseline'] = {
            'model': baseline_model,
            'mae': baseline_mae,
            'input_data': X_test_baseline_processed,  # Use processed data
            'test_predictions': baseline_model.predict(X_test_baseline_processed)  # Use processed data
        }

        # 2. Full feature model
        logger.info("Training full feature model...")
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            feature_sets['full_features'],
            window_size=window_sizes['full'],
            step_size=1
        )
        full_model, full_mae, X_test_full_processed = train_full_feature_model(
            X_train, y_train, X_test, y_test
        )
        results['full'] = {
            'model': full_model,
            'mae': full_mae,
            'input_data': X_test_full_processed,  # Use processed data
            'test_predictions': full_model.predict(X_test_full_processed)  # Use processed data
        }

        # 3. Reduced feature model
        logger.info("Training reduced feature model...")
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            feature_sets['reduced_features'],
            window_size=window_sizes['reduced'],
            step_size=1
        )
        reduced_model, reduced_mae, X_test_reduced_processed = train_reduced_feature_model(
            X_train, y_train, X_test, y_test
        )
        results['reduced'] = {
            'model': reduced_model,
            'mae': reduced_mae,
            'input_data': X_test_reduced_processed,  # Use processed data
            'test_predictions': reduced_model.predict(X_test_reduced_processed)  # Use processed data
        }

        # Set up the experiment name for MLflow logging
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "citibike-demand-prediction")

        # Create mapping for models and their info
        model_registry_mapping = {
            "citibike-baseline-model": {
                "model": results['baseline']['model'],
                "input_data": results['baseline']['input_data'],
                "params": {"model_type": "baseline", "window_size": window_sizes['baseline']},
                "score": results['baseline']['mae'],
                "feature_set": "baseline_features"
            },
            "citibike-full-model": {
                "model": results['full']['model'],
                "input_data": results['full']['input_data'],
                "params": {
                    "model_type": "lightgbm",
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 8,
                    "num_leaves": 31,
                    "window_size": window_sizes['full']
                },
                "score": results['full']['mae'],
                "feature_set": "full_features"
            },
            "citibike-reduced-model": {
                "model": results['reduced']['model'],
                "input_data": results['reduced']['input_data'],
                "params": {
                    "model_type": "lightgbm",
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "num_leaves": 25,
                    "window_size": window_sizes['reduced']
                },
                "score": results['reduced']['mae'],
                "feature_set": "reduced_features"
            }
        }

        # Log all models to MLflow and capture their info
        model_run_info = {}
        for model_name, model_data in model_registry_mapping.items():
            logger.info(f"Logging {model_name} to MLflow...")
            model_info = log_model_to_mlflow(
                model=model_data["model"],
                input_data=model_data["input_data"],
                experiment_name=experiment_name,
                metric_name="mae",
                model_name=model_name,
                params=model_data["params"],
                score=model_data["score"]
            )
            model_run_info[model_name] = model_info

        # Print summary
        logger.info("===== Training Results =====")
        logger.info(f"Baseline Model MAE: {results['baseline']['mae']:.4f}")
        logger.info(f"Full Feature Model MAE: {results['full']['mae']:.4f}")
        logger.info(f"Reduced Feature Model MAE: {results['reduced']['mae']:.4f}")

        # Calculate improvement percentages
        baseline_improvement = ((baseline_mae - full_mae) / baseline_mae) * 100
        reduced_vs_full = ((full_mae - reduced_mae) / full_mae) * 100 if full_mae > 0 else 0

        logger.info(f"Full model improves over baseline by: {baseline_improvement:.2f}%")
        logger.info(f"Reduced model compared to full model: {reduced_vs_full:.2f}%")
        logger.info("============================")

        # Add additional metrics to results dictionary
        results['baseline']['additional_metrics'] = {'model_type': 'baseline'}
        results['full']['additional_metrics'] = {
            'model_type': 'full_feature',
            'baseline_improvement': baseline_improvement
        }
        results['reduced']['additional_metrics'] = {
            'model_type': 'reduced_feature',
            'full_model_comparison': reduced_vs_full
        }

        # Save metrics to CSV
        metrics_path = save_metrics_to_csv(
            results,
            output_path="metrics/model_metrics.csv",
            s3_bucket=os.getenv("AWS_S3_BUCKET")
        )

        # Append to metrics history
        history_df = append_to_metrics_history(
            results,
            history_path="metrics/model_metrics_history.csv"
        )



        logger.info(f"Full model improves over baseline by: {baseline_improvement:.2f}%")
        logger.info(f"Reduced model compared to full model: {reduced_vs_full:.2f}%")
        logger.info("============================")

        # Find the best model based on MAE
        best_model_name = min(
            ["citibike-baseline-model", "citibike-full-model", "citibike-reduced-model"],
            key=lambda name: model_registry_mapping[name]["score"]
        )

        # Log metrics files to MLflow
        with mlflow.start_run(run_id=model_run_info[best_model_name]["run_id"]):
            mlflow.log_artifact(metrics_path, "metrics")
            mlflow.log_artifact("metrics/model_metrics_history.csv", "metrics")
        logger.info("Model metrics saved to CSV and MLflow")

        best_mae = model_registry_mapping[best_model_name]["score"]
        best_model = model_registry_mapping[best_model_name]["model"]
        best_window_size = model_registry_mapping[best_model_name]["params"]["window_size"]
        best_feature_set = model_registry_mapping[best_model_name]["feature_set"]

        # Register the best model as Production in MLflow Model Registry
        logger.info(f"Registering {best_model_name} as Production model in MLflow Registry")
        register_best_model(
            model_name=best_model_name,
            run_id=model_run_info[best_model_name]["run_id"],  # Updated to access run_id from dictionary
            stage="Production"
        )

        # Register other models as Staging
        for model_name in model_registry_mapping.keys():
            if model_name != best_model_name:
                logger.info(f"Registering {model_name} as Staging model in MLflow Registry")
                register_best_model(
                    model_name=model_name,
                    run_id=model_run_info[model_name]["run_id"],  # Updated to access run_id from dictionary
                    stage="Staging"
                )

        # Also save best model to local file system and S3
        logger.info(f"Saving best model: {best_model_name} with MAE: {best_mae:.4f}")
        best_model_path = save_best_model(best_model, "best_model")

        # Store the original test data and indices for each model
        baseline_test_data = {'features': X_test, 'targets': y_test, 'indices': X_test.index}
        results['baseline']['original_test_data'] = baseline_test_data

        # Get a sample station ID for visualization
        # Use the original test data to get the station ID
        original_test_data = baseline_test_data['features']  # Use baseline test data for visualization

        # Find station ID column in original data
        id_col = "start_station_id" if "start_station_id" in original_test_data.columns else "pickup_location_id"
        sample_station = original_test_data[id_col].iloc[0]

        # Create model comparison plot
        logger.info(f"Generating model comparison visualizations for station ID: {sample_station}...")
        comparison_plot = plot_model_comparison(
            test_features=original_test_data,  # Use original data for visualization
            test_targets=baseline_test_data['targets'],
            predictions={
                'Baseline': results['baseline']['test_predictions'],
                'Full Features': results['full']['test_predictions'],
                'Reduced Features': results['reduced']['test_predictions']
            },
            station_id=sample_station
        )
        # Save plots locally and to S3
        comparison_plot.write_html("model_comparison.html")
        save_plot_to_s3(comparison_plot, "model_comparison.html")

        # Log the comparison plot to MLflow
        with mlflow.start_run(
                run_id=model_run_info[best_model_name]["run_id"]):  # Updated to access run_id from dictionary
            mlflow.log_artifact("model_comparison.html", "visualizations")
        logger.info("Model comparison visualization saved to local, S3, and MLflow")

        # Generate forecast for February to May 2025
        logger.info("Generating forecast for February to May 2025...")
        forecast_df = generate_feb_may_2025_forecast(
            model=best_model,
            historical_data=feature_sets[best_feature_set],
            window_size=best_window_size
        )

        # Save forecast to CSV locally and to S3
        forecast_df.to_csv("feb_may_2025_forecast.csv", index=False)

        # Upload full hourly forecast to S3
        s3_bucket = os.getenv("AWS_S3_BUCKET")
        if s3_bucket:
            try:
                # Upload to S3 with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_client = boto3.client('s3')
                s3_client.upload_file(
                    "feb_may_2025_forecast.csv",
                    s3_bucket,
                    f"forecasts/citibike_hourly_forecast_{timestamp}.csv"
                )
                logger.info(
                    f"Hourly forecast saved to S3: s3://{s3_bucket}/forecasts/citibike_hourly_forecast_{timestamp}.csv")

                # Log to MLflow as well
                with mlflow.start_run(
                        run_id=model_run_info[best_model_name]["run_id"]):  # Updated to access run_id from dictionary
                    mlflow.log_artifact("feb_may_2025_forecast.csv", "forecasts")
            except Exception as e:
                logger.error(f"Error uploading forecast to S3: {e}")

        logger.info("Forecast saved locally, to S3, and to MLflow")

        # Create forecast plot for sample station
        # We use the original sample_station ID from before
        logger.info(f"Generating forecast visualization for station ID: {sample_station}...")
        forecast_plot = plot_future_forecast(
            historical_data=feature_sets[best_feature_set],  # Original historical data
            future_predictions=forecast_df,
            station_id=sample_station,
            model_name=best_model_name.replace('citibike-', '').replace('-model', '')
        )

        # Save forecast plots
        forecast_plot.write_html("future_forecast.html")
        save_plot_to_s3(forecast_plot, "future_forecast.html")

        # Log to MLflow
        with mlflow.start_run(
                run_id=model_run_info[best_model_name]["run_id"]):  # Updated to access run_id from dictionary
            mlflow.log_artifact("future_forecast.html", "visualizations")
        logger.info("Future forecast visualization saved")

        # Create summary forecast plot
        summary_plot = plot_forecast_summary(forecast_df, group_by='month')
        summary_plot.write_html("forecast_summary.html")
        save_plot_to_s3(summary_plot, "forecast_summary.html")

        # Log to MLflow
        with mlflow.start_run(
                run_id=model_run_info[best_model_name]["run_id"]):  # Updated to access run_id from dictionary
            mlflow.log_artifact("forecast_summary.html", "visualizations")
        logger.info("Forecast summary visualization saved")

        # Return results
        return results, forecast_df

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    run_training_pipeline()