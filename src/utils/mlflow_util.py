import logging
import os
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_mlflow_tracking():
    """
    Set up MLflow tracking with DagsHub.
    """
    # Get DagsHub credentials from environment
    dagshub_username = os.environ.get("DAGSHUB_USERNAME")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    repo_name = os.environ.get("DAGSHUB_REPO_NAME")

    # Set up tracking URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri and dagshub_username and repo_name:
        # Construct tracking URI if not explicitly provided
        tracking_uri = f"https://dagshub.com/{dagshub_username}/{repo_name}.mlflow"

    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI or DagsHub credentials not provided")

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    # Set up authentication
    if dagshub_username and dagshub_token:
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        logger.info(f"DagsHub authentication credentials set for user: {dagshub_username}")

    return mlflow


def log_model_to_mlflow(
        model,
        input_data,
        experiment_name,
        metric_name="metric",
        model_name=None,
        params=None,
        score=None,
):
    """
    Log a trained model, parameters, and metrics to MLflow.

    Parameters:
    - model: Trained model object (e.g., sklearn model).
    - input_data: Input data used for training (for signature inference).
    - experiment_name: Name of the MLflow experiment.
    - metric_name: Name of the metric to log (e.g., "RMSE", "accuracy").
    - model_name: Optional name for the registered model.
    - params: Optional dictionary of hyperparameters to log.
    - score: Optional evaluation metric to log.

    Returns:
    - dict: Dictionary with model_info and run_id
    """
    try:
        # Set the experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment set to: {experiment_name}")

        # Start an MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log hyperparameters if provided
            if params:
                mlflow.log_params(params)
                logger.info(f"Logged parameters: {params}")

            # Log metrics if provided
            if score is not None:
                mlflow.log_metric(metric_name, score)
                logger.info(f"Logged {metric_name}: {score}")

            # Infer the model signature
            try:
                # For some custom models, we might need to handle signature inference differently
                predictions = model.predict(input_data)
                signature = infer_signature(input_data, predictions)
                logger.info("Model signature inferred.")
            except Exception as e:
                logger.warning(f"Could not infer model signature: {e}. Proceeding without signature.")
                signature = None

            # Determine the model name
            if not model_name:
                model_name = model.__class__.__name__

            # Log the model with registration
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data,
                registered_model_name=model_name,  # This registers the model
            )
            logger.info(f"Model logged and registered with name: {model_name}")

            # Return both model_info and run_id as a dictionary
            return {
                "model_info": model_info,
                "run_id": run_id
            }

    except Exception as e:
        logger.error(f"An error occurred while logging to MLflow: {e}")
        raise

def register_best_model(
        model_name,
        run_id,
        stage="Production"
):
    """
    Register the best model version to the specified stage.

    Args:
        model_name (str): Name of the registered model
        run_id (str): Run ID of the run containing the model
        stage (str): Stage to transition to (Staging, Production, Archived)
    """
    try:
        client = MlflowClient()

        # Find the version associated with this run
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if not versions:
            logger.warning(f"No model versions found for run {run_id}")
            return None

        version = versions[0]
        version_num = version.version

        # Transition to the specified stage
        client.transition_model_version_stage(
            name=model_name,
            version=version_num,
            stage=stage,
            archive_existing_versions=(stage == "Production")  # Archive other production models
        )

        logger.info(f"Model {model_name} version {version_num} transitioned to {stage}")
        return version

    except Exception as e:
        logger.error(f"Error registering model to {stage}: {e}")
        raise


def load_production_model(model_name):
    """
    Load a model from the MLflow registry in Production stage.

    Args:
        model_name (str): Name of the registered model

    Returns:
        The loaded model
    """
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        logger.info(f"Loaded Production version of model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name} from Production: {e}")

        # Try loading latest version as fallback
        try:
            client = MlflowClient()
            latest_versions = client.get_latest_versions(model_name)
            if latest_versions:
                version = latest_versions[0].version
                model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
                logger.info(f"Loaded version {version} of model: {model_name}")
                return model
            else:
                logger.error(f"No versions found for model: {model_name}")
                return None
        except Exception as inner_e:
            logger.error(f"Error loading latest version: {inner_e}")
            return None

def get_latest_full_model_from_mlflow(model_name="citibike-full-model"):
    """
    Get the latest version of the full feature model from MLflow registry.

    Args:
        model_name: Name of the model in the MLflow registry

    Returns:
        The loaded model
    """
    # Set up MLflow tracking
    mlflow = set_mlflow_tracking()
    client = MlflowClient()

    try:
        logger.info(f"Searching for model: {model_name}")

        # First try to get all models with this name (without filtering by stage)
        all_versions = []
        try:
            # Simple filter just by name
            all_versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as e:
            logger.warning(f"Error when searching by name: {str(e)}")
            # Try alternative approach - get all registered models
            registered_models = client.search_registered_models()
            for rm in registered_models:
                if rm.name == model_name:
                    # Get all versions for this model
                    all_versions = client.get_latest_versions(model_name)
                    break

        if not all_versions:
            # If we still can't find versions, try one more approach
            try:
                # Try to list versions directly
                all_versions = client.get_latest_versions(model_name)
            except Exception as inner_e:
                logger.error(f"Failed to get versions via get_latest_versions: {str(inner_e)}")
                raise ValueError(f"No versions found for model: {model_name}")

        # Find the production model if available
        production_version = None
        latest_version = None

        for version in all_versions:
            # Check if this is a production version
            if hasattr(version, 'current_stage') and version.current_stage == 'Production':
                production_version = version.version
                logger.info(f"Found production model: {model_name} version {production_version}")
                break

            # Track the latest version by number
            if latest_version is None or (hasattr(version, 'version') and int(version.version) > int(latest_version)):
                latest_version = version.version

        # Use production version if found, otherwise use latest
        model_version = production_version if production_version else latest_version

        if not model_version:
            raise ValueError(f"Could not determine the version for model: {model_name}")

        logger.info(f"Using model version: {model_name} version {model_version}")

        # Load the model
        model_uri = f"models:/{model_name}/{model_version}"
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        return model

    except Exception as e:
        logger.error(f"Error loading model from MLflow: {str(e)}")

        # Fallback: try to find model from local path if available
        try:
            logger.info("Attempting to load model from local models directory...")
            import joblib
            from pathlib import Path

            # Try to find a model file in the models directory
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pkl"))
                if model_files:
                    logger.info(f"Found local model: {model_files[0]}")
                    return joblib.load(model_files[0])
        except Exception as local_e:
            logger.error(f"Failed to load local model: {str(local_e)}")

        # If all else fails, raise the original error
        raise
