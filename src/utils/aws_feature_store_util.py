import os
import boto3
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

# Load environment variables from .env
load_dotenv()


class AWSFeatureStore:
    """
    AWS SageMaker Feature Store wrapper for the Citibike project.
    """

    def __init__(self, region_name=None, prefix="citibike", role_arn=None):
        """
        Initialize AWS Feature Store client.

        Args:
            region_name (str, optional): AWS region name, defaults to env variable
            prefix (str): Prefix for feature group names
            role_arn (str, optional): SageMaker execution role ARN, defaults to env variable
        """
        # Get configuration from environment variables if not provided
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.prefix = prefix
        self.role_arn = role_arn or os.getenv("SAGEMAKER_ROLE_ARN")

        # Set up AWS clients
        self.boto_session = boto3.Session(region_name=self.region_name)
        self.sagemaker_client = self.boto_session.client(service_name="sagemaker")
        self.featurestore_runtime = self.boto_session.client(service_name="sagemaker-featurestore-runtime")

        # Initialize SageMaker session
        self.sagemaker_session = Session(
            boto_session=self.boto_session,
            sagemaker_client=self.sagemaker_client,
            sagemaker_featurestore_runtime_client=self.featurestore_runtime
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized AWSFeatureStore with region: {self.region_name}")

    def _prepare_data_for_feature_store(self, df, entity_column, timestamp_column):
        """
        Prepare DataFrame for feature store by ensuring required columns are present
        and converting data types to be compatible with Feature Store.

        Args:
            df (pd.DataFrame): Input DataFrame
            entity_column (str): Entity column name
            timestamp_column (str): Timestamp column name

        Returns:
            pd.DataFrame: Prepared DataFrame
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # STEP 1: Convert ALL datetime columns to strings first (including pickup_hour)
        # This ensures we don't miss any datetime columns
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                self.logger.info(f"Converting datetime column {col} to string format")
                df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # STEP 2: Entity ID column - rename if needed
        if entity_column != "entity_id":
            df_copy["entity_id"] = df_copy[entity_column].astype(str)  # Convert to string to be safe

        # STEP 3: Handle event_time column
        # If timestamp_column is already converted to string in step 1
        if timestamp_column in df_copy.columns:
            # If timestamp_column is now a string (after conversion in step 1)
            df_copy["event_time"] = df_copy[timestamp_column]
        else:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

        # STEP 4: Add record insertion time (as a string timestamp)
        df_copy["record_insertion_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Final check: ensure no datetime columns remain
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                self.logger.warning(f"Column {col} is still datetime type, converting again")
                df_copy[col] = df_copy[col].astype(str)

        return df_copy

    def _get_feature_definitions(self, df):
        """
        Create explicit feature definitions to avoid type inference issues.

        Args:
            df (pd.DataFrame): DataFrame with features

        Returns:
            list: List of feature definitions
        """
        feature_definitions = []

        for column_name, dtype in df.dtypes.items():
            # Skip any columns that should be ignored
            if column_name in ['_ignored_column']:
                continue

            # Map pandas dtypes to Feature Store types
            if pd.api.types.is_integer_dtype(dtype):
                feature_type = "Integral"  # Correct type for integers
            elif pd.api.types.is_float_dtype(dtype):
                feature_type = "Fractional"
            elif pd.api.types.is_bool_dtype(dtype):
                feature_type = "String"  # Feature Store doesn't have boolean, use string
            elif pd.api.types.is_datetime64_dtype(dtype):
                feature_type = "String"  # Store datetimes as strings
            else:
                feature_type = "String"  # Default to string for other types

            feature_definitions.append(
                {"FeatureName": column_name, "FeatureType": feature_type}
            )

        return feature_definitions

    def _get_default_bucket(self):
        """
        Get S3 bucket for Feature Store from environment variable.

        Returns:
            str: S3 bucket name to use for offline storage
        """
        # Check for environment variable first
        bucket_name = os.getenv("FEATURE_STORE_BUCKET")

        if bucket_name:
            self.logger.info(f"Using S3 bucket from environment variable: {bucket_name}")
            return bucket_name

        # Fall back to SageMaker default bucket
        default_bucket = self.sagemaker_session.default_bucket()
        self.logger.info(f"Using SageMaker default bucket: {default_bucket}")
        return default_bucket

    def _get_role_arn(self):
        """
        Get SageMaker execution role ARN from environment variable.

        Returns:
            str: ARN of the SageMaker execution role
        """
        if not self.role_arn:
            raise ValueError(
                "SageMaker execution role ARN not provided and SAGEMAKER_ROLE_ARN "
                "environment variable not set. Please set it to the ARN of the "
                "SageMaker execution role."
            )
        return self.role_arn

    def create_feature_group(self, feature_group_name, df, entity_column, timestamp_column, description=""):
        """
        Create a new feature group in AWS Feature Store.

        Args:
            feature_group_name (str): Name of the feature group
            df (pd.DataFrame): DataFrame with features
            entity_column (str): Entity column name
            timestamp_column (str): Timestamp column name
            description (str): Description of the feature group

        Returns:
            FeatureGroup: Created feature group
        """
        # Prefix the feature group name
        full_feature_group_name = f"{self.prefix}-{feature_group_name}"

        # Prepare data
        prepared_df = self._prepare_data_for_feature_store(df, entity_column, timestamp_column)

        # Create feature group
        feature_group = FeatureGroup(
            name=full_feature_group_name,
            sagemaker_session=self.sagemaker_session
        )

        # Load feature definitions explicitly to avoid inference issues
        feature_group.load_feature_definitions(data_frame=prepared_df)

        # Get bucket name from environment or default
        bucket_name = self._get_default_bucket()
        s3_uri = f"s3://{bucket_name}/{full_feature_group_name}"

        # Determine record identifier and event time feature names
        record_identifier_name = "entity_id"  # Column used to identify unique records
        event_time_feature_name = "event_time"  # Timestamp column

        # Create the feature group in AWS
        self.logger.info(f"Creating feature group {full_feature_group_name} with offline store at {s3_uri}")
        feature_group.create(
            s3_uri=s3_uri,
            record_identifier_name=record_identifier_name,
            event_time_feature_name=event_time_feature_name,
            description=description,
            enable_online_store=True,
            role_arn=self._get_role_arn()
        )

        # Wait for feature group to be created with polling
        self.logger.info(f"Waiting for feature group {full_feature_group_name} to be created")
        self._wait_for_feature_group_creation(feature_group)

        return feature_group

    def insert_features(self, feature_group, df, entity_column, timestamp_column, description=""):
        """
        Insert features into AWS Feature Store.

        Args:
            feature_group (str): Name of the feature group
            df (pd.DataFrame): DataFrame with features
            entity_column (str): Entity column name
            timestamp_column (str): Timestamp column name
            description (str): Description of the feature group
        """
        # Prefix the feature group name
        full_feature_group_name = f"{self.prefix}-{feature_group}"

        # Check if feature group exists, create if not
        try:
            feature_group_obj = FeatureGroup(
                name=full_feature_group_name,
                sagemaker_session=self.sagemaker_session
            )
            feature_group_obj.describe()
            self.logger.info(f"Feature group {full_feature_group_name} exists")
        except Exception as e:
            self.logger.info(f"Feature group {full_feature_group_name} does not exist or error: {str(e)}")
            self.logger.info(f"Creating feature group {full_feature_group_name}")
            feature_group_obj = self.create_feature_group(
                feature_group,
                df,
                entity_column,
                timestamp_column,
                description
            )

        # Prepare data
        prepared_df = self._prepare_data_for_feature_store(df, entity_column, timestamp_column)

        # Ingest features
        self.logger.info(f"Ingesting {len(prepared_df)} records into {full_feature_group_name}")
        feature_group_obj.ingest(data_frame=prepared_df, max_workers=5, wait=True)
        self.logger.info(f"Successfully ingested features into {full_feature_group_name}")


    def _wait_for_feature_group_creation(self, feature_group, max_wait_time_seconds=900, poll_interval_seconds=10):
        """
        Wait for feature group creation by polling the status.

        Args:
            feature_group (FeatureGroup): Feature group to wait for
            max_wait_time_seconds (int): Maximum time to wait in seconds
            poll_interval_seconds (int): Time between polls in seconds

        Returns:
            None
        """
        import time

        start_time = time.time()
        while (time.time() - start_time) < max_wait_time_seconds:
            try:
                # Get the latest feature group description
                response = feature_group.describe()

                # Check if creation is complete
                status = response.get("FeatureGroupStatus")
                self.logger.info(f"Current feature group status: {status}")

                if status == "Created":
                    self.logger.info(f"Feature group {feature_group.name} successfully created")
                    return
                elif status in ["CreateFailed", "DeleteFailed"]:
                    error_message = response.get("FailureReason", "Unknown error")
                    raise Exception(f"Feature group creation failed: {error_message}")

                # Wait before polling again
                time.sleep(poll_interval_seconds)

            except Exception as e:
                if "ResourceNotFound" in str(e):
                    # Resource not found yet, keep waiting
                    time.sleep(poll_interval_seconds)
                else:
                    # Re-raise other exceptions
                    raise

        # If we get here, we timed out
        raise TimeoutError(f"Timed out waiting for feature group {feature_group.name} to be created")