# src/utils/aws_utils.py
import boto3
import os
from pathlib import Path
from io import StringIO, BytesIO
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def upload_to_s3(
        local_path,
        bucket_name,
        s3_key=None,
        region_name=None
):
    """
    Upload a file to S3

    Args:
        local_path: Path to the local file
        bucket_name: S3 bucket name
        s3_key: S3 object key (folder/filename in S3)
        region_name: AWS region name (optional, uses env var if not specified)
    """
    # If no region provided, try to get from environment
    if not region_name:
        region_name = os.environ.get('AWS_REGION', 'us-east-1')

    # If no S3 key provided, use the filename
    if not s3_key:
        s3_key = Path(local_path).name

    # Create S3 client
    s3_client = boto3.client('s3', region_name=region_name)

    # Upload file
    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=bucket_name,
        Key=s3_key
    )

    return f"s3://{bucket_name}/{s3_key}"


def get_s3_client():
    """
    Create and return an S3 client with credentials from environment variables.

    Returns:
        boto3.client: Configured S3 client
    """
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    )


def load_features_from_s3(bucket, key):
    """
    Load feature data from an S3 bucket.

    Args:
        bucket (str): S3 bucket name
        key (str): Path to the file within the bucket

    Returns:
        pd.DataFrame: DataFrame containing the loaded feature data
    """
    try:
        s3_client = get_s3_client()
        logger.info(f"Loading data from s3://{bucket}/{key}")

        # Determine file format from extension
        if key.endswith('.csv'):
            response = s3_client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(data))
        elif key.endswith('.parquet'):
            response = s3_client.get_object(Bucket=bucket, Key=key)
            buffer = BytesIO(response['Body'].read())
            df = pd.read_parquet(buffer)
        else:
            raise ValueError(f"Unsupported file format for {key}")

        # Convert pickup_hour to datetime if it exists
        if 'pickup_hour' in df.columns:
            df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])

        logger.info(f"Successfully loaded {len(df)} records from S3")
        return df

    except Exception as e:
        logger.error(f"Error loading data from S3: {str(e)}")
        raise


def save_forecast_to_s3(df, bucket, key):
    """
    Save forecast data to an S3 bucket.

    Args:
        df (pd.DataFrame): DataFrame containing forecast data
        bucket (str): S3 bucket name
        key (str): Path to save the file within the bucket
    """
    try:
        s3_client = get_s3_client()
        logger.info(f"Saving forecast to s3://{bucket}/{key}")

        # Determine format based on file extension
        if key.endswith('.csv'):
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_buffer.getvalue()
            )
        elif key.endswith('.parquet'):
            # For parquet, we need to use a different approach
            # First, save to an in-memory buffer using pyarrow directly
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Convert pandas DataFrame to PyArrow Table
            table = pa.Table.from_pandas(df)

            # Write to BytesIO buffer
            parquet_buffer = BytesIO()
            pq.write_table(table, parquet_buffer)

            # Reset buffer position
            parquet_buffer.seek(0)

            # Upload to S3
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=parquet_buffer.getvalue()
            )
        else:
            # Default to CSV if format not specified
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_buffer.getvalue()
            )

        logger.info(f"Successfully saved {len(df)} records to S3")

    except Exception as e:
        logger.error(f"Error saving forecast to S3: {str(e)}")
        raise
