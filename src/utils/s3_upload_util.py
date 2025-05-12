# src/utils/aws_utils.py
import boto3
import os
from pathlib import Path


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