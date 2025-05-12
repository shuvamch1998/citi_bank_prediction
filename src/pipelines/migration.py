import os
import time
import boto3
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def export_feature_store_via_athena():
    """
    Export AWS Feature Store data by querying through Athena and saving to S3.
    """
    # Get AWS credentials and configuration
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    target_bucket = os.getenv("TARGET_S3_BUCKET")

    if not target_bucket:
        raise ValueError("TARGET_S3_BUCKET environment variable not set")

    # Initialize AWS clients with explicit credentials
    session_kwargs = {"region_name": aws_region}
    if aws_access_key and aws_secret_key:
        session_kwargs.update({
            "aws_access_key_id": aws_access_key,
            "aws_secret_access_key": aws_secret_key
        })

    boto_session = boto3.Session(**session_kwargs)
    sagemaker_client = boto_session.client('sagemaker')
    athena_client = boto_session.client('athena')
    glue_client = boto_session.client('glue')
    s3_client = boto_session.client('s3')

    # Define our feature groups
    feature_group_names = [
        'citibike-citibike_baseline_features',
        'citibike-citibike_full_features',
        'citibike-citibike_reduced_features'
    ]

    # Target paths for cleaned export
    target_paths = {
        'citibike-citibike_baseline_features': 'citibike/features/baseline',
        'citibike-citibike_full_features': 'citibike/features/full',
        'citibike-citibike_reduced_features': 'citibike/features/reduced'
    }

    # Discover feature groups and their Glue tables
    feature_groups = {}

    for fg_name in feature_group_names:
        try:
            # Get feature group details
            fg_details = sagemaker_client.describe_feature_group(FeatureGroupName=fg_name)

            # Check if it has offline store config
            if 'OfflineStoreConfig' in fg_details and 'DataCatalogConfig' in fg_details['OfflineStoreConfig']:
                catalog_config = fg_details['OfflineStoreConfig']['DataCatalogConfig']
                database = catalog_config.get('Database')
                table_name = catalog_config.get('TableName')
                catalog = catalog_config.get('Catalog', 'AWSGlue')

                if database and table_name:
                    feature_groups[fg_name] = {
                        'database': database,
                        'table_name': table_name,
                        'catalog': catalog,
                        'target_path': target_paths.get(fg_name, f'citibike/features/{fg_name}')
                    }
                    logger.info(f"Found feature group {fg_name} in Glue database {database}, table {table_name}")
                else:
                    logger.warning(f"Feature group {fg_name} doesn't have complete catalog config")
            else:
                logger.warning(f"Feature group {fg_name} doesn't have offline store with data catalog")
        except Exception as e:
            logger.error(f"Error getting feature group {fg_name}: {str(e)}")

    if not feature_groups:
        logger.error("No feature groups with Glue catalog found")
        return

    # Create Athena query results location if it doesn't exist
    athena_output_location = f"s3://{target_bucket}/athena-query-results/"

    # Set up Athena workgroup if needed
    workgroup_name = "FeatureStoreExport"
    try:
        # Check if workgroup exists
        athena_client.get_work_group(WorkGroup=workgroup_name)
        logger.info(f"Using existing Athena workgroup: {workgroup_name}")
    except athena_client.exceptions.InvalidRequestException:
        # Create workgroup
        logger.info(f"Creating Athena workgroup: {workgroup_name}")
        athena_client.create_work_group(
            Name=workgroup_name,
            Configuration={
                'ResultConfiguration': {
                    'OutputLocation': athena_output_location
                },
                'EnforceWorkGroupConfiguration': True,
                'PublishCloudWatchMetricsEnabled': True,
                'BytesScannedCutoffPerQuery': 10737418240  # 10GB
            },
            Description='Workgroup for Feature Store exports'
        )

    # Process each feature group
    for fg_name, fg_info in feature_groups.items():
        logger.info(f"Processing feature group: {fg_name}")

        database = fg_info['database']
        table_name = fg_info['table_name']
        target_path = fg_info['target_path']

        try:
            # First, list the available partitions to get year/month values
            logger.info(f"Getting partitions for {database}.{table_name}")

            # Describe table to find partition keys
            table_info = glue_client.get_table(
                DatabaseName=database,
                Name=table_name
            )

            # Check if table has partitions
            partition_keys = []
            if 'PartitionKeys' in table_info['Table']:
                partition_keys = [key['Name'] for key in table_info['Table']['PartitionKeys']]

            logger.info(f"Partition keys: {partition_keys}")

            # Get partitions
            if partition_keys and 'year' in partition_keys and 'month' in partition_keys:
                # Get distinct year/month combinations
                partitions_query = f"""
                    SELECT DISTINCT year, month 
                    FROM {database}.{table_name}
                    ORDER BY year, month
                """

                # Execute query to list partitions
                partitions_response = athena_client.start_query_execution(
                    QueryString=partitions_query,
                    QueryExecutionContext={
                        'Database': database
                    },
                    WorkGroup=workgroup_name
                )

                partitions_query_id = partitions_response['QueryExecutionId']

                # Wait for query to complete
                partition_results = wait_for_athena_query(athena_client, partitions_query_id)

                if not partition_results:
                    logger.error("Failed to get partitions")
                    continue

                # Extract year/month values
                year_months = []
                for row in partition_results['ResultSet']['Rows'][1:]:  # Skip header
                    # First column is year, second is month
                    year = row['Data'][0]['VarCharValue']
                    month = row['Data'][1]['VarCharValue']
                    year_months.append((year, month))

                logger.info(f"Found {len(year_months)} year/month partitions")

                # Process each year/month partition
                for year, month in year_months:
                    logger.info(f"Exporting data for {year}-{month}")

                    # Create query to extract data for this partition
                    export_query = f"""
                        SELECT * 
                        FROM {database}.{table_name}
                        WHERE year = '{year}' AND month = '{month}'
                    """

                    # Set up result configuration with partition-specific path
                    result_location = f"{athena_output_location}{fg_name}/{year}/{month}/"

                    # Execute query
                    export_response = athena_client.start_query_execution(
                        QueryString=export_query,
                        QueryExecutionContext={
                            'Database': database
                        },
                        ResultConfiguration={
                            'OutputLocation': result_location
                        },
                        WorkGroup=workgroup_name
                    )

                    export_query_id = export_response['QueryExecutionId']

                    # Wait for query to complete
                    logger.info(f"Waiting for Athena query to complete for {year}-{month}...")
                    query_status = wait_for_athena_query_status(athena_client, export_query_id)

                    if query_status == 'SUCCEEDED':
                        # Get the result file location
                        query_results = athena_client.get_query_execution(
                            QueryExecutionId=export_query_id
                        )

                        result_file = query_results['QueryExecution']['ResultConfiguration']['OutputLocation']
                        # Format: s3://bucket/path/query-id.csv

                        # Copy result to our clean target location
                        source_bucket = result_file.split('//')[1].split('/')[0]
                        source_key = '/'.join(result_file.split('//')[1].split('/')[1:])

                        target_key = f"{target_path}/{year}/{month}/data.csv"

                        # Copy file
                        copy_source = {'Bucket': source_bucket, 'Key': source_key}
                        s3_client.copy(copy_source, target_bucket, target_key)

                        logger.info(f"Exported data to s3://{target_bucket}/{target_key}")

                        # Convert to parquet (optional)
                        # If you want to convert to parquet, you'd need to:
                        # 1. Download the CSV
                        # 2. Use pandas to convert to parquet
                        # 3. Upload the parquet file

                    else:
                        logger.error(f"Query for {year}-{month} failed with status: {query_status}")
            else:
                # No partitioning, get all data at once
                logger.info(f"Table {table_name} doesn't have standard year/month partitions, exporting all data")

                export_query = f"SELECT * FROM {database}.{table_name}"

                # Execute query
                export_response = athena_client.start_query_execution(
                    QueryString=export_query,
                    QueryExecutionContext={
                        'Database': database
                    },
                    WorkGroup=workgroup_name
                )

                export_query_id = export_response['QueryExecutionId']

                # Wait for query to complete
                logger.info("Waiting for Athena query to complete...")
                query_status = wait_for_athena_query_status(athena_client, export_query_id)

                if query_status == 'SUCCEEDED':
                    # Get the result file location
                    query_results = athena_client.get_query_execution(
                        QueryExecutionId=export_query_id
                    )

                    result_file = query_results['QueryExecution']['ResultConfiguration']['OutputLocation']

                    # Copy result to our clean target location
                    source_bucket = result_file.split('//')[1].split('/')[0]
                    source_key = '/'.join(result_file.split('//')[1].split('/')[1:])

                    target_key = f"{target_path}/data.csv"

                    # Copy file
                    copy_source = {'Bucket': source_bucket, 'Key': source_key}
                    s3_client.copy(copy_source, target_bucket, target_key)

                    logger.info(f"Exported data to s3://{target_bucket}/{target_key}")
                else:
                    logger.error(f"Query failed with status: {query_status}")

        except Exception as e:
            logger.error(f"Error processing feature group {fg_name}: {str(e)}")

    logger.info("Export completed")


def wait_for_athena_query_status(athena_client, query_id, max_retries=60, sleep_seconds=5):
    """Wait for an Athena query to complete and return its status."""
    for _ in range(max_retries):
        try:
            response = athena_client.get_query_execution(QueryExecutionId=query_id)
            state = response['QueryExecution']['Status']['State']

            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                return state

            time.sleep(sleep_seconds)
        except Exception as e:
            logger.error(f"Error checking query status: {str(e)}")
            return None

    return None  # Timed out


def wait_for_athena_query(athena_client, query_id, max_retries=60, sleep_seconds=5):
    """Wait for an Athena query to complete and return the results."""
    status = wait_for_athena_query_status(athena_client, query_id, max_retries, sleep_seconds)

    if status == 'SUCCEEDED':
        # Get the results
        results = athena_client.get_query_results(QueryExecutionId=query_id)
        return results
    else:
        logger.error(f"Query failed with status: {status}")
        return None


if __name__ == "__main__":
    export_feature_store_via_athena()