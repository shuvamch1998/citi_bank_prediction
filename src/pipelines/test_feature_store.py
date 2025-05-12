"""Query and display contents of feature store groups."""
import boto3
import pandas as pd
from dotenv import load_dotenv
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session
import os
load_dotenv()

def inspect_feature_store_contents():


    print("Inspecting AWS Feature Store contents...")

    # Create boto3 session
    boto_session = boto3.Session(region_name=os.getenv("AWS_DEFAULT_REGION"))
    sagemaker_client = boto_session.client(service_name="sagemaker")
    featurestore_runtime = boto_session.client(service_name="sagemaker-featurestore-runtime")

    # Initialize SageMaker session
    sagemaker_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime
    )

    # Feature group names
    feature_groups = [
        "citibike-citibike_baseline_features",
        "citibike-citibike_full_features",
        "citibike-citibike_reduced_features"
    ]

    for fg_name in feature_groups:
        print(f"\n=== Examining feature group: {fg_name} ===")

        # Create feature group object
        fg = FeatureGroup(name=fg_name, sagemaker_session=sagemaker_session)

        try:
            # Get feature group info
            fg_info = fg.describe()
            print(f"Status: {fg_info.get('FeatureGroupStatus')}")
            print(f"Creation time: {fg_info.get('CreationTime')}")
            print(f"Feature definitions: {len(fg_info.get('FeatureDefinitions', []))} features")

            # Print a few feature definitions as examples
            print("\nSample feature definitions:")
            for i, feature in enumerate(fg_info.get('FeatureDefinitions', [])[:5]):  # Show first 5
                print(f"  {feature.get('FeatureName')}: {feature.get('FeatureType')}")

            # Try to query the feature group directly
            print("\nAttempting to get records from online store...")

            try:
                # Try with known station IDs
                station_ids = ["HB102", "JC115", "HB101"]  # Top stations from your logs

                for station_id in station_ids:
                    try:
                        # Get record for this station
                        record_response = featurestore_runtime.get_record(
                            FeatureGroupName=fg_name,
                            RecordIdentifierValueAsString=station_id
                        )

                        # Print record details
                        record = record_response.get('Record', [])
                        if record:
                            print(f"\nFound record for station {station_id}:")
                            # Print just a few key features to keep output manageable
                            key_features = ['entity_id', 'event_time', 'pickup_hour', 'rides', 'rides_lag_24h']
                            for feature in record:
                                if feature['FeatureName'] in key_features:
                                    print(f"  {feature['FeatureName']}: {feature.get('ValueAsString', '')}")
                            print(f"  (plus {len(record) - len(key_features)} more features)")
                            break  # Found one record, no need to check more stations
                        else:
                            print(f"No records found for station {station_id}")
                    except Exception as record_e:
                        print(f"Error getting record for station {station_id}: {str(record_e)}")
                        continue
            except Exception as e:
                print(f"Error retrieving online records: {str(e)}")

            # Check offline store structure in S3
            print("\nInspecting offline store in S3...")
            s3_client = boto_session.client('s3')
            bucket_name = os.getenv("FEATURE_STORE_BUCKET")

            if not bucket_name:
                print("FEATURE_STORE_BUCKET environment variable not set, skipping S3 inspection")
                continue

            # Get account ID from the feature group ARN
            fg_arn = fg_info.get('FeatureGroupArn', '')
            # ARN format: arn:aws:sagemaker:region:account:feature-group/name
            account_id = fg_arn.split(':')[4] if fg_arn else None

            if account_id:
                # Build the expected S3 path structure
                base_path = f"{fg_name}/{account_id}/sagemaker/"

                # List directories
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=base_path,
                    Delimiter='/'
                )

                if 'CommonPrefixes' in response:
                    # Found subdirectories
                    for prefix in response['CommonPrefixes']:
                        region_dir = prefix['Prefix']
                        print(f"Found region directory: {region_dir}")

                        # Check for offline-store
                        offline_response = s3_client.list_objects_v2(
                            Bucket=bucket_name,
                            Prefix=region_dir,
                            Delimiter='/'
                        )

                        if 'CommonPrefixes' in offline_response:
                            for offline_prefix in offline_response['CommonPrefixes']:
                                if 'offline-store' in offline_prefix['Prefix']:
                                    offline_dir = offline_prefix['Prefix']
                                    print(f"Found offline store directory: {offline_dir}")

                                    # Find the specific feature group directory (includes a timestamp)
                                    fg_dir_response = s3_client.list_objects_v2(
                                        Bucket=bucket_name,
                                        Prefix=offline_dir,
                                        Delimiter='/'
                                    )

                                    if 'CommonPrefixes' in fg_dir_response:
                                        for fg_dir_prefix in fg_dir_response['CommonPrefixes']:
                                            fg_dir = fg_dir_prefix['Prefix']
                                            print(f"Found feature group directory: {fg_dir}")

                                            # Check for data directory
                                            data_response = s3_client.list_objects_v2(
                                                Bucket=bucket_name,
                                                Prefix=fg_dir,
                                                Delimiter='/'
                                            )

                                            if 'CommonPrefixes' in data_response:
                                                for data_prefix in data_response['CommonPrefixes']:
                                                    if 'data' in data_prefix['Prefix']:
                                                        data_dir = data_prefix['Prefix']
                                                        print(f"Found data directory: {data_dir}")

                                                        # Check for year partitions
                                                        year_response = s3_client.list_objects_v2(
                                                            Bucket=bucket_name,
                                                            Prefix=data_dir,
                                                            Delimiter='/'
                                                        )

                                                        if 'CommonPrefixes' in year_response:
                                                            print("Year partitions:")
                                                            for year_prefix in year_response['CommonPrefixes']:
                                                                year_dir = year_prefix['Prefix']
                                                                print(f"  {year_dir}")

                                                                # Check for month partitions
                                                                month_response = s3_client.list_objects_v2(
                                                                    Bucket=bucket_name,
                                                                    Prefix=year_dir,
                                                                    Delimiter='/'
                                                                )

                                                                if 'CommonPrefixes' in month_response:
                                                                    print("  Month partitions:")
                                                                    for month_prefix in month_response[
                                                                        'CommonPrefixes']:
                                                                        month_dir = month_prefix['Prefix']
                                                                        print(f"    {month_dir}")

                                                                        # Count parquet files
                                                                        files_response = s3_client.list_objects_v2(
                                                                            Bucket=bucket_name,
                                                                            Prefix=month_dir
                                                                        )

                                                                        if 'Contents' in files_response:
                                                                            parquet_files = [obj for obj in
                                                                                             files_response['Contents']
                                                                                             if obj['Key'].endswith(
                                                                                    '.parquet')]
                                                                            print(
                                                                                f"      {len(parquet_files)} parquet files")

                                                                            # Show a sample filename
                                                                            if parquet_files:
                                                                                print(
                                                                                    f"      Sample file: {os.path.basename(parquet_files[0]['Key'])}")
                else:
                    print(f"No directories found at base path: {base_path}")
            else:
                print("Could not determine account ID from feature group ARN")

        except Exception as e:
            print(f"Error inspecting feature group {fg_name}: {str(e)}")

    print("\nFeature Store inspection complete")

if __name__ == "__main__":
    # Run the inspection
    inspect_feature_store_contents()

    # Or run your regular pipeline
    # run_training_pipeline()