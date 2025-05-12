import os
import sys
from pathlib import Path
import requests
import pandas as pd
import zipfile

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from src.config.citibike_config import RAW_DATA_DIR


def fetch_raw_citibike_data(year: int, month: int) -> Path:
    """
    Downloads Citi Bike trip data for a specific year and month.

    Args:
        year (int): Year to download data for
        month (int): Month to download data for (1-12)

    Returns:
        Path: Path to the downloaded file
    """
    # Format year and month as parts of the filename
    year_month = f"{year}{month:02d}"

    # Construct the URL based on the observed file pattern (JC-YYYYMM-citibike-tripdata.csv.zip)
    URL = f"https://s3.amazonaws.com/tripdata/JC-{year_month}-citibike-tripdata.csv.zip"

    try:
        response = requests.get(URL)

        if response.status_code == 200:
            # Use the exact filename format from the website
            filename = f"JC-{year_month}-citibike-tripdata.csv.zip"
            path = RAW_DATA_DIR / filename
            open(path, "wb").write(response.content)
            return path
        else:
            raise Exception(f"URL {URL} returned status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        raise


def filter_citibike_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Filters Citi Bike ride data for a specific year and month, removing outliers and invalid records.

    Args:
        rides (pd.DataFrame): DataFrame containing Citi Bike ride data
        year (int): Year to filter for
        month (int): Month to filter for (1-12)

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid rides
    """
    # Validate inputs
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")
    if not isinstance(year, int) or not isinstance(month, int):
        raise ValueError("Year and month must be integers.")

    # Calculate start and end dates for the specified month
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year=year + (month // 12), month=(month % 12) + 1, day=1)

    # Convert date columns to datetime if they aren't already
    # Handle the date format from the Citi Bike data (MM-DD-YYYY HH:MM)
    try:
        # First attempt to parse as is
        rides['started_at'] = pd.to_datetime(rides['started_at'], errors='coerce')
        rides['ended_at'] = pd.to_datetime(rides['ended_at'], errors='coerce')

        # Check if we have NaT values, which might indicate format issues
        if rides['started_at'].isna().any() or rides['ended_at'].isna().any():
            print("Some datetime parsing failed. Trying with explicit format...")
            # Try with explicit format for MM-DD-YYYY HH:MM
            rides['started_at'] = pd.to_datetime(rides['started_at'],
                                                 format='%m-%d-%Y %H:%M',
                                                 errors='coerce')
            rides['ended_at'] = pd.to_datetime(rides['ended_at'],
                                               format='%m-%d-%Y %H:%M',
                                               errors='coerce')
    except Exception as e:
        print(f"Error parsing dates: {str(e)}")
        # If all else fails, try a more flexible approach
        rides['started_at'] = pd.to_datetime(rides['started_at'], errors='coerce')
        rides['ended_at'] = pd.to_datetime(rides['ended_at'], errors='coerce')

    # Drop rows with unparseable dates
    date_parse_filter = ~(rides['started_at'].isna() | rides['ended_at'].isna())
    rides = rides[date_parse_filter].copy()

    print(f"Records after date parsing: {len(rides):,}")

    # Add a duration column for filtering
    rides["duration"] = rides["ended_at"] - rides["started_at"]

    # Define filters
    duration_filter = (rides["duration"] > pd.Timedelta(0)) & (rides["duration"] <= pd.Timedelta(hours=24))
    date_range_filter = (rides["started_at"] >= start_date) & (rides["started_at"] < end_date)

    # Add filter for valid stations (remove stations with null IDs)
    valid_station_filter = ~rides["start_station_id"].isna() & ~rides["end_station_id"].isna()

    # Combine all filters
    final_filter = duration_filter & date_range_filter & valid_station_filter

    # Calculate dropped records
    total_records = len(rides)
    valid_records = final_filter.sum()
    records_dropped = total_records - valid_records
    percent_dropped = (records_dropped / total_records) * 100

    print(f"Total records: {total_records:,}")
    print(f"Valid records: {valid_records:,}")
    print(f"Records dropped: {records_dropped:,} ({percent_dropped:.2f}%)")

    # Keep all original columns
    validated_rides = rides[final_filter].copy()

    # Verify we have data in the correct time range
    if validated_rides.empty:
        raise ValueError(f"No valid rides found for {year}-{month:02} after filtering.")

    return validated_rides


def load_and_process_citibike_data(
        year: int,
        month_start: int,
        month_end: int = None,
) -> pd.DataFrame:
    """
    Load and process Citi Bike ride data for a specified year and range of months.

    Args:
        year (int): Year to load data for
        month_start (int): Starting month (1-12)
        month_end (int, optional): Ending month (1-12). If None, only loads the start month.

    Returns:
        pd.DataFrame: Combined and processed ride data
    """
    # If month_end is not specified, only load month_start
    if month_end is None:
        month_end = month_start

    # Validate input
    if not (1 <= month_start <= 12) or not (1 <= month_end <= 12):
        raise ValueError("Months must be between 1 and 12.")
    if month_end < month_start:
        raise ValueError("End month must be greater than or equal to start month.")

    # Generate list of months to process
    months = list(range(month_start, month_end + 1))

    # List to store DataFrames for each month
    monthly_rides = []

    for month in months:
        # Construct the year-month string for the filename
        year_month = f"{year}{month:02d}"

        # Construct the file path using the observed filename pattern
        file_path = RAW_DATA_DIR / f"JC-{year_month}-citibike-tripdata.csv.zip"

        try:
            # Download the file if it doesn't exist
            if not file_path.exists():
                print(f"Downloading data for {year}-{month:02d}...")
                fetch_raw_citibike_data(year, month)
                print(f"Successfully downloaded data for {year}-{month:02d}.")
            else:
                print(f"File already exists for {year}-{month:02d}.")

            # Load the data from the zip file
            print(f"Loading data for {year}-{month:02d}...")
            try:
                # First, inspect the zip contents
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Get all filenames in the zip
                    filenames = zip_ref.namelist()

                    # Filter out any __MACOSX directory files or hidden files
                    csv_files = [f for f in filenames if
                                 not f.startswith('__MACOSX/') and not f.startswith('.') and f.endswith('.csv')]

                    if not csv_files:
                        raise ValueError(f"No CSV files found in ZIP file {file_path}")

                    # Use the first CSV file found
                    csv_filename = csv_files[0]
                    print(f"Using CSV file: {csv_filename}")

                    # Extract and read the CSV file
                    with zip_ref.open(csv_filename) as csv_file:
                        rides = pd.read_csv(csv_file)

                # Check if required columns exist
                required_columns = [
                    "ride_id", "rideable_type", "started_at", "ended_at",
                    "start_station_name", "start_station_id", "end_station_name",
                    "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng",
                    "member_casual"
                ]

                missing_columns = [col for col in required_columns if col not in rides.columns]
                if missing_columns:
                    print(f"Warning: Missing columns in data: {missing_columns}")

                # Filter and process the data
                rides = filter_citibike_data(rides, year, month)
                print(f"Successfully processed data for {year}-{month:02d}.")

                # Append the processed DataFrame to the list
                monthly_rides.append(rides)

            except Exception as e:
                print(f"Error reading CSV from zip for {year}-{month:02d}: {str(e)}")
                continue

        except FileNotFoundError:
            print(f"File not found for {year}-{month:02d}. Skipping...")
        except Exception as e:
            print(f"Error processing data for {year}-{month:02d}: {str(e)}")
            continue

    # Combine all monthly data
    if not monthly_rides:
        raise Exception(f"No data could be loaded for the year {year} and specified months: {months}")

    print("Combining all monthly data...")
    combined_rides = pd.concat(monthly_rides, ignore_index=True)
    print("Data loading and processing complete!")

    return combined_rides




