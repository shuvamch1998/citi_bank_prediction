import pandas as pd
import numpy as np
from typing import List, Dict, Union

def fill_missing_rides_full_range(df, hour_col, station_col, rides_col):
    """
    Fills in missing rides for all hours in the range and all unique stations.

    Parameters:
    - df: DataFrame with columns [hour_col, station_col, rides_col]
    - hour_col: Name of the column containing hourly timestamps
    - station_col: Name of the column containing station IDs
    - rides_col: Name of the column containing ride counts

    Returns:
    - DataFrame with missing hours and stations filled in with 0 rides
    """
    # Ensure the hour column is in datetime format
    df[hour_col] = pd.to_datetime(df[hour_col])

    # Get the full range of hours (from min to max) with hourly frequency
    full_hours = pd.date_range(start=df[hour_col].min(), end=df[hour_col].max(), freq='h')

    # Get all unique station IDs
    all_stations = df[station_col].unique()

    # Create a DataFrame with all combinations of hours and stations
    full_combinations = pd.DataFrame(
        [(hour, station) for hour in full_hours for station in all_stations],
        columns=[hour_col, station_col]
    )

    # Merge the original DataFrame with the full combinations DataFrame
    merged_df = pd.merge(full_combinations, df, on=[hour_col, station_col], how='left')

    # Fill missing rides with 0
    merged_df[rides_col] = merged_df[rides_col].fillna(0).astype(int)

    return merged_df


def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw ride data into time series format.

    Args:
        rides: DataFrame with started_at and other columns

    Returns:
        pd.DataFrame: Time series data with filled gaps
    """
    # Make a copy to avoid modifying the original
    rides_copy = rides.copy()

    # Ensure we have the started_at column
    if 'started_at' not in rides_copy.columns:
        if 'pickup_datetime' in rides_copy.columns:
            rides_copy['started_at'] = rides_copy['pickup_datetime']
        else:
            raise ValueError("Neither 'started_at' nor 'pickup_datetime' found in DataFrame")

    # Floor datetime to hour efficiently
    rides_copy['pickup_hour'] = pd.to_datetime(rides_copy['started_at']).dt.floor('h')

    # Ensure we have the start_station_id column
    if 'start_station_id' not in rides_copy.columns:
        if 'pickup_station_id' in rides_copy.columns:
            rides_copy['start_station_id'] = rides_copy['pickup_station_id']
        else:
            raise ValueError("Neither 'start_station_id' nor 'pickup_station_id' found in DataFrame")

    # Aggregate and fill gaps
    agg_rides = rides_copy.groupby(['pickup_hour', 'start_station_id']).size().reset_index(name='rides')

    # Fill in missing hours and stations with 0 rides
    agg_rides_all_slots = (
        fill_missing_rides_full_range(agg_rides, 'pickup_hour', 'start_station_id', 'rides')
        .sort_values(['start_station_id', 'pickup_hour'])
        .reset_index(drop=True)
    )

    # Set data types for efficiency
    agg_rides_all_slots['rides'] = agg_rides_all_slots['rides'].astype('int16')

    return agg_rides_all_slots


def process_data_for_modeling(df: pd.DataFrame, top_n: int = 3) -> dict:
    """
    Complete end-to-end feature engineering process from raw data.

    Args:
        df: DataFrame with raw Citi Bike data
        top_n: Number of top stations to include

    Returns:
        dict: Dictionary containing feature sets for different model types
    """
    # Extract features from raw data
    #raw_features = extract_raw_features(df)

    # Get top stations
    top_stations = get_top_stations(df, n=top_n)

    # Filter for only top stations
    df_top_stations = df[df['start_station_id'].isin(top_stations)]

    # Transform into time series format
    ts_data = transform_raw_data_into_ts_data(df_top_stations)

    # Add time series features
    ts_features = prepare_time_series_features(ts_data)

    # Create baseline features (simple lags only)
    baseline_columns = ['pickup_hour', 'start_station_id', 'rides',
                        'rides_lag_24h', 'rides_lag_168h']  # Yesterday and last week
    baseline_features = ts_features[baseline_columns].copy()

    # Apply feature selection for reduced feature set
    from sklearn.feature_selection import SelectKBest, f_regression
    # First create X and y
    X = ts_features.drop(['pickup_hour', 'start_station_id', 'rides'], axis=1)
    y = ts_features['rides']
    # Select top 10 features
    selector = SelectKBest(f_regression, k=10)
    selector.fit(X, y)
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    # Create reduced feature dataframe
    reduced_features = ts_features[['pickup_hour', 'start_station_id', 'rides'] + selected_features]

    return {
        'baseline_features': baseline_features,
        'full_features': ts_features,
        'reduced_features': reduced_features
    }

def get_top_stations(df: pd.DataFrame, n: int = 3) -> List:
    """
    Get the top N most frequently used stations.

    Args:
        df: DataFrame with start_station_id column
        n: Number of top stations to return

    Returns:
        List of top station IDs
    """
    # Check for the column name - could be start_station_id or pickup_station_id
    station_col = 'start_station_id' if 'start_station_id' in df.columns else 'pickup_station_id'

    if station_col not in df.columns:
        raise ValueError(f"DataFrame must have either 'start_station_id' or 'pickup_station_id' column")

    # Get top N stations
    top_stations = df[station_col].value_counts().nlargest(n).index.tolist()

    # Optionally print the top stations if verbose
    if 'start_station_name' in df.columns or 'pickup_station_name' in df.columns:
        station_name_col = 'start_station_name' if 'start_station_name' in df.columns else 'pickup_station_name'
        print(f"Top {n} stations by usage:")
        for i, station_id in enumerate(top_stations, 1):
            mask = df[station_col] == station_id
            if mask.any():
                station_name = df[mask][station_name_col].iloc[0]
                count = mask.sum()
                print(f"{i}. {station_name} (ID: {station_id}): {count} trips")

    return top_stations

def create_geographical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on geographical data in the dataset.

    Args:
        df: DataFrame with latitude and longitude columns

    Returns:
        DataFrame with additional geographical features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Calculate distance between start and end points if coordinates are available
    if all(col in df.columns for col in ['start_lat', 'start_lng', 'end_lat', 'end_lng']):
        # Function to calculate distance between two points using Haversine formula
        def haversine_distance(lat1, lon1, lat2, lon2):
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Radius of Earth in kilometers
            return c * r

        # Calculate distances
        result_df['distance_km'] = haversine_distance(
            result_df['start_lat'],
            result_df['start_lng'],
            result_df['end_lat'],
            result_df['end_lng']
        )

        # Flag for short/medium/long trips
        result_df['trip_length'] = pd.cut(
            result_df['distance_km'],
            bins=[0, 1, 3, 100],
            labels=['short', 'medium', 'long']
        )

    return result_df

def create_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to trip characteristics.

    Args:
        df: DataFrame with trip data

    Returns:
        DataFrame with additional trip features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Add duration features if start and end times are available
    if all(col in df.columns for col in ['started_at', 'ended_at']):
        # Convert to datetime if needed
        for col in ['started_at', 'ended_at']:
            if not pd.api.types.is_datetime64_dtype(df[col]):
                result_df[col] = pd.to_datetime(df[col])

        # Calculate trip duration in minutes
        result_df['duration_minutes'] = (result_df['ended_at'] - result_df['started_at']).dt.total_seconds() / 60

        # Create duration categories
        result_df['duration_category'] = pd.cut(
            result_df['duration_minutes'],
            bins=[0, 5, 15, 30, 60, 1440],
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )

    # Add bike type features if available
    if 'rideable_type' in df.columns:
        # One-hot encode bike types
        bike_dummies = pd.get_dummies(df['rideable_type'], prefix='bike')
        result_df = pd.concat([result_df, bike_dummies], axis=1)

    # Add user type features if available
    if 'member_casual' in df.columns:
        # One-hot encode user types
        user_dummies = pd.get_dummies(df['member_casual'], prefix='user')
        result_df = pd.concat([result_df, user_dummies], axis=1)

    return result_df

def create_time_features(df: pd.DataFrame, datetime_col: str = 'started_at') -> pd.DataFrame:
    """
    Create time-based features from a datetime column.

    Args:
        df: DataFrame with datetime column
        datetime_col: Name of the datetime column

    Returns:
        DataFrame with additional time features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_dtype(df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Extract basic time components
    result_df['hour'] = result_df[datetime_col].dt.hour
    result_df['day_of_week'] = result_df[datetime_col].dt.dayofweek
    result_df['day_of_month'] = result_df[datetime_col].dt.day
    result_df['week_of_year'] = result_df[datetime_col].dt.isocalendar().week
    result_df['month'] = result_df[datetime_col].dt.month
    result_df['year'] = result_df[datetime_col].dt.year

    # Add derived time features
    result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
    result_df['is_morning_rush'] = ((result_df['hour'] >= 7) & (result_df['hour'] <= 9) &
                                    ~result_df['is_weekend'].astype(bool)).astype(int)
    result_df['is_evening_rush'] = ((result_df['hour'] >= 17) & (result_df['hour'] <= 19) &
                                    ~result_df['is_weekend'].astype(bool)).astype(int)

    # Add cyclical encoding to handle periodicity
    result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
    result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
    result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
    result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)

    return result_df

def create_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to stations.

    Args:
        df: DataFrame with station information

    Returns:
        DataFrame with additional station features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Create station-specific features if station IDs are available
    if 'start_station_id' in df.columns:
        # Get counts by station to identify popularity
        station_counts = df['start_station_id'].value_counts()

        # Add station popularity
        result_df['station_popularity'] = result_df['start_station_id'].map(station_counts)

        # Create popularity categories
        result_df['station_popularity_category'] = pd.qcut(
            result_df['station_popularity'],
            q=4,
            labels=['low', 'medium', 'high', 'very_high']
        )

    # Check if both start and end stations are available
    if 'start_station_id' in df.columns and 'end_station_id' in df.columns:
        # Create a feature for trips that start and end at the same station
        result_df['round_trip'] = (result_df['start_station_id'] == result_df['end_station_id']).astype(int)

    return result_df

def extract_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all useful features from raw Citi Bike data.

    Args:
        df: DataFrame with raw Citi Bike data

    Returns:
        DataFrame with all extracted features
    """
    # Apply all feature extraction functions
    result_df = df.copy()

    # Apply time features to started_at
    if 'started_at' in result_df.columns:
        result_df = create_time_features(result_df, 'started_at')

    # Apply station features
    result_df = create_station_features(result_df)

    # Apply trip features
    result_df = create_trip_features(result_df)

    # Apply geographical features
    result_df = create_geographical_features(result_df)

    return result_df

def prepare_time_series_features(ts_df: pd.DataFrame, window_size: int = 28*24) -> pd.DataFrame:
    """
    Prepare time series features for the aggregated hourly data.

    Args:
        ts_df: DataFrame with time series data (pickup_hour, start_station_id, rides)
        window_size: Maximum window size for lag features

    Returns:
        DataFrame with time series features
    """
    # Make a copy to avoid modifying the original
    result_df = ts_df.copy()

    # Add time features based on pickup_hour
    result_df = create_time_features(result_df, 'pickup_hour')

    # Add lag features for each station
    for station_id in result_df['start_station_id'].unique():
        station_mask = result_df['start_station_id'] == station_id
        station_data = result_df[station_mask].copy().sort_values('pickup_hour')

        # Important lag values
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 24*7, 24*14, 24*21, 24*28]

        # Add lag features
        for lag in lag_hours:
            if lag <= window_size:
                lag_col = f'rides_lag_{lag}h'
                result_df.loc[station_mask, lag_col] = station_data['rides'].shift(lag).values

        # Add rolling statistics
        for window in [24, 24*7, 24*28]:
            if window <= window_size:
                # Rolling mean
                result_df.loc[station_mask, f'rolling_mean_{window}h'] = (
                    station_data['rides'].rolling(window=window, min_periods=1).mean().values
                )

                # Rolling max
                result_df.loc[station_mask, f'rolling_max_{window}h'] = (
                    station_data['rides'].rolling(window=window, min_periods=1).max().values
                )

                # Rolling standard deviation
                result_df.loc[station_mask, f'rolling_std_{window}h'] = (
                    station_data['rides'].rolling(window=window, min_periods=1).std().values
                )

    # Fill NaN values in lag features
    lag_cols = [col for col in result_df.columns if 'lag' in col or 'rolling' in col]
    result_df[lag_cols] = result_df[lag_cols].fillna(0)

    return result_df