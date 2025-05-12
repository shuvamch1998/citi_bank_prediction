import sys
from pathlib import Path
import os
import boto3
from io import StringIO
from datetime import datetime, timedelta, date, time
import pytz

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Page configuration
st.set_page_config(
    page_title="Citibike Demand Forecast",
    page_icon="🚲",
    layout="wide"
)


# S3 access functions
def get_s3_client():
    """Create and return an S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    )

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_predictions(start_date, end_date):
    """
    Fetch prediction data from S3 for a given date range

    Args:
        start_date (date): Start date to filter data
        end_date (date): End date to filter data

    Returns:
        pd.DataFrame: Dataframe with prediction data
    """
    s3_client = get_s3_client()
    bucket = "migration-citi-bike"
    key = "forecasts/citibike_hourly_forecast_latest.csv"

    # Get predictions file from S3
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))

        # Display debug information
        st.sidebar.info(f"Loaded {len(df)} predictions from S3")

        # Ensure pickup_hour is datetime with timezone
        df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])

        # Convert date range to datetime with UTC timezone
        start_datetime = datetime.combine(start_date, time.min).replace(tzinfo=pytz.UTC)
        end_datetime = datetime.combine(end_date, time.max).replace(tzinfo=pytz.UTC)

        # Filter data within the date range
        df = df[(df['pickup_hour'] >= start_datetime) & (df['pickup_hour'] <= end_datetime)]

        # Rename predicted_rides to predicted_demand for consistency
        if 'predicted_rides' in df.columns:
            df = df.rename(columns={'predicted_rides': 'predicted_demand'})

        # Display debug information
        if len(df) > 0:
            st.sidebar.success(
                f"Filtered to {len(df)} predictions from {df['pickup_hour'].min()} to {df['pickup_hour'].max()}")
        else:
            st.sidebar.warning("No predictions found in the selected date range")

        return df
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return pd.DataFrame()


def plot_station_forecast(station_id, predictions_df):
    """Create a plot for a specific station"""
    id_col = "start_station_id" if "start_station_id" in predictions_df.columns else "pickup_location_id"

    # Filter data for this station
    station_predictions = predictions_df[predictions_df[id_col] == station_id]

    if station_predictions.empty:
        return None

    # Create the plot
    fig = go.Figure()

    # Add predictions
    fig.add_trace(go.Scatter(
        x=station_predictions['pickup_hour'],
        y=station_predictions['predicted_demand'],
        mode='lines+markers',
        name='Predicted Demand',
        line=dict(color='red', width=2)
    ))

    # Update layout
    fig.update_layout(
        title=f"Predicted Demand for Station {station_id}",
        xaxis_title="Time",
        yaxis_title="Predicted Number of Rides",
        template="plotly_white",
        height=400
    )

    return fig


def plot_demand_by_time(predictions_df):
    """Create a plot showing overall demand by time"""
    id_col = "start_station_id" if "start_station_id" in predictions_df.columns else "pickup_location_id"

    # Group by hour and sum predictions
    demand_by_hour = predictions_df.groupby('pickup_hour')['predicted_demand'].sum().reset_index()

    # Create time series chart
    fig = px.line(
        demand_by_hour,
        x='pickup_hour',
        y='predicted_demand',
        title='Total Predicted Rides by Hour',
        labels={'pickup_hour': 'Time', 'predicted_demand': 'Predicted Rides'},
        markers=True
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Predicted Rides",
        template="plotly_white",
        height=400
    )

    return fig


def plot_daily_pattern(predictions_df):
    """Create a plot showing average demand by hour of day"""
    # Extract hour of day
    predictions_df['hour_of_day'] = predictions_df['pickup_hour'].dt.hour

    # Group by hour of day
    hourly_pattern = predictions_df.groupby('hour_of_day')['predicted_demand'].mean().reset_index()

    # Create bar chart
    fig = px.bar(
        hourly_pattern,
        x='hour_of_day',
        y='predicted_demand',
        title='Average Predicted Rides by Hour of Day',
        labels={'hour_of_day': 'Hour of Day', 'predicted_demand': 'Average Predicted Rides'}
    )

    # Improve layout
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        xaxis_title="Hour of Day",
        yaxis_title="Average Predicted Rides",
        template="plotly_white",
        height=400
    )

    return fig


def plot_weekly_pattern(predictions_df):
    """Create a plot showing average demand by day of week"""
    # Extract day of week
    predictions_df['day_of_week'] = predictions_df['pickup_hour'].dt.dayofweek
    predictions_df['day_name'] = predictions_df['pickup_hour'].dt.day_name()

    # Group by day of week
    weekly_pattern = predictions_df.groupby(['day_of_week', 'day_name'])['predicted_demand'].mean().reset_index()

    # Sort by day of week
    weekly_pattern = weekly_pattern.sort_values('day_of_week')

    # Create bar chart
    fig = px.bar(
        weekly_pattern,
        x='day_name',
        y='predicted_demand',
        title='Average Predicted Rides by Day of Week',
        labels={'day_name': 'Day of Week', 'predicted_demand': 'Average Predicted Rides'}
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Average Predicted Rides",
        template="plotly_white",
        height=400
    )

    return fig


# Main app interface
st.title("Citibike Demand Forecast")
st.write("This dashboard shows forecasted demand for Citibike rides.")

# Sidebar for user input
st.sidebar.header("Settings")

# Date range selection based on prediction data availability
min_date = date(2025, 5, 12)  # First date in the dataset (May 12, 2025)
max_date = date(2025, 8, 10)  # Last date in the dataset (August 10, 2025)

# Default to July 2025 (a month in the middle of your range)
default_start_date = date(2025, 7, 1)  # Start of July
default_end_date = date(2025, 7, 7)  # One week in July

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    value=default_end_date,
    min_value=start_date,
    max_value=max_date
)

if start_date > end_date:
    st.sidebar.error("End date must be after start date")
    st.stop()

# Station filter
with st.sidebar.expander("Station Filters", expanded=False):
    filter_by_station = st.checkbox("Show specific station details", value=False)
    if filter_by_station:
        available_stations = ["All Stations"]  # Placeholder, will be populated after loading data

# Fetch prediction data
with st.spinner("Fetching forecasts..."):
    predictions_df = fetch_predictions(start_date, end_date)

if predictions_df.empty:
    st.warning("No forecast data available for the selected date range.")
    st.stop()

# Extract station IDs for filter
id_col = "start_station_id" if "start_station_id" in predictions_df.columns else "pickup_location_id"
available_stations = ["All Stations"] + sorted(predictions_df[id_col].unique().tolist())

# Update station selection if filtering by station
selected_station = None
if filter_by_station:
    with st.sidebar.expander("Station Filters", expanded=True):
        selected_station = st.selectbox("Select Station", available_stations)

# Display data statistics
st.subheader("Forecast Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Average Predicted Rides",
        f"{predictions_df['predicted_demand'].mean():.1f}",
    )
with col2:
    st.metric(
        "Maximum Predicted Rides",
        f"{predictions_df['predicted_demand'].max():.1f}",
    )
with col3:
    st.metric(
        "Total Stations",
        f"{predictions_df[id_col].nunique()}",
    )

# Show demand by time (overall trend)
st.subheader("Predicted Demand Over Time")
time_fig = plot_demand_by_time(predictions_df)
st.plotly_chart(time_fig, use_container_width=True)

# Show daily and weekly patterns
col1, col2 = st.columns(2)
with col1:
    daily_pattern_fig = plot_daily_pattern(predictions_df)
    st.plotly_chart(daily_pattern_fig, use_container_width=True)

with col2:
    weekly_pattern_fig = plot_weekly_pattern(predictions_df)
    st.plotly_chart(weekly_pattern_fig, use_container_width=True)

# Show top stations
st.subheader("Top Stations by Predicted Demand")
top_stations = (
    predictions_df.groupby(id_col)['predicted_demand']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    .head(10)
)
top_stations.columns = ["Station ID", "Average Predicted Rides"]
st.dataframe(top_stations, use_container_width=True, hide_index=True)

# If a specific station is selected, show its forecast
if filter_by_station and selected_station != "All Stations":
    st.subheader(f"Detailed Forecast for Station {selected_station}")
    station_fig = plot_station_forecast(selected_station, predictions_df)
    if station_fig:
        st.plotly_chart(station_fig, use_container_width=True)
    else:
        st.warning(f"No forecast data available for station {selected_station}")
# If no specific station is selected but filter_by_station is true, show the top 5 stations
elif filter_by_station:
    st.subheader("Detailed Forecasts for Top Stations")
    for station_id in top_stations["Station ID"].head(5):
        station_fig = plot_station_forecast(station_id, predictions_df)
        if station_fig:
            st.plotly_chart(station_fig, use_container_width=True)

# Add data download option
st.subheader("Download Forecast Data")
csv = predictions_df.to_csv(index=False)
st.download_button(
    label="Download Full Forecast as CSV",
    data=csv,
    file_name=f"citibike_forecast_{start_date}_to_{end_date}.csv",
    mime="text/csv",
)

# Add a footer with information
st.markdown("---")
st.caption("Forecast generated using machine learning models trained on historical Citibike data.")
st.caption(f"Data range: {predictions_df['pickup_hour'].min()} to {predictions_df['pickup_hour'].max()}")
