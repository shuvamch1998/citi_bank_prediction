import sys
from pathlib import Path
import os
import boto3
from io import StringIO
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytz
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)


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
def fetch_predictions():
    """Fetch the latest predictions from S3"""
    s3_client = get_s3_client()
    bucket = "migration-citi-bike"
    key = "forecasts/citibike_hourly_forecast_latest.csv"

    # Get predictions file from S3
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))

        # Ensure pickup_hour is datetime
        df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])

        # Rename for consistency
        if 'predicted_rides' in df.columns:
            df = df.rename(columns={'predicted_rides': 'predicted_demand'})

        return df
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_historical_data():
    """Fetch historical data for context"""
    s3_client = get_s3_client()
    bucket = "migration-citi-bike"
    key = "citibike/features/full/data.csv"

    # Get data file from S3
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))

        # Ensure pickup_hour is datetime
        df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])

        # Get the most recent 7 days of data for context
        cutoff_time = (datetime.now() - timedelta(days=7)).replace(tzinfo=pytz.UTC)
        df = df[df['pickup_hour'] >= cutoff_time]

        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()


def plot_station_forecast(station_id, predictions_df, historical_df):
    """Create a plot for a specific station"""
    # Get station name/ID for display
    id_col = "start_station_id" if "start_station_id" in predictions_df.columns else "pickup_location_id"

    # Filter data for this station
    station_predictions = predictions_df[predictions_df[id_col] == station_id]
    station_historical = historical_df[
        historical_df[id_col] == station_id] if not historical_df.empty else pd.DataFrame()

    if station_predictions.empty:
        return None

    # Create the plot
    fig = go.Figure()

    # Add historical data if available
    if not station_historical.empty:
        fig.add_trace(go.Scatter(
            x=station_historical['pickup_hour'],
            y=station_historical['rides'],
            mode='lines',
            name='Historical Rides',
            line=dict(color='blue', width=2)
        ))

    # Add predictions
    fig.add_trace(go.Scatter(
        x=station_predictions['pickup_hour'],
        y=station_predictions['predicted_demand'],
        mode='lines+markers',
        name='Predicted Demand',
        line=dict(color='red', width=2)
    ))

    # Add vertical line separating past from future
    now = datetime.now()
    fig.add_shape(
        type="line",
        x0=now,
        y0=0,
        x1=now,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=1, dash="dash"),
    )

    fig.add_annotation(
        x=now,
        y=1,
        yref="paper",
        text="Current Time",
        showarrow=False,
        textangle=-90,
        xanchor="center",
        yanchor="bottom"
    )

    # Update layout
    fig.update_layout(
        title=f"Demand for Station {station_id}",
        xaxis_title="Time",
        yaxis_title="Number of Rides",
        legend_title="Data Type",
        template="plotly_white"
    )

    return fig


# Main app
st.set_page_config(layout="wide", page_title="Citibike Demand Dashboard")

current_date = datetime.now()
st.title(f"Citibike Demand Forecast")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

# Sidebar
st.sidebar.title("Dashboard Controls")
forecast_hours = st.sidebar.slider("Forecast Hours to Display", 12, 168, 48)

# Progress bar
progress_bar = st.sidebar.progress(0)
N_STEPS = 3

# Step 1: Load predictions
with st.spinner("Loading latest predictions..."):
    predictions_df = fetch_predictions()
    if predictions_df.empty:
        st.error("Unable to load predictions. Please check S3 access.")
        st.stop()

    # Filter predictions for the requested time period
    if st.sidebar.checkbox("Show all available forecast data", False):
        filtered_predictions = predictions_df
    else:
        end_time = predictions_df['pickup_hour'].max()
        start_time = end_time - timedelta(hours=forecast_hours)
        filtered_predictions = predictions_df[predictions_df['pickup_hour'].between(start_time, end_time)]
    st.sidebar.write("Predictions loaded successfully.")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Load historical data
with st.spinner("Loading historical data..."):
    historical_df = fetch_historical_data()
    st.sidebar.write("Historical data loaded successfully.")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Create dashboard
with st.spinner("Creating dashboard..."):
    # Display data statistics
    st.subheader("Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Predicted Rides",
            f"{filtered_predictions['predicted_demand'].mean():.1f}",
        )
    with col2:
        st.metric(
            "Maximum Predicted Rides",
            f"{filtered_predictions['predicted_demand'].max():.1f}",
        )
    with col3:
        st.metric(
            "Total Stations",
            f"{filtered_predictions['start_station_id'].nunique()}",
        )

    # Show demand by time
    st.subheader("Predicted Demand Over Time")

    # Group by hour and sum predictions
    id_col = "start_station_id" if "start_station_id" in filtered_predictions.columns else "pickup_location_id"
    demand_by_hour = filtered_predictions.groupby('pickup_hour')['predicted_demand'].sum().reset_index()

    # Create time series chart
    time_fig = px.line(
        demand_by_hour,
        x='pickup_hour',
        y='predicted_demand',
        title='Total Predicted Rides by Hour',
        labels={'pickup_hour': 'Time', 'predicted_demand': 'Predicted Rides'},
        markers=True
    )

    st.plotly_chart(time_fig, use_container_width=True)

    # Show top stations
    st.subheader("Top Stations by Predicted Demand")
    top_stations = (
        filtered_predictions.groupby(id_col)['predicted_demand']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .head(10)
    )

    st.dataframe(top_stations)

    # Show detailed charts for top 5 stations
    st.subheader("Detailed Predictions for Top Stations")
    for station_id in top_stations[id_col].head(5):
        fig = plot_station_forecast(station_id, filtered_predictions, historical_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    progress_bar.progress(3 / N_STEPS)
