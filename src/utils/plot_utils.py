from datetime import timedelta
from typing import Optional, List, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import boto3


def plot_time_series_for_station(
        features: pd.DataFrame,
        targets: pd.Series,
        station_id: str,
        predictions: Optional[pd.Series] = None,
):
    """
    Plots the time series data for a specific station from Citibike data.

    Args:
        features (pd.DataFrame): DataFrame containing feature data, including historical ride counts.
        targets (pd.Series): Series containing the target values (actual ride counts).
        station_id (str): ID of the station to plot.
        predictions (Optional[pd.Series]): Series containing predicted values (optional).

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object showing the time series plot.
    """
    # Extract the specific station's features
    id_col = "start_station_id" if "start_station_id" in features.columns else "pickup_location_id"
    station_features = features[features[id_col] == station_id].sort_values("pickup_hour")

    if len(station_features) == 0:
        raise ValueError(f"No data found for station ID {station_id}")

    # Get the corresponding targets
    station_targets = targets[station_features.index]

    # Identify time series columns (historical ride counts)
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    time_series_columns.sort(key=lambda x: int(x.split('-')[1]), reverse=True)  # Sort by lag time

    # Create a data frame for plotting
    plot_data = pd.DataFrame({
        'time': station_features['pickup_hour'],
        'actual': station_targets
    })

    # Add historical values
    for col in time_series_columns:
        lag_hours = int(col.split('-')[1])
        plot_data[f'lag_{lag_hours}h'] = station_features[col]

    # Create the plot title
    title = f"Rides for Station ID: {station_id}"

    # Create the base line plot
    fig = px.line(
        plot_data,
        x='time',
        y='actual',
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Add a trace for predictions if provided
    if predictions is not None:
        station_preds = predictions[features[id_col] == station_id]
        fig.add_scatter(
            x=plot_data['time'],
            y=station_preds,
            line_color="red",
            mode="markers+lines",
            marker_symbol="x",
            marker_size=8,
            name="Predictions"
        )

    return fig


def plot_model_comparison(test_features, test_targets, predictions, station_id):
    """
    Create a plotly visualization comparing different model predictions for a specific station.

    Args:
        test_features (pd.DataFrame): Test features
        test_targets (pd.Series): Test target values
        predictions (dict): Dictionary of model predictions with model names as keys
        station_id: ID of the station to visualize

    Returns:
        plotly.graph_objects.Figure: Plotly figure with model comparison
    """
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    # Find station ID column
    id_col = "start_station_id" if "start_station_id" in test_features.columns else "pickup_location_id"

    # Create a master dataframe for visualization that combines all data
    visualization_data = []

    # Extract station data from test features
    station_mask = test_features[id_col] == station_id
    station_features = test_features[station_mask].copy()

    if len(station_features) == 0:
        raise ValueError(f"No test data found for station ID {station_id}")

    # Get actual values for this station
    station_idx = station_features.index
    station_targets = test_targets.loc[station_idx]

    # Create base dataframe with timestamps and actual values
    for idx, timestamp in zip(station_features.index, station_features['pickup_hour']):
        row = {
            'pickup_hour': timestamp,
            'Actual': test_targets.loc[idx]
        }
        visualization_data.append(row)

    # Convert to dataframe
    viz_df = pd.DataFrame(visualization_data)

    # Add predictions from each model
    # We need to map predictions to timestamps
    for model_name, model_preds in predictions.items():
        # Create a mapping from index to prediction
        pred_map = {idx: pred for idx, pred in zip(test_features.index, model_preds)}

        # Add predictions for available indices
        viz_df[model_name] = viz_df.index.map(lambda i: pred_map.get(station_idx[i], np.nan))

    # Create a plotly figure
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=viz_df['pickup_hour'],
        y=viz_df['Actual'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=2)
    ))

    # Add each model's predictions
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, model_name in enumerate(predictions.keys()):
        # Skip if no predictions for this model
        if model_name not in viz_df.columns or viz_df[model_name].isna().all():
            continue

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=viz_df['pickup_hour'],
            y=viz_df[model_name],
            mode='lines+markers',
            name=model_name,
            line=dict(color=color)
        ))

    # Set layout
    fig.update_layout(
        title=f'Model Predictions vs Actual for Station {station_id}',
        xaxis_title='Time',
        yaxis_title='Number of Rides',
        legend_title='Models',
        template='plotly_white'
    )

    return fig


def plot_future_forecast(historical_data, future_predictions, station_id, model_name="best_model"):
    """
    Create a plot comparing historical data with future predictions for a specific station.

    Args:
        historical_data (pd.DataFrame): Historical data with actual rides
        future_predictions (pd.DataFrame): Future predictions with predicted rides
        station_id: ID of the station to visualize
        model_name (str): Name of the model used for predictions

    Returns:
        plotly.graph_objects.Figure: Plotly figure with historical and future data
    """
    import plotly.graph_objects as go
    import pandas as pd

    # Find station ID column
    id_col = "start_station_id" if "start_station_id" in historical_data.columns else "pickup_location_id"

    # Filter data for the specific station
    station_historical = historical_data[historical_data[id_col] == station_id].copy()
    station_forecast = future_predictions[future_predictions[id_col] == station_id].copy()

    # Make sure we have data for this station
    if len(station_historical) == 0:
        raise ValueError(f"No historical data found for station ID {station_id}")

    if len(station_forecast) == 0:
        raise ValueError(f"No forecast data found for station ID {station_id}")

    # Sort by time
    station_historical = station_historical.sort_values('pickup_hour')
    station_forecast = station_forecast.sort_values('pickup_hour')

    # Create a plotly figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=station_historical['pickup_hour'],
        y=station_historical['rides'],
        mode='lines',
        name='Historical Rides',
        line=dict(color='blue', width=2)
    ))

    # Add forecast data
    fig.add_trace(go.Scatter(
        x=station_forecast['pickup_hour'],
        y=station_forecast['predicted_rides'],
        mode='lines',
        name='Forecasted Rides',
        line=dict(color='red', width=2)
    ))

    # Add a vertical line to separate historical and forecast data
    # Convert timestamp to string format for plotly
    forecast_start = station_forecast['pickup_hour'].min()

    # Using shapes instead of add_vline to avoid timestamp issues
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=forecast_start,
                y0=0,
                x1=forecast_start,
                y1=1,
                line=dict(
                    color="gray",
                    width=2,
                    dash="dash",
                )
            )
        ]
    )

    # Add annotation for the forecast start
    fig.add_annotation(
        x=forecast_start,
        y=1,
        text="Forecast Start",
        showarrow=True,
        arrowhead=1,
        yref="paper"
    )

    # Set layout
    fig.update_layout(
        title=f'Historical vs Forecast Rides for Station {station_id} ({model_name})',
        xaxis_title='Time',
        yaxis_title='Number of Rides',
        legend_title='Data Type',
        template='plotly_white'
    )

    return fig


def plot_forecast_summary(forecast_df, group_by='month'):
    """
    Create a summary plot of forecasted rides by month, day or hour.

    Args:
        forecast_df (pd.DataFrame): DataFrame with forecasted rides
        group_by (str): How to group the data. Options: 'month', 'day', 'hour'

    Returns:
        plotly.graph_objects.Figure: Plotly figure with forecast summary
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    # Add time components for grouping
    df = forecast_df.copy()
    df['month'] = df['pickup_hour'].dt.month
    df['day'] = df['pickup_hour'].dt.day
    df['hour'] = df['pickup_hour'].dt.hour
    df['day_of_week'] = df['pickup_hour'].dt.dayofweek

    # Find station ID column
    id_col = "start_station_id" if "start_station_id" in df.columns else "pickup_location_id"

    # Calculate average rides by station and time component
    if group_by == 'month':
        grouped = df.groupby([id_col, 'month'])['predicted_rides'].mean().reset_index()
        x_title = 'Month'
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        grouped['month_name'] = grouped['month'].apply(lambda x: month_names[x - 1])
        x_values = grouped['month_name']

    elif group_by == 'day':
        grouped = df.groupby([id_col, 'day'])['predicted_rides'].mean().reset_index()
        x_title = 'Day of Month'
        x_values = grouped['day']

    elif group_by == 'hour':
        grouped = df.groupby([id_col, 'hour'])['predicted_rides'].mean().reset_index()
        x_title = 'Hour of Day'
        x_values = grouped['hour']

    elif group_by == 'day_of_week':
        grouped = df.groupby([id_col, 'day_of_week'])['predicted_rides'].mean().reset_index()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        grouped['day_name'] = grouped['day_of_week'].apply(lambda x: day_names[x])
        x_title = 'Day of Week'
        x_values = grouped['day_name']

    else:
        raise ValueError(f"Invalid group_by value: {group_by}. Must be 'month', 'day', 'hour', or 'day_of_week'")

    # Create a figure
    fig = go.Figure()

    # Get unique stations
    stations = df[id_col].unique()

    # Add a trace for each station
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'brown']

    for i, station in enumerate(stations):
        station_data = grouped[grouped[id_col] == station]

        # Use modulo to cycle through colors if more stations than colors
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=station_data[x_values.name],
            y=station_data['predicted_rides'],
            mode='lines+markers',
            name=f'Station {station}',
            line=dict(color=color)
        ))

    # Set layout
    fig.update_layout(
        title=f'Average Forecasted Rides by {group_by.capitalize()}',
        xaxis_title=x_title,
        yaxis_title='Average Predicted Rides',
        legend_title='Station',
        template='plotly_white'
    )

    return fig

def save_plot_to_s3(fig, filename):
    """
    Save a Plotly figure to HTML and upload to S3.

    Args:
        fig: Plotly figure object
        filename (str): Name for the saved file
    """
    # Save locally
    fig.write_html(filename)

    # Upload to S3 if bucket is configured
    s3_bucket = os.getenv("AWS_S3_BUCKET")
    if s3_bucket:
        try:
            s3_client = boto3.client('s3')
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"plots/{filename.replace('.html', '')}_{timestamp}.html"
            s3_client.upload_file(filename, s3_bucket, s3_key)
            print(f"Plot uploaded to S3: s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"Error uploading plot to S3: {e}")