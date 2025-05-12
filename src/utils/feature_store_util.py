# src/utils/simple_feature_store.py

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path


class SQLFeatureStore:
    """
    A lightweight feature store implementation using SQLite.
    No external dependencies beyond pandas and standard library.
    """

    def __init__(self, db_path):
        """Initialize the feature store with a database path."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._setup_tables()

    def _setup_tables(self):
        """Create the necessary database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Table to track feature groups
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_groups (
            name TEXT PRIMARY KEY,
            description TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        ''')

        # Table to store features
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            feature_group TEXT,
            entity_id TEXT,
            timestamp TEXT,
            feature_data TEXT,
            PRIMARY KEY (feature_group, entity_id, timestamp),
            FOREIGN KEY (feature_group) REFERENCES feature_groups(name)
        )
        ''')

        self.conn.commit()

    def create_feature_group(self, name, description=""):
        """Create or update a feature group."""
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()

        # Check if feature group exists
        cursor.execute("SELECT name FROM feature_groups WHERE name = ?", (name,))
        exists = cursor.fetchone() is not None

        if exists:
            # Update existing group
            cursor.execute(
                "UPDATE feature_groups SET description = ?, updated_at = ? WHERE name = ?",
                (description, now, name)
            )
        else:
            # Create new group
            cursor.execute(
                "INSERT INTO feature_groups VALUES (?, ?, ?, ?)",
                (name, description, now, now)
            )

        self.conn.commit()
        return name

    def list_feature_groups(self):
        """List all available feature groups."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, description, created_at FROM feature_groups")

        return [
            {
                "name": name,
                "description": desc,
                "created_at": created_at
            }
            for name, desc, created_at in cursor.fetchall()
        ]

    def insert_features(self, feature_group, df, entity_column="start_station_id",
                        timestamp_column="pickup_hour", description=None):
        """
        Insert features into the store.

        Args:
            feature_group: Name of the feature group
            df: DataFrame containing the features
            entity_column: Name of the column that contains the entity ID
            timestamp_column: Name of the column that contains the timestamp
            description: Optional description of the feature group
        """
        # Create or update feature group
        self.create_feature_group(
            name=feature_group,
            description=description or f"Feature group for {feature_group}"
        )

        # Copy DataFrame to avoid modifying the original
        df = df.copy()

        # Ensure timestamp column is in string format
        if pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df[timestamp_column] = df[timestamp_column].astype(str)

        # Prepare rows for insertion
        rows = []
        for _, row in df.iterrows():
            # Get entity ID and timestamp
            entity_id = str(row[entity_column])
            timestamp = str(row[timestamp_column])

            # Create feature data dictionary (excluding entity and timestamp columns)
            feature_data = {}
            for col in df.columns:
                if col not in [entity_column, timestamp_column]:
                    value = row[col]
                    # Convert to Python native types for JSON serialization
                    if pd.isna(value):
                        feature_data[col] = None
                    elif pd.api.types.is_integer_dtype(type(value)):
                        feature_data[col] = int(value)
                    elif pd.api.types.is_float_dtype(type(value)):
                        feature_data[col] = float(value)
                    else:
                        feature_data[col] = str(value)

            # Convert to JSON
            feature_json = json.dumps(feature_data)
            rows.append((feature_group, entity_id, timestamp, feature_json))

        # Insert data in batches
        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?)",
            rows
        )

        self.conn.commit()
        print(f"Inserted {len(rows)} rows into feature group '{feature_group}'")

    def get_features(self, feature_group, entity_ids=None, start_time=None, end_time=None):
        """
        Retrieve features from the store.

        Args:
            feature_group: Name of the feature group
            entity_ids: List of entity IDs to filter (optional)
            start_time: Start time for filtering (inclusive, optional)
            end_time: End time for filtering (inclusive, optional)

        Returns:
            DataFrame with the requested features
        """
        # Build query with parameters
        query = "SELECT entity_id, timestamp, feature_data FROM features WHERE feature_group = ?"
        params = [feature_group]

        # Add filters if specified
        if entity_ids:
            if isinstance(entity_ids, (str, int)):
                entity_ids = [str(entity_ids)]
            else:
                entity_ids = [str(eid) for eid in entity_ids]

            placeholders = ','.join(['?'] * len(entity_ids))
            query += f" AND entity_id IN ({placeholders})"
            params.extend(entity_ids)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(str(start_time))

        if end_time:
            query += " AND timestamp <= ?"
            params.append(str(end_time))

        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(query, params)

        # Process results into a DataFrame
        results = []
        for entity_id, timestamp, feature_data in cursor.fetchall():
            # Start with entity_id and timestamp
            row = {
                "entity_id": entity_id,
                "timestamp": timestamp
            }

            # Add all feature values
            features = json.loads(feature_data)
            row.update(features)

            results.append(row)

        # Return empty DataFrame if no results
        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert timestamp to datetime if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def close(self):
        """Close the database connection."""
        self.conn.close()
        print(f"Connection to {self.db_path} closed.")