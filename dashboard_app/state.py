"""
State Management Module.
"""

import pandas as pd
from typing import Optional
from src.preprocessor import DataCleaner, GeoDistanceTransformer

CITY_CENTER = (55.7558, 37.6173)


class AuditState:
    """
    Singleton class to manage the dataset state across the application lifecycle.

    Attributes:
        df (pd.DataFrame): The current working dataframe (includes generated features).
        original_df (pd.DataFrame): The original dataframe loaded from disk (backup).
    """
    _instance: Optional['AuditState'] = None

    def __new__(cls):
        """
        Ensures only one instance of AuditState exists.
        """
        if cls._instance is None:
            cls._instance = super(AuditState, cls).__new__(cls)
            cls._instance.df = pd.DataFrame()
            cls._instance.original_df = pd.DataFrame()
        return cls._instance

    def load_data(self, path: str = "data/raw_sample.csv") -> None:
        """
        Loads and pre-processes initial data from a CSV file.

        This method performs the following steps:
        1. Loads the CSV file.
        2. Renames geographical columns to standard names.
        3. Applies data cleaning (filling missing values).
        4. Calculates distance to the city center.
        5. Stores the result in the state.

        Args:
            path (str): Path to the CSV data file. Defaults to "data/raw_sample.csv".
        """
        try:
            raw = pd.read_csv(path).sample(2000, random_state=42)
            
            rename_map = {'geo_lat': 'latitude', 'geo_lon': 'longitude'}
            df = raw.rename(columns=rename_map)
            
            cleaner = DataCleaner()
            geo = GeoDistanceTransformer(*CITY_CENTER)
            
            df = cleaner.fit_transform(df)
            df = geo.transform(df)
            
            self.original_df = df.copy()
            self.df = df.copy()
            print("Data loaded into AuditState successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")


state = AuditState()
