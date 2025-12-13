"""
Module for data preprocessing and feature generation.
Contains transformer classes compatible with scikit-learn Pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List

from .profiling import profiler


class GeoDistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for calculating Haversine distance to a specific point.
    """

    def __init__(self, center_lat: float, center_lon: float):
        """
        Initializes the transformer with center coordinates.

        Args:
            center_lat (float): Latitude of the central point.
            center_lon (float): Longitude of the central point.
        """
        self.center_lat = center_lat
        self.center_lon = center_lon

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'GeoDistanceTransformer':
        """
        Fit method (does nothing, returns self).

        Args:
            X (pd.DataFrame): Input data.
            y (pd.Series, optional): Target variable.

        Returns:
            GeoDistanceTransformer: The instance itself.
        """
        return self

    @profiler.profile
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the distance to the center and adds the 'distance_to_center' column.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with the new feature.
        """
        X = X.copy()
        if not {'latitude', 'longitude'}.issubset(X.columns):
            rename_dict = {'geo_lat': 'latitude', 'geo_lon': 'longitude'}
            X = X.rename(columns=rename_dict)
            
        if {'latitude', 'longitude'}.issubset(X.columns):
            X['distance_to_center'] = self._haversine(
                X['latitude'], X['longitude'], self.center_lat, self.center_lon
            )
        return X

    @staticmethod
    @profiler.profile
    def _haversine(lat1: pd.Series, lon1: pd.Series, lat2: float, lon2: float) -> pd.Series:
        """
        Calculates the Haversine distance in kilometers.

        Args:
            lat1, lon1: Series of point coordinates.
            lat2, lon2: Coordinates of the fixed point.

        Returns:
            pd.Series: Distances in km.
        """
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer for data cleaning: imputing missing values, removing infinities.
    """

    def __init__(self) -> None:
        self.fill_values_: dict = {}
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataCleaner':
        """
        Computes medians and modes for imputing missing values.

        Args:
            X (pd.DataFrame): Training data.
            y (pd.Series, optional): Target variable.

        Returns:
            DataCleaner: Fitted instance.
        """
        self.numeric_cols_ = X.select_dtypes(include='number').columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude='number').columns.tolist()

        if 'rooms' in X.columns:
            self.fill_values_['rooms'] = X['rooms'].median()
        
        for col in self.numeric_cols_:
            if col not in self.fill_values_:
                self.fill_values_[col] = X[col].median()
        
        for col in self.categorical_cols_:
            self.fill_values_[col] = 'unknown'
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning to the data.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        for col, value in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        
        return X
