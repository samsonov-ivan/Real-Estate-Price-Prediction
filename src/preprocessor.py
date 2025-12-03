"""
Module for data preprocessing, including handling missing values and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Any, Optional


class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom preprocessor for handling missing values and scaling numerical features.

    This preprocessor fills missing values for categorical and numerical features,
    engineers a new feature (distance to city center), and scales numerical features.
    """

    def __init__(self, city_center_coords: Tuple[float, float] = (55.7558, 37.6173)) -> None:
        """
        Initialize the preprocessor.

        Args:
            city_center_coords (Tuple[float, float]): Coordinates of the city center. Defaults to Moscow.

        Returns:
            None
        """
        self.city_center_coords = city_center_coords
        self.cat_features: list[str] = ['building_type', 'region']
        self.fill_values_: dict[str, Any] = {}

    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and infinities.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        # Additional cleaning steps can be added here if necessary
        X = X.copy()
        if {'geo_lat', 'geo_lon'}.issubset(X.columns):
            X = X.dropna(subset=['geo_lat', 'geo_lon'])
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in self.cat_features:
            if col in X.columns:
                mode = X[col].mode()
                fill = mode.iloc[0] if not mode.empty else 'unknown'
                X[col] = X[col].fillna(fill)
        if 'area' in X.columns:
            X['area'] = X['area'].fillna(X['area'].mean())
        if 'kitchen_area' in X.columns:
            X['kitchen_area'] = X['kitchen_area'].fillna(X['kitchen_area'].mean())
        if 'rooms' in X.columns:
            X['rooms'] = X['rooms'].fillna(X['rooms'].median())
        if 'living_area' not in X.columns and 'area' in X.columns:
            X['living_area'] = X['area'] * 0.6
        elif 'living_area' in X.columns:
            if 'area' in X.columns:
                X['living_area'] = X['living_area'].fillna(X['area'] * 0.6)
            else:
                X['living_area'] = X['living_area'].fillna(X['living_area'].median())
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna('unknown')
        return X

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'CustomPreprocessor':
        """
        Fit the preprocessor to the data.

        Args:
            X (pd.DataFrame): Input data.
            y (Optional[Any]): Target values (ignored).

        Returns:
            CustomPreprocessor: Fitted preprocessor.
        """
        self.fill_values_ = {}
        if 'building_type' in X.columns:
            mode = X['building_type'].mode()
            self.fill_values_['building_type'] = mode.iloc[0] if not mode.empty else 'unknown'
        if 'region' in X.columns:
            mode = X['region'].mode()
            self.fill_values_['region'] = mode.iloc[0] if not mode.empty else 'unknown'
        if 'rooms' in X.columns:
            self.fill_values_['rooms'] = X['rooms'].median()
        if 'area' in X.columns:
            self.fill_values_['living_area'] = X['area'].mean() * 0.6
        if 'kitchen_area' in X.columns:
            self.fill_values_['kitchen_area'] = X['kitchen_area'].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data.
        """
        X = X.copy()
        # Fill missing values
        for feature, fill_value in self.fill_values_.items():
            if feature in X.columns:
                X[feature] = X[feature].fillna(fill_value)
        # Feature engineering: distance to city center
        def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
            """
            Calculate haversine distance.

            Args:
                lat1 (np.ndarray): Latitudes.
                lon1 (np.ndarray): Longitudes.
                lat2 (float): City center latitude.
                lon2 (float): City center longitude.

            Returns:
                np.ndarray: Distances.
            """
            R = 6371  # Earth radius in kilometers
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat / 2) ** 2 +
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
        if 'geo_lat' in X.columns and 'geo_lon' in X.columns:
            X['distance_to_center'] = haversine(
                X['geo_lat'], X['geo_lon'],
                self.city_center_coords[0], self.city_center_coords[1]
            )
        return X
