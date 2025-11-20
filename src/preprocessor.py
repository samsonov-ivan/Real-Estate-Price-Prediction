import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom preprocessor for handling missing values and scaling numerical features.
    Parameters:
    """

    def __init__(self, city_center_coords=(55.7558, 37.6173)):
        # Initialize with default city center coordinates (Moscow)
        self.city_center_coords = city_center_coords
        self.cat_features = ['building_type', 'region']

    def fit(self, X, y=None):
        
        self.fill_values_ = {
            'building_type': X['building_type'].mode()[0],
            'region': X['region'].mode()[0],
            'rooms': X['rooms'].median(),
            'living_area': X['area'].mean() * 0.6,
            'kitchen_area': X['area'].mean() * 0.2,
        }
        return self
    
    def transform(self, X):
        X = X.copy()

        # Fill missing values
        for feature, fill_value in self.fill_values_.items():
            X[feature].fillna(fill_value, inplace=True)

        # Feature engineering: distance to city center
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in kilometers
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat / 2) ** 2 +
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        X['distance_to_center'] = haversine(
            X['latitude'], X['longitude'],
            self.city_center_coords[0], self.city_center_coords[1]
        )

        # Scaling numerical features
        num_features = ['area', 'living_area', 'kitchen_area', 'rooms', 'distance_to_center']
        for feature in num_features:
            mean = X[feature].mean()
            std = X[feature].std()
            X[feature] = (X[feature] - mean) / std

        return X
