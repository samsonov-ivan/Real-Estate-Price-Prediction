"""
Module containing strategies for various machine learning models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor


class ModelStrategy(ABC):
    """Abstract base class for model strategies."""

    @abstractmethod
    def get_model(self, **kwargs: Any) -> BaseEstimator:
        """
        Returns an initialized model.

        Args:
            **kwargs: Hyperparameters for the model.

        Returns:
            BaseEstimator: An instance of a scikit-learn compatible model.
        """
        pass


class CatBoostStrategy(ModelStrategy):
    """Strategy for the CatBoost model."""

    def get_model(self, **kwargs: Any) -> BaseEstimator:
        """Creates a CatBoostRegressor model."""
        params = {"verbose": 0, "allow_writing_files": False}
        params.update(kwargs)
        return CatBoostRegressor(**params)


class LinearRegressionStrategy(ModelStrategy):
    """Strategy for the Linear Regression model."""

    def get_model(self, **kwargs: Any) -> BaseEstimator:
        """Creates a LinearRegression model."""
        return LinearRegression(**kwargs)


class RandomForestStrategy(ModelStrategy):
    """Strategy for the Random Forest model."""

    def get_model(self, **kwargs: Any) -> BaseEstimator:
        """Creates a RandomForestRegressor model."""
        params = {"n_jobs": -1, "random_state": 42}
        params.update(kwargs)
        return RandomForestRegressor(**params)


class ModelFactory:
    """Factory for retrieving model strategies."""

    _strategies: Dict[str, ModelStrategy] = {
        "catboost": CatBoostStrategy(),
        "linear_regression": LinearRegressionStrategy(),
        "random_forest": RandomForestStrategy(),
    }

    @classmethod
    def get_strategy(cls, model_name: str) -> BaseEstimator:
        """
        Returns a model instance by its name.

        Args:
            model_name (str): Name of the model ('catboost', 'linear_regression', 'random_forest').

        Returns:
            BaseEstimator: Initialized model.

        Raises:
            ValueError: If the model with the given name is not implemented.
        """
        if model_name not in cls._strategies:
            raise ValueError(f"Model '{model_name}' is not implemented.")
        return cls._strategies[model_name].get_model()
