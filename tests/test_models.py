import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from src.models import ModelFactory

def test_model_factory_linear():
    model = ModelFactory.get_strategy("linear_regression")
    assert isinstance(model, BaseEstimator)
    assert hasattr(model, 'regressor') or isinstance(model, LinearRegression)

def test_model_factory_error():
    with pytest.raises(ValueError):
        ModelFactory.get_strategy("unsupported_model")
