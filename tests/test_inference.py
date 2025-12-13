"""
Unit tests for the Inference Module.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.inference import ModelService

@patch("src.inference.Path.exists")
@patch("src.inference.joblib.load")
def test_model_service_init_success(mock_load, mock_exists):
    """
    Verifies that the ModelService initializes correctly when the model file exists.
    """
    mock_exists.return_value = True
    mock_load.return_value = "dummy_model_object"

    service = ModelService("models/test_model.pkl")

    assert service.model == "dummy_model_object"
    mock_load.assert_called_once()

@patch("src.inference.Path.exists")
def test_model_service_file_not_found(mock_exists):
    """
    Verifies that a FileNotFoundError is raised if the model path is invalid.
    """
    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError) as excinfo:
        ModelService("ghost_model.pkl")
    
    assert "Model file not found" in str(excinfo.value)

@patch("src.inference.Path.exists")
@patch("src.inference.joblib.load")
def test_predict_logic_success(mock_load, mock_exists):
    """
    Verifies the prediction logic, including the mathematical transformation
    (expm1) of the model's output.
    """
    mock_exists.return_value = True
    
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([4.6151205])
    
    mock_load.return_value = mock_pipeline
    
    service = ModelService("dummy.pkl")
    
    input_data = [
        {"area": 50, "rooms": 2}
    ]
    
    predictions = service.predict(input_data)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert predictions[0] == pytest.approx(100.0, abs=0.001)
    
    args, _ = mock_pipeline.predict.call_args
    assert "area" in args[0].columns

@patch("src.inference.Path.exists")
@patch("src.inference.joblib.load")
def test_predict_error_handling(mock_load, mock_exists):
    """
    Verifies that internal model errors during prediction are caught
    and raised as a RuntimeError.
    """
    mock_exists.return_value = True
    
    mock_pipeline = MagicMock()
    mock_pipeline.predict.side_effect = ValueError("Number of features mismatch")
    
    mock_load.return_value = mock_pipeline
    
    service = ModelService("dummy.pkl")
    
    with pytest.raises(RuntimeError) as excinfo:
        service.predict([{"bad_data": 0}])
    
    assert "Prediction failed" in str(excinfo.value)
    assert "mismatch" in str(excinfo.value)
