"""
Unit tests for the FastAPI application.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch
from api import app

client = TestClient(app)

def test_health_check():
    """
    Tests the /health endpoint to ensure the API is running.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

@patch("api.model_service")
def test_predict_endpoint_valid(mock_service):
    """
    Tests the /predict endpoint with valid input data.
    Mocks the ModelService to return a dummy prediction.
    """
    mock_service.predict.return_value = [15000000.55]
    
    payload = [{
        "latitude": 55.75,
        "longitude": 37.61,
        "area": 50.0,
        "kitchen_area": 10.0,
        "rooms": 2,
        "building_type": "1",
        "id_region": "77"
    }]

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["predicted_price"] == 15000000.55

def test_predict_endpoint_validation_error():
    """
    Tests that the API returns 422 Unprocessable Entity for invalid data types.
    """
    payload = [{
        "latitude": "not_a_number", 
        "area": -50
    }]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

@patch("api.model_service")
def test_predict_service_error(mock_service):
    """
    Tests proper error handling when the internal model fails.
    """
    mock_service.predict.side_effect = RuntimeError("Model exploded")

    payload = [{
        "latitude": 55.75,
        "longitude": 37.61,
        "area": 50.0,
        "kitchen_area": 10.0,
        "rooms": 2,
        "building_type": "1",
        "id_region": "77"
    }]

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Prediction error" in response.json()["detail"]
