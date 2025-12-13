"""
FastAPI Application for Real Estate Price Prediction.

Endpoints:
- POST /predict: Generate price prediction.
- GET /health: Check API status.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List
from src.inference import ModelService

app = FastAPI(
    title="Real Estate Price Predictor",
    description="API for the Mysterious 'F' Organization Audit",
    version="1.0"
)

MODEL_PATH = "models/best_model.pkl"

try:
    model_service = ModelService(MODEL_PATH)
except FileNotFoundError:
    model_service = None
    print(f"WARNING: Model not found at {MODEL_PATH}. Please run 'python main.py' first.")


class ApartmentFeatures(BaseModel):
    """Schema for apartment features input."""
    
    latitude: float = Field(..., description="Latitude coordinate (e.g. 55.75)")
    longitude: float = Field(..., description="Longitude coordinate (e.g. 37.61)")
    area: float = Field(..., gt=0, description="Total area in square meters")
    kitchen_area: float = Field(..., gt=0, description="Kitchen area in square meters")
    rooms: int = Field(..., ge=0, description="Number of rooms")

    building_type: int = Field(..., description="Type of building (categorical code)")
    id_region: int = Field(..., description="Region ID (categorical code)")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "latitude": 55.7558,
                    "longitude": 37.6173,
                    "area": 45.5,
                    "kitchen_area": 9.0,
                    "rooms": 2,
                    "building_type": 2,
                    "id_region": 77
                }
            ]
        }
    )

class PredictionResponse(BaseModel):
    """Schema for prediction output."""
    predicted_price: float

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    status = "ok" if model_service is not None else "error_no_model"
    return {"status": status, "model_loaded": model_service is not None}

@app.post("/predict", response_model=List[PredictionResponse])
def predict_price(apartments: List[ApartmentFeatures]):
    """
    Predicts prices for a list of apartments.
    """
    if model_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please run training (main.py) first to generate 'models/best_model.pkl'."
        )
    
    input_data = [apt.model_dump() for apt in apartments]
    
    try:
        prices = model_service.predict(input_data)
        return [{"predicted_price": round(p, 2)} for p in prices]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    