"""
Inference Module.

Handles loading trained models and performing predictions on new data.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict

class ModelService:
    """
    Service for making predictions using a pre-trained scikit-learn pipeline.
    """

    def __init__(self, model_path: Union[str, Path]):
        """
        Initializes the service by loading the model.

        Args:
            model_path (Union[str, Path]): Path to the .pkl model file.
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        """Internal method to load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        return joblib.load(self.model_path)

    def predict(self, input_data: List[Dict]) -> List[float]:
        """
        Generates predictions for a list of inputs.

        Args:
            input_data (List[Dict]): List of dictionaries containing apartment features.

        Returns:
            List[float]: Predicted prices.
        """
        df = pd.DataFrame(input_data)
        
        try:
            predictions = self.model.predict(df)

            predictions_real = np.expm1(predictions)
            return predictions_real.tolist()
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
