"""
Main entry point for the Real Estate Price Prediction project.

This script orchestrates the entire workflow:
1. Data Loading.
2. pipeline Construction (via Trainer).
3. Model Training & Evaluation.
4. Reporting.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.trainer import Trainer
from src.utils import setup_logging

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_FILE = "raw_sample.csv"
TARGET_COLUMN = "price"

MOSCOW_CENTER = (55.7558, 37.6173)

MODELS_TO_RUN = ["linear_regression", "catboost", "random_forest"]


def load_data(file_name: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.

    Args:
        file_name (str): Name of the file in the data directory.

    Returns:
        pd.DataFrame: Loaded data.
    """
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)


def main():
    """
    Main execution function.
    """
    logger = setup_logging()
    logger.info("Starting Real Estate Price Prediction pipeline...")

    try:
        data = load_data(RAW_DATA_FILE)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # data = data.sample(n=50000, random_state=42)
    data['price'] = np.log1p(data['price'])
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
    print(data.head(5))

    rename_dict = {'geo_lat': 'latitude', 'geo_lon': 'longitude'}
    data = data.rename(columns=rename_dict)
    logger.info("Column renaming completed.")

    trainer = Trainer(
        df=data,
        target_col=TARGET_COLUMN,
        center_coords=MOSCOW_CENTER
    )

    logger.info(f"Starting comparison for models: {MODELS_TO_RUN}")
    
    results = trainer.run_comparison(
        model_names=MODELS_TO_RUN,
        test_size=0.2
    )

    logger.info("Model training and evaluation completed.")
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(results)
    print("="*50 + "\n")

    report_path = REPORTS_DIR / "final_report.csv"
    
    results.to_csv(report_path, index=False)
    logger.info(f"Results saved to {report_path}")
    
    trainer.plot_metrics(output_dir=REPORTS_DIR)

    if not results.empty:
        best_model = results.sort_values(by="r2_score", ascending=False).iloc[0]
        logger.info(f"Best model based on R2 Score: {best_model['model_name']} (R2: {best_model['r2_score']:.4f})")


if __name__ == "__main__":
    main()
