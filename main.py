import pandas as pd
import numpy as np
import kagglehub
from src.preprocessor import CustomPreprocessor
from src.trainer import ModelTrainer


def load_data(file_path = None):
    """
    Load dataset from a CSV file.
    """

    if file_path is None:
        raise ValueError("No file path provided for dataset.")
    
    return pd.read_csv(file_path)


def main():
    data = load_data("data/raw.csv")
    print(data.head(5))
    
    target_column = 'price'
    
    preprocessor = CustomPreprocessor()

    rename_dict = {'geo_lat': 'latitude', 'geo_lon': 'longitude'}
    data = data.rename(columns=rename_dict)

    data = preprocessor.clean_data(data)
    preprocessor.fit(data)
    processed_data = preprocessor.transform(data)
    print("Data preprocessing completed.")
    print("Size of processed data:", processed_data.shape)
    print("Columns in processed data:", processed_data.columns.tolist())
    print(processed_data.head())

    trainer = ModelTrainer()

    
    results = trainer.train_and_evaluate(processed_data, target_column)

    print("Model training and evaluation completed.")
    print("Results:" , results)

if __name__ == "__main__":
    main()
