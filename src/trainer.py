"""
Module for managing the model training and evaluation process.
"""

from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from .models import ModelFactory
from .preprocessor import DataCleaner, GeoDistanceTransformer


@dataclass
class TrainingResult:
    """
    Data class for storing training results of a single model.

    Attributes:
        model_name (str): Name of the model.
        r2_score (float): Coefficient of determination.
        mae (float): Mean Absolute Error.
    """
    model_name: str
    r2_score: float
    mae: float
    pipeline: Pipeline = None


class Trainer:
    """Orchestrator class for training and comparing models."""

    def __init__(self, df: pd.DataFrame, target_col: str, center_coords: tuple):
        """
        Initializes the Trainer.

        Args:
            df (pd.DataFrame): Source dataset.
            target_col (str): Name of the target column.
            center_coords (tuple): Coordinates (lat, lon) for feature generation.
        """
        self.df = df
        self.target_col = target_col
        self.center_coords = center_coords
        self.results: List[TrainingResult] = []

    def _create_pipeline(self, model_name: str) -> Pipeline:
        """
        Assembles the full processing and modeling pipeline.

        Args:
            model_name (str): Name of the model from ModelFactory.

        Returns:
            Pipeline: A scikit-learn Pipeline ready for training.
        """
        feature_eng_steps = [
            ('cleaner', DataCleaner()),
            ('geo_features', GeoDistanceTransformer(*self.center_coords))
        ]

        numeric_features = ['area', 'kitchen_area', 'distance_to_center']
        if 'rooms' in self.df.columns:
            numeric_features.append('rooms')
            
        categorical_features = ['building_type', 'id_region']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop' 
        )

        model = ModelFactory.get_strategy(model_name)

        return Pipeline(steps=[
            *feature_eng_steps,
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    def run_comparison(self, model_names: List[str], test_size: float = 0.2) -> pd.DataFrame:
        """
        Runs training for a list of models and returns a comparison.

        Args:
            model_names (List[str]): List of model names to train.
            test_size (float): Fraction of the dataset to use for testing.

        Returns:
            pd.DataFrame: Table with metric results.
        """
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        for name in model_names:
            print(f"Training model: {name}...")
            pipeline = self._create_pipeline(name)
            
            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                
                res = TrainingResult(
                    model_name=name,
                    r2_score=r2_score(y_test, preds),
                    mae=mean_absolute_error(y_test, preds)
                )
                self.results.append(res)
            except Exception as e:
                print(f"Error training {name}: {e}")

        return pd.DataFrame([vars(r) for r in self.results])

    def save_best_model(self, output_dir: Path) -> None:
        """
        Saves the model with the highest R2 Score to disk.
        
        Args:
            output_dir (Path): Directory to save the model.
        """
        if not self.results:
            print("No models trained to save.")
            return
            
        best_res = max(self.results, key=lambda x: x.r2_score)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "best_model.pkl"
        
        joblib.dump(best_res.pipeline, save_path)
        print(f"Best model ({best_res.model_name}) saved to {save_path}")
    
    def plot_metrics(self, output_dir: Path) -> None:
        """
        Generate and save plot of comparision models side-by-side.

        Args:
            output_dir (Path): Path to plot saved folder.
        """
        if not self.results:
            print("No results to plot.")
            return

        df_res = pd.DataFrame([asdict(r) for r in self.results])
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "model_comparison.png" 

        sns.set_theme(style="whitegrid")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        sns.barplot(
            data=df_res, x='model_name', y='r2_score', 
            ax=ax1, palette="viridis", alpha=0.8
        )
        ax1.set_title('Сравнение метрики R2 Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R2 Score', fontweight='bold')
        ax1.set_xlabel('Название модели')
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45) 

        ax2 = axes[1]
        sns.barplot(
            data=df_res, x='model_name', y='mae', 
            ax=ax2, palette="magma", alpha=0.8
        )
        ax2.set_title('Сравнение метрики MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error (MAE)', fontweight='bold')
        ax2.set_xlabel('Название модели')
        ax2.tick_params(axis='x', rotation=45) 
        
        fig.suptitle('Сравнение моделей: R2 Score vs MAE', fontsize=16, y=1.05) 

        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Comparison plot saved to {save_path}")
