"""
Module for training and evaluating machine learning models for regression tasks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from dataclasses import dataclass, field
import os


@dataclass
class ModelTrainer:
    """
    A class to handle model training, evaluation, and selection for regression tasks.
    """
    models: Dict[str, Pipeline] = field(default_factory=dict)
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def build_pipeline(self, model: str, cat_features: List[str], num_features: List[str]) -> Pipeline:
        """
        Build a machine learning pipeline with preprocessing and the given model.

        Args:
            model (str): Model type ('catboost', 'linear_regression', 'random_forest').
            cat_features (List[str]): Categorical features.
            num_features (List[str]): Numerical features.

        Returns:
            Pipeline: Built pipeline.
        """
        if model == 'catboost':
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                ],
                remainder='passthrough'
            )
            cat_model = CatBoostRegressor(verbose=0)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', cat_model)
            ])
            cat_indices = list(range(len(num_features), len(num_features) + len(cat_features)))
            pipeline.named_steps['model'].set_params(cat_features=cat_indices)
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ]
            )
            if model == 'linear_regression':
                model_obj = LinearRegression()
            elif model == 'random_forest':
                model_obj = RandomForestRegressor()
            else:
                raise ValueError(f"Unsupported model type: {model}")
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model_obj)
            ])
        return pipeline

    def train_and_evaluate(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Train and evaluate models on the provided DataFrame using cross-validation.

        Args:
            df (pd.DataFrame): Input data.
            target_column (str): Target column name.

        Returns:
            pd.DataFrame: Results DataFrame.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_features = X.select_dtypes(include=['number']).columns.tolist()

        for model_name in ['linear_regression', 'catboost', 'random_forest']:
            pipeline = self.build_pipeline(model_name, cat_features, num_features)

            cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
            cv_neg_mae = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
            cv_neg_mse = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')

            r2_mean = cv_r2.mean()
            r2_std = cv_r2.std()
            mae_mean = -cv_neg_mae.mean()
            mae_std = cv_neg_mae.std()
            rmse_mean = np.sqrt(-cv_neg_mse.mean())
            rmse_std = np.sqrt(cv_neg_mse.std())  # Approximate std for RMSE

            self.results[model_name] = {
                'r2_mean': r2_mean,
                'r2_std': r2_std,
                'mae_mean': mae_mean,
                'mae_std': mae_std,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std
            }

            # Fit on full data
            pipeline.fit(X, y)
            self.models[model_name] = pipeline

        self.plot_results()

        results_df = pd.DataFrame(self.results).T
        results_df[['r2_mean', 'r2_std']] = results_df[['r2_mean', 'r2_std']].round(4)
        results_df[['mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']] = results_df[['mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']].round(0)
        return results_df

    def plot_results(self, save: bool = False, output_dir: str = '.') -> None:
        """
        Plot the evaluation results of the trained models.

        Args:
            save (bool): Whether to save the plot to file.
            output_dir (str): Directory to save the plot.

        Returns:
            None
        """
        model_names = list(self.results.keys())
        r2_scores = [self.results[model]['r2_mean'] for model in model_names]
        r2_stds = [self.results[model]['r2_std'] for model in model_names]
        maes = [self.results[model]['mae_mean'] for model in model_names]
        mae_stds = [self.results[model]['mae_std'] for model in model_names]
        rmses = [self.results[model]['rmse_mean'] for model in model_names]
        rmse_stds = [self.results[model]['rmse_std'] for model in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        fig, ax1 = plt.subplots()
        bars1 = ax1.bar(x - width, r2_scores, width, yerr=r2_stds, capsize=5, label='R2 Score', color='b')
        ax1.set_ylabel('R2 Score')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x, maes, width, yerr=mae_stds, capsize=5, label='MAE', color='r')
        bars3 = ax2.bar(x + width, rmses, width, yerr=rmse_stds, capsize=5, label='RMSE', color='g')
        ax2.set_ylabel('Error')
        ax2.legend(loc='upper right')

        plt.title('Model Evaluation Results')
        if save:
            plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        plt.show()

    def get_results(self) -> pd.DataFrame:
        """
        Get the evaluation results of the trained models.

        Returns:
            pd.DataFrame: Results DataFrame.
        """
        return pd.DataFrame(self.results)
