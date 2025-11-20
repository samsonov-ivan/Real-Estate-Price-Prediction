import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

class ModelTrainer:
    """
    A class to handle model training, evaluation, and selection for regression tasks.
    """

    def __init__(self):
        self.models = {}
        self.results = {}
    
    def build_pipeline(self, model, cat_features, num_features):
        """
        Build a machine learning pipeline with preprocessing and the given model.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ]
        )
        
        if model == 'catboost':
            cat_model = CatBoostRegressor(verbose=0)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', cat_model)
            ])
        elif model == 'linear_regression':
            lin_model = LinearRegression()
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', lin_model)
            ])
        elif model == 'random_forest':
            rf_model = RandomForestRegressor()
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', rf_model)
            ])
        else:
            raise ValueError("Unsupported model type.", model)

        return pipeline
    
    
    def train_and_evaluate(self, df, target_column, test_size=0.2, random_state=42):
        """
        Train and evaluate models on the provided DataFrame.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_features = X.select_dtypes(include=['number']).columns.tolist()
        
        for model_name in ['linear_regression', 'catboost', 'random_forest']:
            pipeline = self.build_pipeline(model_name, cat_features, num_features)
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.models[model_name] = pipeline
            self.results[model_name] = {'r2_score': r2, 'mae': mae}
        
        return self.results


    def get_results(self):
        """
        Get the evaluation results of the trained models.
        """
        return pd.DataFrame(self.results)
