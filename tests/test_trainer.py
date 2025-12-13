"""
Unit and integration tests for the Trainer module.
"""
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from src.trainer import Trainer, TrainingResult

CENTER_COORDS = (55.75, 37.61)

def test_trainer_initialization(sample_df):
    """Test that Trainer initializes with correct state."""
    trainer = Trainer(sample_df, "price", CENTER_COORDS)
    
    assert trainer.df.equals(sample_df)
    assert trainer.target_col == "price"
    assert trainer.center_coords == CENTER_COORDS
    assert trainer.results == []

def test_create_pipeline(sample_df):
    """Test that _create_pipeline returns a valid sklearn Pipeline."""
    trainer = Trainer(sample_df, "price", CENTER_COORDS)
    
    pipeline = trainer._create_pipeline("linear_regression")
    
    assert isinstance(pipeline, Pipeline)
    step_names = [step[0] for step in pipeline.steps]
    assert "cleaner" in step_names
    assert "geo_features" in step_names
    assert "preprocessor" in step_names
    assert "model" in step_names

def test_run_comparison_integration(sample_df):
    """
    Integration test: Ensure run_comparison executes without error
    and returns a valid DataFrame.
    """
    trainer = Trainer(sample_df, "price", CENTER_COORDS)
    
    models_to_test = ["linear_regression"]
    
    results_df = trainer.run_comparison(models_to_test, test_size=0.5)
    
    assert "r2_score" in results_df.columns
    assert "mae" in results_df.columns
    assert len(trainer.results) == 1
    assert trainer.results[0].model_name == "linear_regression"

def test_save_best_model(tmp_path, sample_df):
    """Test saving the best model to disk."""
    trainer = Trainer(sample_df, "price", CENTER_COORDS)
    
    dummy_model = LinearRegression()
    dummy_result = TrainingResult(
        model_name="dummy", 
        r2_score=0.95, 
        mae=100.0, 
        pipeline=dummy_model
    )
    trainer.results = [dummy_result]
    
    trainer.save_best_model(tmp_path)
    
    assert (tmp_path / "best_model.pkl").exists()

def test_plot_metrics(tmp_path, sample_df):
    """Test that plot generation creates a file."""
    trainer = Trainer(sample_df, "price", CENTER_COORDS)
    
    dummy_result = TrainingResult("dummy", 0.5, 100, None)
    trainer.results = [dummy_result]
    
    trainer.plot_metrics(tmp_path)
    
    assert (tmp_path / "model_comparison.png").exists()
