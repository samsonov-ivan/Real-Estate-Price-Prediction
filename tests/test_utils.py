"""
Unit tests for the utils module.
"""
import logging
import pytest
from pathlib import Path
from src.utils import setup_logging, save_object, load_object

def test_setup_logging():
    """Test that the logger is configured correctly."""
    logger = setup_logging()
    
    assert isinstance(logger, logging.Logger)
    
    assert len(logger.handlers) > 0
    
    assert logger.name == "RealEstateProject"

def test_save_and_load_object(tmp_path):
    """
    Test saving an object to a file and reloading it.
    """
    test_data = {"model": "test", "score": 0.99, "params": [1, 2, 3]}
    file_path = tmp_path / "test_model.pkl"
    
    save_object(test_data, file_path)
    assert file_path.exists()
    
    loaded_data = load_object(file_path)
    assert loaded_data == test_data
    assert loaded_data["score"] == 0.99

def test_load_object_not_found():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_object(Path("non_existent_ghost_file.pkl"))