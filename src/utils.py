"""
Utility module for logging, file operations, and model persistence.
"""

import logging
import sys
import pickle
from pathlib import Path
from typing import Any

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configures a professional logger with a standard format.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("RealEstateProject")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def save_object(obj: Any, filepath: Path) -> None:
    """
    Saves a Python object to a file using pickle.

    Args:
        obj (Any): Object to save.
        filepath (Path): Destination path.
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        logging.getLogger("RealEstateProject").info(f"Object saved to {filepath}")
    except Exception as e:
        logging.getLogger("RealEstateProject").error(f"Failed to save object: {e}")
        raise

def load_object(filepath: Path) -> Any:
    """
    Loads a Python object from a pickle file.

    Args:
        filepath (Path): Path to the file.

    Returns:
        Any: Loaded object.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logging.getLogger("RealEstateProject").info(f"Object loaded from {filepath}")
    return obj
