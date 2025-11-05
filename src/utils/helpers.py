"""Helper utility functions."""

import os
import random
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
    logger.info(f"Random seed set to {seed}")


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save to
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
        
    logger.info(f"Object saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to load from
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
        
    logger.info(f"Object loaded from {filepath}")
    
    return obj


def timer(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        elapsed = end_time - start_time
        logger.info(f"{func.__name__} took {elapsed:.2f} seconds")
        
        return result
        
    return wrapper


def create_submission_file(
    predictions: np.ndarray,
    ids: np.ndarray,
    filepath: str,
    competition_format: str = 'kaggle'
) -> None:
    """
    Create submission file for competition.
    
    Args:
        predictions: Model predictions
        ids: Sample IDs
        filepath: Output file path
        competition_format: Format type
    """
    import pandas as pd
    
    if competition_format == 'kaggle':
        submission = pd.DataFrame({
            'Id': ids,
            'Predicted': predictions
        })
    else:
        submission = pd.DataFrame({
            'id': ids,
            'prediction': predictions
        })
        
    submission.to_csv(filepath, index=False)
    logger.info(f"Submission file saved to {filepath}")


def print_system_info() -> None:
    """Print system and environment information."""
    import platform
    import sys
    
    logger.info("="*50)
    logger.info("System Information")
    logger.info("="*50)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info("="*50)


def memory_usage() -> None:
    """Print current memory usage."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    logger.info(f"Current memory usage: {memory_mb:.2f} MB")    