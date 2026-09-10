"""Utility functions for logging, reproducibility, and file I/O."""
import json
import logging
import random
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Union

import numpy as np


def setup_logger(name: str = "apple_support_agent", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def seed_everything(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def timed_execution(func: Callable) -> Callable:
    """Decorator to measure and log execution time of functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"Executed `{func.__name__}` in {elapsed:.2f}s")
        return result

    return wrapper


def save_json(data: Union[Dict[str, Any], list], file_path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary or list as formatted JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved JSON to {path}")


def load_json(file_path: Union[str, Path]) -> Any:
    """Load JSON from file path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
