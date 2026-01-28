"""
config.py - Shared Configuration and Constants
===============================================
Central configuration for the quant_research library.
"""

import platform
from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Data directories (absolute paths)
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"

# Random seed for reproducibility
SEED = 42

# Platform detection
IS_WINDOWS = platform.system() == 'Windows'
DEFAULT_PARALLEL = not IS_WINDOWS  # Parallel disabled by default on Windows

# Training defaults
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_LBFGS_LR = 1.0
DEFAULT_EPOCHS = 6000
DEFAULT_TEST_SIZE = 0.25

# Time constants
TRADING_DAYS_PER_YEAR = 365
TRADING_HOURS_PER_DAY = 24

# Training logging
LOG_INTERVAL_DIVISOR = 10  # Log every n_epochs // 10
