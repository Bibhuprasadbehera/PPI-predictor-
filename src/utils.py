import os
import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def ensure_dir(directory):
    """Ensures that the specified directory exists. Creates it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def seed_everything(seed=42):
    """Sets random seeds for Python, NumPy, and PyTorch for reproducibility."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logger(log_file, log_level=logging.INFO):
    """Sets up a logger that writes to both console and a log file."""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Create file handler and set level
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add formatter to handlers
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def calculate_metrics(y_true, y_pred):
    """Calculates various regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr
    }
