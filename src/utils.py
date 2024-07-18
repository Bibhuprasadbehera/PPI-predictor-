# src/utils.py
import os
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def seed_everything(seed=42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
