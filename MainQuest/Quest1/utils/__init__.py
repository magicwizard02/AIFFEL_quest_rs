# ===========================================================================
# utils/__init__.py
# Centralized interface for GPT Pretraining and SFT Utilities
# ===========================================================================

# 1. Data Processing & Analysis
from .analyze import analyze_corpus, find_threshold_by_coverage
from .data_utils import pad_sequence, to_tensor

# 2. Training Logic & Masking
from .trainer import train_one_epoch, validate
from .masking import create_look_ahead_mask

# 3. Experiment Tracking & Weights
from .trainer_utils import save_weights, update_results, load_weights, get_lr_lambda

# 4. Inference
from .inference import greedy_decode_gpt

# Define __all__ to control 'from utils import *' behavior
__all__ = [
    'analyze_corpus',
    'find_threshold_by_coverage',
    'pad_sequence',
    'to_tensor',
    'train_one_epoch',
    'validate',
    'create_look_ahead_mask',
    'save_weights',
    'update_results',
    'load_weights',
    'get_lr_lambda',
    'greedy_decode_gpt'
]