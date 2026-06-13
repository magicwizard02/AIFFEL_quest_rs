# ===========================================================================
# models/__init__.py
# Package-level exports
# ===========================================================================

from .transformer import Transformer
from .layers import get_lr_lambda

# Explicitly define what is available when using 'from models import *'
__all__ = ['Transformer', 'get_lr_lambda']