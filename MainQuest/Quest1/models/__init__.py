# 1. Import the primary class from the specific file
from .gpt_model import GPTModel

# 2. Define __all__ to support 'from models import *'
# This ensures that only the intended class is exposed, 
# keeping the namespace clean.
__all__ = ['GPTModel']