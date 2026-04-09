# models/__init__.py

"""
ResNet Model Package
This file allows for cleaner imports by exposing the main builder 
and blocks directly at the package level.
"""

# Import the main builder function from resnet_builder.py
from .resnet_builder import build_resnet

# Import the individual building blocks from blocks.py
# (Useful for verifying architectures or manual customization)
from .blocks import BasicBlock, BottleneckBlock

# Defines the public API for the models package
# Enables: from models import build_resnet, BasicBlock, BottleneckBlock
__all__ = [
    'build_resnet', 
    'BasicBlock', 
    'BottleneckBlock'
]