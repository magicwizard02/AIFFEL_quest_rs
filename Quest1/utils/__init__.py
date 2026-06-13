"""
Utility Package for Deep Learning Model Training and Explainability Analysis.
Exposes modular functions for training, weight management, and XAI visualization.
"""

from .trainer import train_one_epoch, validate
from .saver import save_weights, load_weights, update_results_refined
from .dataset import ImageFolderWithXMLBBox
from .data_utils import get_dog_dataloader
from .visualizer import (
    unnormalize, 
    calculate_iou,
    calculate_iou_at_threshold,
    save_individual_heatmap, 
    save_multi_layer_results
)

__all__ = [
    'train_one_epoch', 'validate',
    'save_weights', 'load_weights', 'update_results_refined',
    'ImageFolderWithXMLBBox',
    'get_dog_dataloader',
    'unnormalize', 'calculate_iou', 'calculate_iou_at_threshold', 'save_individual_heatmap', 'save_multi_layer_results'
]