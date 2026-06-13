# models/__init__.py

from .resnet50 import ResNet50_Standard, ResNet50_GAP
from .vgg16 import VGG16_Standard, VGG16_GAP
from .alexnet import AlexNet_Standard, AlexNet_GAP

__all__ = [
    'ResNet50_Standard', 'ResNet50_GAP', 
    'VGG16_Standard', 'VGG16_GAP', 
    'AlexNet_Standard', 'AlexNet_GAP', 
    'get_model'
]

def get_model(model_name, num_classes=120):
    """
    Factory function to retrieve models by name.
    """
    map_dict = {
        "vgg16_gap": VGG16_GAP,
        "resnet50": ResNet50_GAP,  
        "alexnet_gap": AlexNet_GAP
    }
    
    if model_name not in map_dict:
        raise KeyError(f"Model name '{model_name}' not found. Available: {list(map_dict.keys())}")
        
    return map_dict[model_name](num_classes=num_classes)