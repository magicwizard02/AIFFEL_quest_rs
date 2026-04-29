import torch.nn as nn
from torchvision import models
import torch
class ResNet50_Standard(nn.Module):
    def __init__(self, num_classes=120):
        super(ResNet50_Standard, self).__init__()
        # Load pre-trained ResNet50 from torchvision
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        
        # Modify the final fully connected layer for custom classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Forward pass through the entire model
        return self.model(x)

# ResNet50 with GAP for CAM (ResNet already has GAP internally, but we structure it similarly)
class ResNet50_GAP(nn.Module):
    def __init__(self, num_classes=120):
        super(ResNet50_GAP, self).__init__()
        
        # 1. Load the pre-trained ResNet50 model with ImageNet weights
        # Using 'IMAGENET1K_V1' ensures we start with a model that already understands visual features.
        resnet_base = models.resnet50(weights='IMAGENET1K_V1')
        
        # 2. Extract the feature extractor (Backbone)
        # We exclude the last two layers: the default Global Average Pooling (GAP) and the Fully Connected (FC) layer.
        # This leaves us with the final convolutional block, which outputs the 7x7 (for 224 input) feature maps.
        self.features = nn.Sequential(*list(resnet_base.children())[:-2])
        
        # 3. Explicit Global Average Pooling (GAP)
        # This layer reduces each [C, H, W] feature map to a [C, 1, 1] vector by averaging all pixel values.
        # This is essential for CAM as it maintains a direct relationship between the feature maps and the weights.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Custom Classifier (Linear Layer)
        # ResNet50's final conv block outputs 2048 channels.
        # We map these 2048 features to the specific number of classes in your dataset (e.g., 120).
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Step A: Extract spatial feature maps from the backbone
        # Shape: [Batch, 2048, H', W']
        x = self.features(x)
        
        # Step B: Apply Global Average Pooling
        # Shape: [Batch, 2048, 1, 1]
        x = self.gap(x)
        
        # Step C: Flatten the tensor for the linear layer
        # Shape: [Batch, 2048]
        x = torch.flatten(x, 1)
        
        # Step D: Final classification
        # Shape: [Batch, num_classes]
        x = self.classifier(x)
        
        return x