import torch.nn as nn
from torchvision import models
import torch

class AlexNet_Standard(nn.Module):
    def __init__(self, num_classes=120):
        super(AlexNet_Standard, self).__init__()
        self.model = models.alexnet(weights='IMAGENET1K_V1')
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

class AlexNet_GAP(nn.Module):
    def __init__(self, num_classes=120):
        super(AlexNet_GAP, self).__init__()
        # Load pre-trained AlexNet
        base_model = models.alexnet(weights='IMAGENET1K_V1')
        
        # Use only the feature extractor
        self.features = base_model.features
        
        # Replace original classifier with GAP for CAM compatibility
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classifier (input size for AlexNet features is 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x