import torch.nn as nn
from torchvision import models

class VGG16_Standard(nn.Module):
    def __init__(self, num_classes=120):
        super(VGG16_Standard, self).__init__()
        # Original VGG16 with large classifier (Flatten -> 4096 -> 4096 -> num_classes)
        self.model = models.vgg16(weights='IMAGENET1K_V1')
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

class VGG16_GAP(nn.Module):
    def __init__(self, num_classes=120):
        super(VGG16_GAP, self).__init__()
        # 1. Load pre-trained VGG16 backbone
        vgg_base = models.vgg16(weights='IMAGENET1K_V1')
        
        # 2. Extract only the convolutional features (512 channels at the end)
        self.features = vgg_base.features
        
        # 3. Global Average Pooling: Redrawing spatial dimensions [7x7] to [1x1]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Dense Layer (Classifier): Maps 512 features to 120 classes
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1) # Flatten [Batch, 512, 1, 1] to [Batch, 512]
        x = self.classifier(x)
        return x