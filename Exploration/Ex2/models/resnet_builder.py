import torch
import torch.nn as nn
from .blocks import BasicBlock, BottleneckBlock, conv_bn_relu

# =========================================================
# How to switch models using build_resnet function:
# =========================================================
# ResNet-18:
#    build_resnet(num_blocks_list=[2, 2, 2, 2], use_bottleneck=False)
#
# ResNet-34:
#    build_resnet(num_blocks_list=[3, 4, 6, 3], use_bottleneck=False)
#
# ResNet-50:
#    build_resnet(num_blocks_list=[3, 4, 6, 3], use_bottleneck=True)
#
# ResNet-101:
#    build_resnet(num_blocks_list=[3, 4, 23, 3], use_bottleneck=True)
#
# ResNet-152:
#    build_resnet(num_blocks_list=[3, 8, 36, 3], use_bottleneck=True)
#
# For Ablation Studies (Plain models), set: is_plain=True
# =========================================================

class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks_list, num_classes=10, is_plain=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.is_plain = is_plain

        # ----- Stage 1: The Stem (Entry Flow) -----
        # Aggressive downsampling using 7x7 conv and 3x3 MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ----- Stage 2-5: Residual/Plain Body -----
        # Each stage consists of multiple blocks (e.g., [3, 4, 6, 3])
        self.stage2 = self._make_layer(block_type, 64,  num_blocks_list[0], stride=1)
        self.stage3 = self._make_layer(block_type, 128, num_blocks_list[1], stride=2)
        self.stage4 = self._make_layer(block_type, 256, num_blocks_list[2], stride=2)
        self.stage5 = self._make_layer(block_type, 512, num_blocks_list[3], stride=2)

        # ----- Global Pooling & Final Classifier -----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Reduces spatial (H, W) to 1x1
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)

    def _make_layer(self, block_type, base_channels, num_blocks, stride):
        """
        Builds a stage by stacking a specific number of blocks.
        Only the first block in the stage performs downsampling via 'stride'.
        """
        layers = []
        
        # ----- First Block of the Stage -----
        # Handles stride for downsampling and channel expansion (1x -> 4x for Bottleneck)
        layers.append(block_type(self.in_channels, base_channels, stride, self.is_plain))
        
        # Update current input channels to match the expanded block output
        self.in_channels = base_channels * block_type.expansion

        # ----- Remaining Blocks in the Stage -----
        # Stride remains 1 to maintain feature map resolution within the stage
        for _ in range(1, num_blocks):
            layers.append(block_type(self.in_channels, base_channels, 1, self.is_plain))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. Stem processing
        x = self.conv1(x)
        
        # 2. Forward through sequential residual stages
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        # 3. Final pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flattens tensor for the linear layer
        x = self.fc(x)
        return x

def build_resnet(num_classes=10, num_blocks_list=[3, 4, 6, 3], use_bottleneck=False, is_plain=False):
    """
    Factory function to generate custom ResNet or Plain-Net architectures.
    Setting is_plain=True facilitates Ablation Studies on residual learning.
    """
    block_type = BottleneckBlock if use_bottleneck else BasicBlock
    return ResNet(block_type, num_blocks_list, num_classes, is_plain)