import torch
import torch.nn as nn

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, apply_relu=True):
    """
    Standard Helper: Convolution -> Batch Normalization -> (Optional) ReLU.
    - bias=False: Since BN includes learnable affine parameters, the bias in 
      the preceding Conv layer is mathematically redundant.
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if apply_relu:
        layers.append(nn.ReLU(inplace=True)) # Inplace saves memory during backprop
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    """
    Basic Block used in ResNet-18 and ResNet-34.
    Consists of two 3x3 convolutions with an optional identity shortcut.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, is_plain=False):
        super(BasicBlock, self).__init__()
        self.is_plain = is_plain
        
        # ----- First 3x3 Convolution -----
        # Responsible for spatial downsampling when stride > 1
        self.conv1 = conv_bn_relu(in_channels, out_channels, stride=stride)
        
        # ----- Second 3x3 Convolution -----
        # ReLU is omitted here because addition must occur before the final activation
        self.conv2 = conv_bn_relu(out_channels, out_channels, apply_relu=False)
        
        # ----- Shortcut Connection (Skip Connection) -----
        self.shortcut = nn.Sequential()
        if not is_plain:
            # If resolution drops or channels increase, project identity to match dimensions
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # ----- Save original input for skip connection -----
        identity = x

        # ----- Forward pass through convolutional path -----
        out = self.conv1(x)   # Conv -> BN -> ReLU
        out = self.conv2(out) # Conv -> BN

        # ----- Element-wise Addition: Core Idea of ResNet -----
        if not self.is_plain:
            # Learning the residual mapping: F(x) + x
            out += self.shortcut(identity)
            
        # Final non-linearity applied after the signal merge
        return self.relu(out)

class BottleneckBlock(nn.Module):
    """
    Bottleneck Block used in ResNet-50, 101, and 152.
    Uses 1x1 convolutions to reduce/expand channels for computational efficiency.
    """
    expansion = 4

    def __init__(self, in_channels, base_channels, stride=1, is_plain=False):
        super(BottleneckBlock, self).__init__()
        self.is_plain = is_plain
        out_channels = base_channels * self.expansion
        
        # ----- 1x1 Reduction Layer -----
        # Squashes channel depth to 'base_channels' to limit 3x3 conv workload
        self.conv1 = conv_bn_relu(in_channels, base_channels, kernel_size=1, stride=stride, padding=0)
        
        # ----- 3x3 Processing Layer -----
        # Bottleneck bottlenecking: performing spatial convolution in reduced space
        self.conv2 = conv_bn_relu(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        # ----- 1x1 Expansion Layer -----
        # Restores depth to 4x the base channel count (no ReLU)
        self.conv3 = conv_bn_relu(base_channels, out_channels, kernel_size=1, padding=0, apply_relu=False)
        
        # ----- Shortcut Connection Logic -----
        self.shortcut = nn.Sequential()
        if not is_plain:
            # Projection shortcut to match the 4x expanded output
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # ----- Save original input for residual learning -----
        identity = x

        # ----- Forward pass through bottleneck path -----
        out = self.conv1(x)   # 1x1 Reduction
        out = self.conv2(out) # 3x3 Conv
        out = self.conv3(out) # 1x1 Expansion (No ReLU)

        # ----- Residual Connection -----
        if not self.is_plain:
            # Combines learned features with original identity
            out += self.shortcut(identity)
            
        return self.relu(out)