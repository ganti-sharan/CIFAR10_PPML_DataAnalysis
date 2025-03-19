import torch
import torch.nn as nn
import torch.nn.functional as F

# Adjust SEAttention block output and linear layer input size to match
class SEAttention(nn.Module):
    def __init__(self, in_channels,num_classes, reduction_ratio=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, num_classes),  # Adjust to match num_classes
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y)
        return x * y.view(batch_size, channels, 1, 1)  # Ensure output size matches input size


# Define Residual Block (ResNet component) with GroupNorm
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels)  # Adjust num_groups as needed
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels)  # Adjust num_groups as needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)  # Adjust num_groups as needed
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

# Define Dense Block (DenseNet component) with GroupNorm
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layer_list = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate)
            self.layer_list.append(layer)

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.GroupNorm(8, in_channels),  # Adjust num_groups as needed
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        return layer

    def forward(self, x):
        for layer in self.layer_list:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x

# Define EfficientNet Block (EfficientNet component) with GroupNorm
class EfficientNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(EfficientNetBlock, self).__init__()
        # Define layers of EfficientNet block here
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels)  # Adjust num_groups as needed
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels)  # Adjust num_groups as needed

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return out

# Define HybridNet combining all components with GroupNorm
class HybridNet(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridNet, self).__init__()
        # Define the backbone using a combination of EfficientNet, ResNet, DenseNet blocks and SE Attention
        self.backbone = nn.Sequential(
            EfficientNetBlock(3, 64),
            ResidualBlock(64, 128),
            DenseBlock(128, 32, 4),
            SEAttention(160, 10),  # Applying SE Attention after Dense Block
            EfficientNetBlock(160, 128),
            ResidualBlock(128, 256),
            DenseBlock(256, 64, 4),
            SEAttention(320, 10),  # Applying SE Attention after Dense Block
            EfficientNetBlock(320, 256),
            ResidualBlock(256, 512),
            DenseBlock(512, 128, 4),
            SEAttention(640, 10),  # Applying SE Attention after Dense Block
        )
        # Add global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes) 

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate and use the HybridNet with GroupNorm
model = HybridNet(num_classes=10)
