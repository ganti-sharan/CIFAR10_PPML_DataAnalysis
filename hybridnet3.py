import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNormResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=8):
        super(GroupNormResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class GroupNormDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, num_groups=8):
        super(GroupNormDenseBlock, self).__init__()
        self.layer_list = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate, num_groups)
            self.layer_list.append(layer)

    def _make_layer(self, in_channels, growth_rate, num_groups):
        layer = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        return layer

    def forward(self, x):
        for layer in self.layer_list:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x

class MobileNetBlockWithGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_groups=8):
        super(MobileNetBlockWithGroupNorm, self).__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.gn_dw = nn.GroupNorm(num_groups, in_channels)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn_pw = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = F.relu(self.gn_dw(self.conv_dw(x)))
        x = F.relu(self.gn_pw(self.conv_pw(x)))
        return x

class HybridNet(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridNet, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn_initial = nn.GroupNorm(8, 64)
        self.res_block = GroupNormResidualBlock(64, 128, stride=2)
       # First original DenseBlock
        self.dense_block_1 = GroupNormDenseBlock(128, 32, 4)  # Output channels = 128 + 32*4 = 256
        self.dense_block_2 = GroupNormDenseBlock(256, 32, 4)  # Output channels = 256 + 32*4 = 384
        self.dense_block_3 = GroupNormDenseBlock(384, 32, 4)  # Output channels = 384 + 32*4 = 512
        self.dense_block_4 = GroupNormDenseBlock(512, 32, 4)  # Output channels = 512 + 32*4 = 640 (Adjust if needed)
        self.mobile_block = MobileNetBlockWithGroupNorm(640, 512, stride=2)  # Adjusted output channels from DenseBlock4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # Adjusted input features to match last DenseBlock output


    def forward(self, x):
        x = F.relu(self.gn_initial(self.initial_conv(x)))
        x = self.res_block(x)
        # Forward pass through additional DenseBlocks
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = self.dense_block_3(x)
        x = self.dense_block_4(x)
        x = self.mobile_block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
