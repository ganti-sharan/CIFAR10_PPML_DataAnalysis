# import torch
# import torch.nn as nn

# class HybridBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate):
#         super(HybridBlock, self).__init__()
#         # Using a bottleneck layer from DenseNet for feature reduction
#         inter_channels = growth_rate * 4
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

#         # Applying MobileNet's depth-wise separable convolution for efficient computation
#         # Ensure the number of output channels matches the number of groups for the depth-wise convolution
#         self.bn2 = nn.BatchNorm2d(inter_channels)
#         self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, groups=inter_channels, bias=False)
#         self.conv3 = nn.Conv2d(inter_channels, growth_rate, kernel_size=1, bias=False)

#         # ResNet-like skip connection if applicable
#         self.use_res_connect = in_channels == growth_rate

#     def forward(self, x):
#         out = self.conv1(nn.ReLU()(self.bn1(x)))
#         out = self.conv3(nn.ReLU()(self.bn2(self.conv2(out))))
#         if self.use_res_connect:
#             out = x + out
#         return out


# class HybridNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(HybridNet, self).__init__()
#         # Initial convolution layer similar to what is seen in DenseNet and ResNet
#         self.init_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

#         # Stacked Hybrid Blocks
#         self.block1 = HybridBlock(64, 32)
#         self.block2 = HybridBlock(32, 32)
#         self.block3 = HybridBlock(32, 32)

#         # Final Batch Normalization and Linear classifier
#         self.bn_final = nn.BatchNorm2d(32)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(32, num_classes)

#     def forward(self, x):
#         x = self.init_conv(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.bn_final(x)
#         x = self.adaptive_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


import torch
import torch.nn as nn

class HybridBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(HybridBlock, self).__init__()
        # Using a bottleneck layer from DenseNet for feature reduction
        inter_channels = growth_rate * 4
        self.gn1 = nn.GroupNorm(8, in_channels)  # Group number is chosen arbitrarily; adjust as needed
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        # Applying MobileNet's depth-wise separable convolution for efficient computation
        self.gn2 = nn.GroupNorm(8, inter_channels)  # Adjust the number of groups according to your needs
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, groups=inter_channels, bias=False)
        self.conv3 = nn.Conv2d(inter_channels, growth_rate, kernel_size=1, bias=False)

        # ResNet-like skip connection if applicable
        self.use_res_connect = in_channels == growth_rate

    def forward(self, x):
        out = self.conv1(nn.ReLU()(self.gn1(x)))
        out = self.conv3(nn.ReLU()(self.gn2(self.conv2(out))))
        if self.use_res_connect:
            out = x + out
        return out


class HybridNet(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridNet, self).__init__()
        # Initial convolution layer similar to what is seen in DenseNet and ResNet
        self.init_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Stacked Hybrid Blocks
        self.block1 = HybridBlock(64, 32)
        self.block2 = HybridBlock(32, 32)
        self.block3 = HybridBlock(32, 32)

        # Final Group Normalization and Linear classifier
        self.gn_final = nn.GroupNorm(8, 32)  # Adjust the number of groups according to your needs
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gn_final(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
