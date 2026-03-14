import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, padding=k//2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ShapeCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(

            ConvBlock(1, 32, 3, 2),   # 128x128
            ConvBlock(32, 64, 3, 2),  # 64x64
            ResidualBlock(64),

            ConvBlock(64, 128, 3, 2), # 32x32
            ResidualBlock(128),
            ResidualBlock(128),

            ConvBlock(128, 256, 3, 2), # 16x16
            ResidualBlock(256),

            ConvBlock(256, 512, 3, 2), # 8x8
        )

        self.pred = nn.Conv2d(512, num_classes + 4, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pred(x)

        objectness = torch.sigmoid(x[:, 0])
        class_logits = x[:, 1:self.num_classes+1]
        bbox = torch.sigmoid(x[:, self.num_classes+1:])

        return objectness, class_logits, bbox


class old_ShapeCNN(nn.Module):
    def __init__(self, num_classes, grid_size):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #126x126

            nn.Conv2d(16, 64, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #31x31

            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #7x7

            nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )

        self.pred = nn.Conv2d(128, num_classes + 4, 1)


    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)

        objectness = torch.sigmoid(x[:, 0])
        class_logits = x[:, 1:self.num_classes+1]
        bbox = torch.sigmoid(x[:, self.num_classes+1:])

        return objectness, class_logits, bbox

