import torch.nn as nn
import torch

class ShapeCNN(nn.Module):
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

        #self.class_head = nn.Linear(128, num_classes)
        #self.box_head = nn.Linear(128, 3)


    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)

        #x = x.permutate(0, 2, 3, 1)
        class_logits = x[:, :self.num_classes]
        objectness = torch.sigmoid(x[:, self.num_classes])
        bbox = torch.sigmoid(x[:, self.num_classes+1:])

        return class_logits, objectness, bbox

