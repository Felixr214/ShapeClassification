import torch.nn as nn

class ShapeCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

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
            nn.MaxPool2d(2)
            #7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

