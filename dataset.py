import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
import shapes

class ShapeDataset(Dataset):
    def __init__(self, classes, size):
        self.x, self.y = 512, 512
        self.dataset = []
        self.labels = []
        self.classes = classes
        self.size = size
        self.build()

    def build(self):
        distribution = {}
        for class_ in self.classes:
            distribution[class_] = self.size // len(self.classes)
        choices = self.classes
        while choices:
            next_shape = np.random.choice(choices)
            if next_shape == 0:
                distribution[next_shape] -= 1
                if distribution[next_shape] == 0:
                    choices.remove(next_shape)
                shape = shapes.Rectangle(*np.random.random_integers(10, 400, 2))
            elif next_shape == 1:
                distribution[next_shape] -= 1
                shape = shapes.Circle(np.random.randint(10, 200))
                if distribution[next_shape] == 0:
                    choices.remove(next_shape)
            else:
                distribution[next_shape] -= 1
                shape, label = shapes.createRandomPolygon(next_shape + 1)
                if distribution[next_shape] == 0:
                    choices.remove(next_shape)

            arr = np.zeros((self.y, self.x))
            shapes.addShape(arr, shape, np.random.randint(0, 300, 2))

            self.labels.append(next_shape)
            self.dataset.append(deepcopy(arr))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.dataset[index]
        label = self.labels[index]

        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)

        return img, label
