import numpy as np
import torch
from copy import deepcopy

from matplotlib.mlab import angle_spectrum
from numpy.ma.core import size
from torch.fx.experimental.sym_node import sizes_strides_methods
from torch.utils.data import Dataset
import shapes

class ShapeDataset(Dataset):
    def __init__(self, classes, size, create=True):
        self.x, self.y = 256, 256
        self.dataset = []
        self.labels = []
        self.classes = classes
        self.size = size
        if create:
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
                shape = shapes.Circle(np.random.randint(10, 127))
                if distribution[next_shape] == 0:
                    choices.remove(next_shape)
            else:
                distribution[next_shape] -= 1
                shape, label = shapes.createRandomPolygon(next_shape + 2)
                if distribution[next_shape] == 0:
                    choices.remove(next_shape)

            arr = np.zeros((self.y, self.x))
            shapes.addShape(arr, shape, np.random.randint(0, 255, 2))
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


def create_datasets(classes, train_size, test_size, eval_size = 0):
    np.random.seed(0)
    size = train_size + test_size + eval_size
    x_max, y_max = 256, 256
    total = 0
    min_radius = 15
    stepsize = 1
    x_values = np.arange(min_radius, x_max - min_radius, stepsize)
    y_values = np.arange(min_radius, y_max - min_radius, stepsize)
    np.random.shuffle(x_values)
    np.random.shuffle(y_values)
    dataset = []
    labels = []

    for x in x_values:
        if total >= size:
            break
        for y in y_values:
            if total >= size:
                break
            max_radius = min(min(x, y), x_max - max(x, y))
            radius_stepsize = 1
            radii = np.arange(min_radius, max_radius, radius_stepsize)
            angle = (2 * np.pi / 360) * np.random.random_integers(0, 359)
            np.random.shuffle(radii)
            radii = radii[:min(len(radii), int(np.cbrt(size)))]
            for radius in radii:
                if total >= size:
                    break
                for class_ in classes:
                    if total >= size:
                        break
                    if class_ == 0:
                        shape, label = shapes.Circle(radius), 0
                    elif class_ == 1:
                        shape, label = shapes.createRandomPolygon(3)
                    else:
                        shape, label = shapes.createPolygon(class_ + 2, radius, angle)
                    dataset.append(shapes.get_image(shape, [x, y]))
                    labels.append(label)
                    total += 1

    train_ds = ShapeDataset(classes, size, create=False)
    train_ds.dataset = dataset[:train_size]
    train_ds.labels = labels[:train_size]
    test_ds = ShapeDataset(classes, size, create=False)
    test_ds.dataset = dataset[train_size:train_size + test_size]
    test_ds.labels = labels[train_size:train_size + test_size]
    eval_ds = ShapeDataset(classes, size, create=False)
    eval_ds.dataset = dataset[train_size + test_size:]
    eval_ds.labels = labels[train_size + test_size:]
    return train_ds, test_ds, eval_ds


class DynamicShapesDataset(Dataset):
    def __init__(self, classes, size):
        self.x_range, self.y_range = 256, 256
        self.dataset = []
        self.labels = []
        self.classes = classes
        self.size = size
        self.build()


    def build(self):
        for a in range(self.size):
            x, y = np.random.randint(20, self.x_range - 20), np.random.randint(20, self.y_range - 20)
            radius = np.random.randint(10, min(min(x, y), self.x_range - max(x, y)))
            angle = int(np.pi*np.random.randint(359)/180)
            shape_class = np.random.choice(self.classes)
            if shape_class == 0:
                img = shapes.get_image(shapes.Circle(radius), [x, y], self.x_range)
            else:
                shape, shape_label = shapes.createPolygon(shape_class+2, radius, angle)
                img = shapes.get_image(shape, [x, y], self.x_range)

            label = [shape_class, (2*radius+10)/self.x_range, (x-5)/self.x_range, (y-5)/self.y_range]
            self.dataset.append(img)
            self.labels.append(label)

        self.dataset = np.array(self.dataset)
        self.labels = np.array(self.labels)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.dataset[index]
        label = self.labels[index]

        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)

        return img, int(label[0]), np.array(label[1:], dtype=np.float32)