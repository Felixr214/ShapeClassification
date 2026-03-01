import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy

import shapes

class DynamicShapesDataset(Dataset):
    def __init__(self, classes, size, grid_size, max_shapes_per_image):
        self.x_range, self.y_range = 256, 256
        self.dataset = []
        self.labels = []
        self.classes = classes
        self.size = size
        self.grid_size = grid_size
        self.max_shapes_per_image = max_shapes_per_image
        self.build()


    def build(self):
        ppc = self.x_range//self.grid_size
        shapes_per_image = 1
        for a in range(self.size):
            if self.max_shapes_per_image > 1:
                shapes_per_image = np.random.randint(1, self.max_shapes_per_image)
            image_shapes = {}
            for b in range(shapes_per_image):
                x, y = np.random.randint(20, self.x_range - 20), np.random.randint(20, self.y_range - 20)
                radius = np.random.randint(10, min(x, y, self.x_range - max(x, y)))
                angle = int(np.pi*np.random.randint(359)/180)
                shape_class = np.random.choice(self.classes)
                cell = (y//ppc)*self.grid_size + x//ppc
                if shape_class == 0:
                    shape = shapes.Circle(radius)
                    image_shapes[cell] = [shape_class, (2 * radius + 8) / self.x_range, (x + radius) / self.x_range, (y + radius) / self.y_range, x, y, shape]
                else:
                    shape = shapes.createPolygon(shape_class+2, radius, angle)
                    image_shapes[cell] = [shape_class, (2 * radius + 8) / self.x_range, 0, 0, x, y, shape]

            label = [np.zeros([self.grid_size, self.grid_size], dtype=np.float32) for b in range(5)]

            # objectness, class, bbox
            img = np.zeros([self.x_range, self.y_range])

            for cell in image_shapes.keys():
                shape_class, size, label_x, label_y, x, y, shape = image_shapes[cell]
                img = shapes.addShape(img, shape, [x, y], self.x_range)
                label_x, label_y = shape.center
                cell_x, cell_y = cell % self.grid_size, cell // self.grid_size

                label[0][cell_y][cell_x] = shape_class # class
                label[1][cell_y][cell_x] = 1 # objectness
                label[2][cell_y][cell_x] = size # bbox size
                label[3][cell_y][cell_x] = label_x # bbox x
                label[4][cell_y][cell_x] = label_y # bbox y

            label = np.array(label)
            self.dataset.append(deepcopy(img))
            self.labels.append(deepcopy(label))

        self.dataset = np.array(self.dataset)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.dataset[index]
        label = self.labels[index]

        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)

        class_label = np.array(label[0], dtype=int)
        objectness = torch.tensor(label[1], dtype=torch.float32)#.unsqueeze(0)
        bbox = label[2:]

        return img, class_label, objectness, bbox