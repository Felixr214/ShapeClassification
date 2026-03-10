import numpy as np
import torch
from torch.utils.data import Dataset

torch.manual_seed(1)

xx, yy = torch.meshgrid(torch.arange(0, 256), torch.arange(0, 256), indexing="ij")
coords = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).unsqueeze(2)
angle_offset = {}
for v in range(3, 10):
    angle_offset[v] = (2 * torch.pi * torch.arange(0, 360, 360 / v) / 360)

def getCircle(radius, x, y):
    return (xx - x) ** 2 + (yy - y) ** 2 < radius**2

def getPolygon(vertices, radius, angle, x, y):
    start = torch.tensor([x, y, 1], dtype=torch.float32)

    angles = angle + angle_offset[vertices]
    points = torch.stack([start[0] - torch.sin(angles) * radius,
                          start[1] - torch.cos(angles) * radius,
                          torch.ones(vertices)], dim=-1)

    edges = torch.cross(points.roll(1, 0), points, dim=-1)
    edge_vals = (coords * edges.view(1, 1, vertices, 3)).sum(-1)
    inverse = -(start * edges).sum(-1)
    mask = inverse.view(1, 1, vertices) * edge_vals <= 0
    return mask.all(dim=-1)

def getRandomImage(shapes, max_shapes, grid_cells):
    shapesPerImage = 1
    ppc = 256 // grid_cells
    img = torch.zeros([256, 256])
    label = torch.zeros([5, grid_cells, grid_cells])
    if max_shapes > 1:
        shapesPerImage = torch.randint(1, max_shapes+1, [1])

    for a in range(shapesPerImage):
        shape_class = int(shapes[torch.randint(0, len(shapes), [1])])
        radius = int(torch.randint(10, 60, [1]))
        x, y = torch.randint(radius, 256-radius, [2])
        cellx = x // ppc
        celly = y // ppc
        if label[0, celly, cellx] == 1:
            continue
        else:
            label[0, celly, cellx] = 1
            label[1, celly, cellx] = shape_class
            label[2, celly, cellx] = (x%ppc) / ppc
            label[3, celly, cellx] = (y%ppc) / ppc
            label[4, celly, cellx] = radius / 256

        if shape_class == 0:
            shape = getCircle(radius, x, y)
        else:
            angle = float(torch.pi * torch.randint(0, 89, [1]) / 180)
            shape = getPolygon(shape_class+2, radius, angle, x, y)

        img = torch.logical_or(img, shape)

    img = img.to(torch.float32)

    return img, label

class DynamicShapesDataset(Dataset):
    def __init__(self, classes, size, grid_size):
        self.x_range, self.y_range = 256, 256
        self.classes = classes
        self.size = size
        self.grid_size = grid_size
        self.images = []
        self.labels = []
        self.max_shapes_per_image = grid_size
        self.update()

    def update(self):
        self.images = []
        self.labels = []
        for _ in range(self.size):
            img, label = getRandomImage(self.classes, self.max_shapes_per_image, self.grid_size)
            self.images.append(img)
            self.labels.append(label)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
