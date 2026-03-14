import time

import numpy as np
import torch
from torch.utils.data import Dataset

angle_offset = {}
shapes = {}

def deg2rad(x):
    return torch.pi*x/180

def buildCache(classes, angles, radii):
    rad_angles = deg2rad(angles)
    for v in range(3, max(classes)+3):
        angle_offset[v] = (2 * torch.pi * torch.arange(0, 360, 360 / v) / 360)

    for class_ in classes:
        shapes[class_] = {}

    for radius in radii:
        xx, yy = torch.meshgrid(torch.arange(0, 2*radius), torch.arange(0, 2*radius), indexing="ij")
        coords = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).unsqueeze(2)
        radius = int(radius)
        for class_ in classes:
            vertices = class_+2
            if class_ == 0:
                shapes[0, radius] = getCircle(radius, radius, radius, xx, yy)
                continue

            shapes[class_, radius] = {}
            for a in range(len(angles)):
                angle = int(angles[a])
                rad_angle = rad_angles[a]
                shapes[class_, radius, angle] = getPolygon(vertices, radius, rad_angle, radius, radius, coords)

def getCircle(radius, x, y, xx, yy):
    return (xx - x) ** 2 + (yy - y) ** 2 < radius**2

def getPolygon(vertices, radius, angle, x, y, coords):
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


def getRandomImage(classes, max_shapes, grid_cells, angles, radii, max_radius):
    shapesPerImage = 1
    ppc = 256 // grid_cells
    img = torch.zeros([256, 256])
    label = torch.zeros([5, grid_cells, grid_cells])
    if max_shapes > 1:
        shapesPerImage = torch.randint(1, max_shapes+1, [1])

    for a in range(shapesPerImage):
        shape_class = int(classes[torch.randint(0, len(classes), [1])])
        radius = int(np.random.choice(radii))
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
            label[4, celly, cellx] = radius / max_radius

        if shape_class == 0:
            shape = shapes[0, radius]
        else:
            angle = int(np.random.choice(angles))
            shape = shapes[shape_class, radius, angle]

        # change x and y
        img[y-radius:y+radius, x-radius:x+radius] = torch.logical_or(img[y-radius:y+radius, x-radius:x+radius], shape)

    img = img.to(torch.float32)

    return img, label

class DynamicShapesDataset(Dataset):
    def __init__(self, classes, size, grid_size, max_shapes, testing=False):
        self.x_range, self.y_range = 256, 256
        self.classes = classes
        self.size = size
        self.grid_size = grid_size
        self.images = []
        self.labels = []
        self.max_shapes_per_image = max_shapes
        self.max_radius = 65
        if testing:
            self.angles = torch.tensor([0, 13, 56])
            self.radii = torch.tensor([10, 56, self.max_radius])
        else:
            self.angles = torch.arange(0, 90)
            self.radii = torch.arange(10, self.max_radius)
        t0 = time.time()
        buildCache(self.classes, self.angles, self.radii)
        print(f"built cache, {time.time()-t0} seconds")
        self.update()

    def update(self):
        self.images = []
        self.labels = []
        for _ in range(self.size):
            img, label = getRandomImage(self.classes, self.max_shapes_per_image, self.grid_size, self.angles, self.radii, self.max_radius)
            self.images.append(img)
            self.labels.append(label)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.images[index], self.labels[index]