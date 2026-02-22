import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from copy import deepcopy

import dataset
import models

np.random.seed(1)

classes = [0, 1, 2, 3, 4, 5, 6]

device = torch.device("xpu")
m = models.ShapeCNN(len(classes))
m.load_state_dict(torch.load("trained_models/model.pth"))
m.to(device)

test_dataset = dataset.DynamicShapesDataset(classes, 1)

label_translation = {
    0: "circle",
    1: "triangle",
    2: "rectangle",
    3: "pentagon",
    4: "hexagon",
    5: "heptagon",
    6: "octagon"
}

batch_size = len(test_dataset)
data_loader = DataLoader(test_dataset, batch_size=batch_size)
m.eval()
img_ = []
for img, class_label, box in data_loader:
    img_ = deepcopy(img).numpy()
    box_ = deepcopy(box).numpy()

    img = img.to(device)
    pred_class, pred_box = m(img)
    pred_class = pred_class.argmax(dim=1)
    pred_class = pred_class.cpu().detach().numpy()
    pred_box = pred_box.cpu().detach().numpy() # size, x, y

i = 0
for img in img_:
    fig, ax = plt.subplots()
    ax.imshow(img[0], cmap="gray_r")
    size, x, y = pred_box[i]*256
    #tl
    rect = patches.Rectangle((x, y), size, size, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(label_translation[pred_class[i]])
    fig.savefig(f"test_results/output_{i}.png")
    i += 1
