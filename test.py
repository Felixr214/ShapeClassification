import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from copy import deepcopy

import dataset
import models

np.random.seed(17)

classes = [0, 1, 2, 3, 4, 5, 6]
grid_size = 5

ppc = 256 // grid_size

label_translation = {
    0: "circle",
    1: "triangle",
    2: "rectangle",
    3: "pentagon",
    4: "hexagon",
    5: "heptagon",
    6: "octagon"
}

label_colors = {
    0: "red",
    1: "green",
    2: "blue",
    3: "yellow",
    4: "orange",
    5: "black",
    6: "purple"
}

device = torch.device("xpu")
m = models.ShapeCNN(len(classes), grid_size)
m.load_state_dict(torch.load("trained_models/model.pth"))
m.to(device)
m.eval()

test_dataset = dataset.DynamicShapesDataset(classes, 10, grid_size)
batch_size = len(test_dataset)
data_loader = DataLoader(test_dataset, batch_size=1)
img_ = []
boxes = []
box_labels = []

for img, label in data_loader:
    img_.append(deepcopy(img).numpy())
    img = img.unsqueeze(0)
    img = img.to(device)

    objectness_label = label[0, 0]
    class_label = label[0, 1]
    box_label = label[0, 2:]

    boxes.append([])
    box_labels.append([])
    pred_classes, pred_objectness, pred_box = m(img)

    pred_classes = pred_classes.cpu().detach().squeeze(0)
    pred_objectness = pred_objectness.cpu().detach().squeeze(0)
    pred_box = pred_box.cpu().detach().squeeze(0)
    # x, y, size


    for a in range(grid_size):
        x_offset = a * ppc
        for b in range(grid_size):
            y_offset = b * ppc
            if objectness_label[a, b] == 1:
                label_size, label_x, label_y = 3, int(x_offset + box_label[1, a, b] * ppc), int(y_offset + box_label[0, a, b] * ppc)
                box_labels[-1].append(patches.Circle((label_x, label_y), label_size, fill=True, color='green'))
            if pred_objectness[a, b] >= 0.99:
                pred_class = np.argmax(pred_classes[0, :, a, b])
                size, x, y = int(pred_box[2, a, b]*256), int(x_offset + pred_box[1, a, b]*ppc), int(y_offset + pred_box[0, a, b]*ppc)
                boxes[-1].append(patches.Rectangle((x-(size//2), y-(size//2)), size, size, linewidth=1, edgecolor=label_colors[pred_class], facecolor='none', label=label_translation[pred_class]))


i = 0
for img in img_:
    fig, ax = plt.subplots()
    ax.imshow(img[0], cmap="gray_r")
    print(f"{len(boxes[i])}/{len(box_labels[i])}")
    for dot in box_labels[i]:
        ax.add_patch(dot)

    for box in boxes[i]:
        ax.add_patch(box)

    #ax.set_title(label_translation[classes[i][0]])
    if len(boxes[i]) >= 1:
        ax.legend()
    fig.savefig(f"test_results/output_{i}.png")
    i+=1
