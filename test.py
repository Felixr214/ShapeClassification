import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from copy import deepcopy

import cached_dataset as dataset
import models

np.random.seed(17)
torch.manual_seed(17)

classes = [0, 1, 2, 3, 4, 5, 6]
grid_size = 8
max_shapes = 3

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

device = torch.device("xpu")
m = models.ShapeCNN(len(classes))
m.load_state_dict(torch.load("trained_models/model.pth"))
m.to(device)
m.eval()

test_dataset = dataset.DynamicShapesDataset(classes, 10, grid_size, max_shapes, True)
batch_size = len(test_dataset)
data_loader = DataLoader(test_dataset, batch_size=1)
img_ = []
boxes = []
box_labels = []
shape_centers = []

for img, label in data_loader:
    img_.append(deepcopy(img).numpy())
    img = img.unsqueeze(0)
    img = img.to(device)

    objectness_label = label[0, 0]
    class_label = label[0, 1]
    box_label = label[0, 2:]

    boxes.append([])
    box_labels.append([])
    shape_centers.append([])

    pred_objectness, pred_classes, pred_box = m(img)
    pred_classes = pred_classes.cpu().detach().squeeze(0)
    pred_objectness = pred_objectness.cpu().detach().squeeze(0)
    pred_box = pred_box.cpu().detach().squeeze(0)
    # x, y, size


    for b in range(grid_size):
        x_offset = b * ppc
        for a in range(grid_size):
            y_offset = a * ppc
            if objectness_label[a, b] == 1:
                label_x, label_y, label_size = int(x_offset + box_label[0, a, b] * ppc), int(y_offset + box_label[1, a, b] * ppc), 3
                shape_centers[-1].append(patches.Circle((label_x, label_y), label_size, fill=True, color='green'))
            if pred_objectness[a, b] >= 0.95:
                pred_class = int(np.argmax(pred_classes[:, a, b]))
                x, y, size = int(x_offset + pred_box[0, a, b]*ppc), int(y_offset + pred_box[1, a, b]*ppc), 2*int(pred_box[2, a, b]*test_dataset.max_radius)
                boxes[-1].append(patches.Rectangle((x-(size//2), y-(size//2)), size, size, linewidth=2, edgecolor="red", facecolor='none'))
                box_labels[-1].append([x-(size//2)+3, y-(size//2)-5, label_translation[pred_class]])


i = 0
for img in img_:
    fig, ax = plt.subplots()
    ax.imshow(img[0], cmap="gray_r")
    #print(f"{len(boxes[i])}/{len(shape_centers[i])}")
    #for dot in shape_centers[i]:
        #ax.add_patch(dot)

    for box_index in range(len(boxes[i])):
        box = boxes[i][box_index]
        ax.add_patch(box)
        text = ax.text(*box_labels[i][box_index], color="black", backgroundcolor="red")

    fig.savefig(f"test_results/output_{i}.png")
    i+=1
