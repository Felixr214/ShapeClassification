import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from copy import deepcopy

#import dataset
from test_functions import DynamicShapesDataset
import models

np.random.seed(17)

classes = [0, 1, 2, 3, 4, 5, 6]
grid_size = 1

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

test_dataset = DynamicShapesDataset(classes, 10, grid_size, 1)
batch_size = len(test_dataset)
data_loader = DataLoader(test_dataset, batch_size=1)
img_ = []
boxes = []
box_labels = []
for img, class_label, objectness, box in data_loader:
    img_.append(deepcopy(img).numpy())
    img = img.to(device)
    boxes.append([])
    box_labels.append([])
    pred_classes, pred_objectness, pred_box = m(img)

    pred_classes = pred_classes.cpu().detach().numpy()
    pred_objectness = pred_objectness.cpu().detach().numpy()
    pred_box = pred_box.cpu().detach().numpy() # size, x, y

    for a in range(grid_size):
        for b in range(grid_size):
            if objectness[0, a, b] == 1:
                label_size, label_x, label_y = 3, int(box[0][1][a][b] * 256), int(box[0][2][a][b] * 256)
                box_labels[-1].append(patches.Circle((label_x, label_y), label_size, fill=True, color='green'))
            if pred_objectness[0, a, b] >= 0.5:
                pred_class = np.argmax(pred_classes[0, :, a, b])
                size, x, y = int(pred_box[0][0][a][b]*256), int(pred_box[0][1][a][b]*256), int(pred_box[0][2][a][b]*256)
                boxes[-1].append(patches.Rectangle((x-(size//2), y-(size//2)), size, size, linewidth=1, edgecolor=label_colors[pred_class], facecolor='none', label=label_translation[pred_class]))


i = 0
for img in img_:
    fig, ax = plt.subplots()
    ax.imshow(img[0][0], cmap="gray_r")
    print(f"{len(boxes[i])}/{len(box_labels[i])}")
    for dot in box_labels[i]:
        ax.add_patch(dot)

    for box in boxes[i]:
        ax.add_patch(box)

    #ax.set_title(label_translation[classes[i][0]])
    ax.legend()
    fig.savefig(f"test_results/output_{i}.png")
    i+=1
