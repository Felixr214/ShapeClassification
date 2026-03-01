import torch
import numpy as np
from torch.utils.data import DataLoader

import dataset
import models

np.random.seed(1)

classes = [0, 1, 2, 3, 4, 5, 6]
grid_size = 5
max_shapes_per_image = grid_size

device = torch.device("xpu")
m = models.ShapeCNN(len(classes), grid_size)
m.to(device)

steps = 5000
w = grid_size*2
class_criterion = torch.nn.CrossEntropyLoss(reduction='none')  # no reduction, we'll mask manually
objectness_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  # per-cell
box_criterion = torch.nn.MSELoss(reduction='none')  # fallback, can replace with GIoU
optimizer = torch.optim.SGD(
    m.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
dataset_size = 500

# ----- Loss components -----


eps = 1e-6  # small value to avoid division by zero

# pos_weight for objectness (to balance rare positives vs background)
w = grid_size * 2
pos_weight = torch.tensor([w]).to(device)

def totalLoss(pred_class, pred_box, pred_objectness, target_class, target_box, target_objectness):
    """
    YOLO-style total loss for synthetic shapes.

    pred_class: [B, G, G, num_classes] logits
    pred_box:   [B, 3, G, G] (x_center, y_center, size) normalized 0..1
    pred_objectness: [B, G, G] logits
    target_class: [B, G, G] int labels
    target_box:   [B, G, G, 3] floats
    target_objectness: [B, G, G] 0/1
    """

    mask = target_objectness.float()  # 1 for object cells, 0 for background

    # -------- Class loss --------
    class_loss_raw = class_criterion(pred_class, target_class)  # [B, G, G]
    class_loss = (class_loss_raw * mask).sum() / (mask.sum() + eps)

    # -------- Box loss --------

    box_loss_raw = box_criterion(pred_box, target_box).mean(1)  # reduce last dim (4)
    box_loss = (box_loss_raw * mask).sum() / (mask.sum() + eps)

    # -------- Objectness loss --------
    objectness_loss_raw = objectness_criterion(pred_objectness, target_objectness)
    # apply pos_weight: scale positive cells
    objectness_loss = ((objectness_loss_raw * mask * w) + (objectness_loss_raw * (1 - mask))).mean()

    # -------- Weighted sum --------
    # typical YOLO weights: box loss stronger
    total_loss = class_loss + objectness_loss + 3 * box_loss

    return total_loss

def train_step(model, loader):
    model.train()
    batch = next(iter(loader))
    img, class_label, objectness_label, box_label = batch

    img = img.to(device)
    class_label = class_label.to(device)
    box_label = box_label.to(device)
    objectness_label = objectness_label.to(device)

    optimizer.zero_grad()

    pred_class, pred_objectness, pred_box = model(img)

    loss = totalLoss(
        pred_class,
        pred_box,
        pred_objectness,
        class_label,
        box_label,
        objectness_label
    )

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()

for step in range(steps):
    ds = dataset.DynamicShapesDataset(classes, dataset_size, grid_size, max_shapes_per_image)
    batch_size = len(ds.labels)
    data_loader = DataLoader(ds, batch_size=batch_size)
    train_loss = train_step(m, data_loader)
    if step % 50 == 0:
        print(f"Step {step + 1}: Loss: {train_loss:.4f}")

torch.save(m.state_dict(), "trained_models/model.pth")
