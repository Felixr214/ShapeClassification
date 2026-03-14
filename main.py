import torch
import numpy as np
import time
from torch.utils.data import DataLoader

import cached_dataset as dataset
import models

torch.manual_seed(1)
np.random.seed(1)

classes = [0, 1, 2, 3, 4, 5, 6]
grid_size = 8
dataset_size = 300
max_shapes = 3
steps = 10000

device = torch.device("xpu")
m = models.ShapeCNN(len(classes))
m.to(device)


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

# ----- Loss components -----


eps = 1e-6  # small value to avoid division by zero

# pos_weight for objectness (to balance rare positives vs background)
w = grid_size * 2
pos_weight = torch.tensor([w]).to(device)

def totalLoss(pred_class, pred_box, pred_objectness, target_class, target_box, target_objectness):
    mask = target_objectness.float()

    class_loss_raw = class_criterion(pred_class, target_class)  # [B, G, G]
    class_loss = (class_loss_raw * mask).sum() / (mask.sum() + eps)


    box_loss_raw = box_criterion(pred_box, target_box).mean(1)
    box_loss = (box_loss_raw * mask).sum() / (mask.sum() + eps)

    objectness_loss_raw = objectness_criterion(pred_objectness, target_objectness)
    objectness_loss = ((objectness_loss_raw * mask * w) + (objectness_loss_raw * (1 - mask))).mean()

    total_loss = class_loss + objectness_loss + 3 * box_loss

    return total_loss

def train_step(model, loader):
    model.train()
    batch = next(loader)
    imgs, labels = batch

    imgs = imgs.view(dataset_size, 1, 256, 256)
    labels = labels.view(dataset_size, 5, grid_size, grid_size)

    objectness_label = labels[:, 0]
    class_label = labels[:, 1].long()
    box_label = labels[:, 2:]

    img = imgs.to(device, non_blocking=True)
    class_label = class_label.to(device, non_blocking=True)
    box_label = box_label.to(device, non_blocking=True)
    objectness_label = objectness_label.to(device, non_blocking=True)

    optimizer.zero_grad()

    pred_objectness, pred_class, pred_box = model(img)

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


if __name__ == "__main__":
    ds = dataset.DynamicShapesDataset(classes, dataset_size, grid_size, max_shapes)
    for step in range(steps):
        data_loader = iter(DataLoader(ds, batch_size=dataset_size, pin_memory=True))
        train_loss = train_step(m, data_loader)
        if step % 50 == 0:
            print(f"Step {step + 1}: Loss: {train_loss:.4f}")
        ds.update()

    torch.save(m.state_dict(), "trained_models/model.pth")
