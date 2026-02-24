import torch
import numpy as np
from torch.utils.data import DataLoader

import dataset
import models

np.random.seed(1)

classes = [0, 1, 2, 3, 4, 5, 6]

device = torch.device("xpu")
m = models.ShapeCNN(len(classes))
m.to(device)

def totalLoss(pred_class, pred_box, target_class, target_box):
    class_loss = class_criterion(pred_class, target_class)
    box_loss = box_criterion(pred_box, target_box)

    return class_loss + 8 * box_loss

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for img, class_label, box in loader:
        img, class_label, box = img.to(device), class_label.to(device), box.to(device)

        optimizer.zero_grad()
        pred_class, pred_box = model(img)

        loss = totalLoss(pred_class, pred_box, class_label, box)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

lr = 0.002
class_criterion = torch.nn.CrossEntropyLoss()
box_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(m.parameters(), lr=lr)
dataset_size = 570

for epoch in range(2000):
    ds = dataset.DynamicShapesDataset(classes, dataset_size)
    batch_size = len(ds.labels)
    data_loader = DataLoader(ds, batch_size=batch_size)
    train_loss = train_epoch(m, data_loader)
    if epoch % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss: {train_loss:.4f}")

torch.save(m.state_dict(), "trained_models/model.pth")
