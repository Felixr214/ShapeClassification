import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from copy import deepcopy

import dataset
import models

classes = [0, 1, 2, 3]
train_shape_count = len(classes)*2000
test_shape_count = int(train_shape_count * 0.2)
eval_shape_count = int(len(classes) * 3)

train_ds, test_ds, eval_ds = dataset.create_datasets(classes, train_shape_count, test_shape_count, eval_shape_count)

train_loader = DataLoader(train_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)
eval_loader = DataLoader(eval_ds, batch_size=64)

device = torch.device("xpu")
m = models.ShapeCNN(len(classes))
m.to(device)

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

lr = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m.parameters(), lr=lr)
last_acc = 0

for epoch in range(10):
    train_loss = train_epoch(m, train_loader)
    test_acc = eval_epoch(m, test_loader)

    print(f"Epoch {epoch+1}: Loss: {train_loss:.4f} | Test Acc: {test_acc:.3f}")


label_translation = {
    0: "circle",
    1: "triangle",
    2: "rectangle",
    3: "pentagon",
    4: "hexagon",
    5: "heptagon",
    6: "octagon"
}

for x, y in eval_loader:
    x_ = deepcopy(x).numpy()
    x, y = x.to(device), y.to(device)
    outputs = m(x)
    preds = outputs.argmax(dim=1)
    preds = preds.detach().cpu().numpy()

i = 0
for x in x_:
    plt.imshow(x[0])
    plt.title(label_translation[preds[i]])
    plt.show()
    i+=1
