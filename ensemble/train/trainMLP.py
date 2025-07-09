import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import os

class BisnisAssistantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(7, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.model(x)

X = pd.read_csv("ensemble/data/X_scaled.csv").values
y = pd.read_csv("ensemble/data/y_scaled.csv").values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

model = BisnisAssistantModel()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    for xb, yb in dataloader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

os.makedirs("ensemble/models", exist_ok=True)
torch.save(model.state_dict(), "ensemble/models/mlp_model.pth")
