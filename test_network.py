import torch.nn as nn
import torch 
import time
from LoRA import get_lora_params
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
    

def train_model(model, train_loader, epochs=3, device='cpu'):
    model.to(device)

    lora_params = get_lora_params(model)
    if len(lora_params) == 0:
        print("[INFO] No LoRA layers found — training full model.")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        print(f"[INFO] Found {len(lora_params)} LoRA params — training LoRA only.")
        optimizer = torch.optim.Adam(lora_params, lr=0.001)

    criterion = nn.CrossEntropyLoss()
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time
    return train_time


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

