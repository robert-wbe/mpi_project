import torch
import torch.nn as nn
import torchvision

model = nn.Sequential(
    torchvision.models.resnet50(),
    nn.Linear(1000, 40),
    nn.Sigmoid()
)