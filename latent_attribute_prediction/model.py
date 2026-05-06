import torch
import torch.nn as nn

class ResidualMLPBlock(nn.Module):
    def __init__(self, n_channels: int):
        super(ResidualMLPBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, n_channels),
            nn.SiLU(),
        )
    
    def forward(self, x):
        return x + self.layers(x)

class MLP_ELU_convex(nn.Module):
    def __init__(self):
        super(MLP_ELU_convex, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            ResidualMLPBlock(256),
            ResidualMLPBlock(256),
            ResidualMLPBlock(256),
            nn.Linear(256, 40),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.f(x)