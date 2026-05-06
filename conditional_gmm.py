import torch
from typing import Self
from dataclasses import dataclass
import torch.distributions as D

class ConditionalGMM:
    def __init__(self, n_dim: int, n_classes: int, device='cuda', means: torch.Tensor | None = None, covs: torch.Tensor | None = None):
        self.n_classes = n_classes
        if means==None or covs==None:
            self.means = torch.zeros(n_classes, n_dim, device=device)
            self.covs = torch.zeros(n_classes, n_dim, n_dim, device=device)
            self.N = torch.zeros(n_classes, device=device)
        else:
            self.means = means
            self.covs: torch.Tensor = covs
            self.prior = torch.ones(n_classes, device=means.device) / n_classes
            self.mix = D.Categorical(self.prior)
            self.comp = D.MultivariateNormal(self.means, self.covs)
            self.gmm = D.MixtureSameFamily(self.mix, self.comp)
        
    def update(self, X: torch.Tensor, labels: torch.Tensor):
        for i in range(self.n_classes):
            if i not in labels: continue
            self.means[i] += X[labels == i].sum(dim=0)
            self.covs[i] += (X[labels == i, :, None] * X[labels == i, None, :]).sum(0)
            self.N[i] += (labels == i).sum()
    
    def update_multiple(self, X: torch.Tensor, labels: torch.Tensor):
        for i in range(self.n_classes):
            if not labels[:,i].any(): continue
            self.means[i] += X[labels[:, i].bool()].sum(dim=0)
            self.covs[i] += (X[labels[:, i].bool(), :, None] * X[labels[:, i].bool(), None, :]).sum(0)
            self.N[i] += labels[:, i].sum()
    
    def fit(self):
        self.means /= self.N[:, None]
        self.covs /= (self.N[:, None, None] + 1)
        self.covs -= self.means[:, :, None] * self.means[:, None, :]
        self.prior = self.N / torch.sum(self.N)
        self.mix = D.Categorical(self.prior)
        self.comp = D.MultivariateNormal(self.means, self.covs)
        self.gmm = D.MixtureSameFamily(self.mix, self.comp)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.comp.log_prob(x[..., None, :])

    def uncond_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x).logsumexp(dim=-1)
    
    def posterior(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.log_prob(x) + torch.log(self.prior), dim=-1)
    
    def log_posterior(self, x: torch.Tensor) -> torch.Tensor:
        lp = self.log_prob(x)
        return lp - lp.logsumexp(dim=-1, keepdim=True)