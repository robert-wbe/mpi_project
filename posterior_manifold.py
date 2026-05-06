from riemannian_geometry import PullbackRiemannianManifold
import torch
from collections.abc import Callable
from dataclasses import dataclass

@dataclass
class PosteriorManifold(PullbackRiemannianManifold):
    posterior: Callable[[torch.Tensor], torch.Tensor]

    def coordinate_map(self, coords: torch.Tensor) -> torch.Tensor:
        return self.posterior(coords)

@dataclass
class FisherPosteriorManifold(PullbackRiemannianManifold):
    posterior: Callable[[torch.Tensor], torch.Tensor]
    epsilon = 1e-8 # for numerical stability

    def coordinate_map(self, coords: torch.Tensor) -> torch.Tensor:
        return 2*torch.sqrt(self.posterior(coords) + self.epsilon)

@dataclass
class FisherLogPosteriorManifold(PullbackRiemannianManifold):
    log_posterior: Callable[[torch.Tensor], torch.Tensor]

    def coordinate_map(self, coords: torch.Tensor) -> torch.Tensor:
        return 2*torch.exp(self.log_posterior(coords) / 2)

@dataclass
class NewFisherLogPosteriorManifold(PullbackRiemannianManifold):
    log_posterior: Callable[[torch.Tensor], torch.Tensor]

    def coordinate_map(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_posterior(coords) / 2)
    
    def discrete_curve_kinetic_energy(self, curve: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(self.log_posterior(curve).unfold(0, 2, 1).mean(-1), dim=(0, 1))
