import torch
from riemannian_geometry import PullbackRiemannianManifold
from collections.abc import Callable
from dataclasses import dataclass

@dataclass
class DecoderManifold(PullbackRiemannianManifold):
    decoder: Callable[[torch.Tensor], torch.Tensor]

    def coordinate_map(self, coords: torch.Tensor) -> torch.Tensor:
        return self.decoder(coords)