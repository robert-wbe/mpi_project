import torch
from typing import NamedTuple

class InterpolationGroup(NamedTuple):
    linline: torch.Tensor
    dec_geodesic: torch.Tensor
    ebm_geodesic: torch.Tensor
    geodesic: torch.Tensor
    labels: torch.Tensor

class InterpolationGroupExtra(NamedTuple):
    slinline: torch.Tensor
    ebm_geodesic_2: torch.Tensor

class InterpolationGroupExtra2(NamedTuple):
    interp_latent: torch.Tensor
    interp_outer: torch.Tensor


class MergedInterpolationGroup(NamedTuple):
    linline: torch.Tensor
    dec_geodesic: torch.Tensor
    ebm_geodesic: torch.Tensor
    geodesic: torch.Tensor
    slinline: torch.Tensor
    ebm_geodesic_2: torch.Tensor
    interp_latent: torch.Tensor
    interp_outer: torch.Tensor
    labels: torch.Tensor

interpolations_1 = torch.load('pickled_objects/interpolations_new.pt')
interpolations_2 = torch.load('pickled_objects/interpolations_extra.pt')
interpolations_3 = torch.load('pickled_objects/interpolations_extra2.pt')

interpolations = []

for (
        (linline, dec_geodesic, ebm_geodesic, geodesic, labels),
        (slinline, ebm_geodesic_2),
        (interp_latent, interp_outer),
    ) in zip(interpolations_1, interpolations_2, interpolations_3):
    interpolations.append(MergedInterpolationGroup(
        linline, dec_geodesic, ebm_geodesic, geodesic, slinline, ebm_geodesic_2, interp_latent, interp_outer, labels
    ))

torch.save(interpolations, 'pickled_objects/interpolations_merged.pt')