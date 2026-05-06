import torch
from geometry_tools import slerp
from torch.nn.functional import mse_loss

def curve_true_pos(curve: torch.Tensor, scoring_model, labelsA: torch.Tensor, labelsB: torch.Tensor) -> torch.Tensor:
    """higher is better"""
    return torch.sum(scoring_model(curve) * (labelsA.bool() & labelsB.bool())[..., None, :])

def curve_false_pos(curve: torch.Tensor, scoring_model, labelsA: torch.Tensor, labelsB: torch.Tensor) -> torch.Tensor:
    """lower is better"""
    return torch.sum(scoring_model(curve) * ~(labelsA.bool() | labelsB.bool())[..., None, :])

def curve_posterior_slerp_deviation(curve: torch.Tensor, posterior) -> torch.Tensor:
    curve_posterior = posterior(curve)
    start, end = curve_posterior[0], curve_posterior[-1]
    slerp_target = slerp(start.sqrt(), end.sqrt(), torch.linspace(0, 1, curve.shape[0], device=curve.device)[:, None]).square()
    return mse_loss(curve_posterior, slerp_target, reduction='sum')