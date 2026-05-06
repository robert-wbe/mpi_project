import torch
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

def lerp(u: torch.Tensor, v: torch.Tensor, n=50) -> torch.Tensor:
    lbda = torch.linspace(0, 1, n, device=u.device)
    return torch.movedim(lbda * v[..., None] + (1-lbda) * u[..., None], -1, 0)

def slerp(u, v, t):
    omega = torch.arccos(torch.dot(u, v))
    return (
        (torch.sin((1-t) * omega) / torch.sin(omega)) * u
        + (torch.sin(t * omega) / torch.sin(omega)) * v
    )

def slerp_curve(u, v, n=8):
    t = torch.linspace(0, 1, n, device=u.device)[:, None]
    l1 = torch.norm(u)
    l2 = torch.norm(v)
    u_normed, v_normed = u / l1, v / l2
    omega = torch.arccos(torch.dot(u_normed, v_normed))
    return (
        (torch.sin((1-t) * omega) / torch.sin(omega)) * u
        + (torch.sin(t * omega) / torch.sin(omega)) * v
    )

# trick for numerical stability of interpolant optimization
def subdivide_curve(curve: torch.Tensor, n_subdivisions=3):
    lmda = torch.linspace(0, 1, 2+n_subdivisions, device=curve.device)[:-1][:, None]
    s = lmda * curve[1:, None] + (1-lmda) * curve[:-1, None]
    return torch.cat((torch.flatten(s, 0, 1), curve[-1][None]))

def psd_mat_sqrt(mats: torch.Tensor):
    eigvals, eigvecs = torch.linalg.eigh(mats)
    return torch.einsum('kid, kd, kjd -> kij', eigvecs, torch.sqrt(eigvals), eigvecs)
def unit_circle(resolution: int = 32):
    t = torch.linspace(0, 2*torch.pi, resolution+1)
    return torch.stack((torch.cos(t), torch.sin(t)), dim=-1)
def plot_2d_gmm(means: torch.Tensor, covs: torch.Tensor, scale_fac=1.5, resolution: int=32, color='k'):
    means, covs = means.detach(), covs.detach()
    ellipsoids = scale_fac*torch.einsum('kid,rd->irk', psd_mat_sqrt(covs), unit_circle(resolution)) + means.T[:, None, :]
    plt.plot(*ellipsoids, c=color)

def scalerot_to_cov(scale: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    rot_mat = torch.stack((
        torch.stack((torch.cos(rot), -torch.sin(rot)), dim=-1),
        torch.stack((torch.sin(rot), torch.cos(rot)), dim=-1)
    ), dim=-2)

    return torch.einsum('...ik,...k,...jk->...ij', rot_mat, scale.square(), rot_mat)
