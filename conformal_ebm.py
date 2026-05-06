import torch
from geometry_tools import lerp, subdivide_curve
from tqdm import trange
import numpy as np

def ebm_curve_energy(curve: torch.Tensor, log_likelihood) -> torch.Tensor:
    dx = curve[1:] - curve[:-1]
    curve_inv_lp = -log_likelihood(curve)
    ref = curve_inv_lp[0]
    forward = torch.logsumexp(torch.log(dx.square().sum(-1)) + curve_inv_lp[:-1] - ref, dim=0)
    dx = curve[:-1] - curve[1:]
    backward = torch.logsumexp(torch.log(dx.square().sum(-1)) + curve_inv_lp[1:] - ref, dim=0)
    return (forward + backward) / 2.0

def ebm_curve_length(curve: torch.Tensor, log_likelihood) -> torch.Tensor:
    dx = curve[1:] - curve[:-1]
    curve_inv_lp = -log_likelihood(curve)
    ref = curve_inv_lp[0]
    forward = torch.logsumexp(0.5*(torch.log(dx.square().sum(-1)) + curve_inv_lp[:-1] - ref), dim=0)
    dx = curve[:-1] - curve[1:]
    backward = torch.logsumexp(0.5*(torch.log(dx.square().sum(-1)) + curve_inv_lp[1:] - ref), dim=0)
    return (forward + backward) / 2.0


def optimize_ebm_geodesic(x1: torch.Tensor, x2: torch.Tensor, log_likelihood, n_segments=16, n_iter=10000, lr=1e-3, n_subdiv=0, eps=1e-4, patience=5, device='cuda', verbose=True) -> torch.Tensor:
    """numerically optimize a geodesic with two given endpoints"""
    og_device = x1.device
    x1, x2 = x1.to(device), x2.to(device)
    curve = lerp(x1, x2, n_segments).detach()
    curve_interior = curve[1:-1].requires_grad_(True)
    optimizer = torch.optim.Adam([curve_interior], lr=lr)
    
    best_energy = np.inf
    last_energy = best_energy
    pat = patience
    best_curve = torch.empty_like(curve)
    pbar = trange(n_iter, desc='optimizing ebm geodesic', disable = not verbose)
    for _ in pbar:
        optimizer.zero_grad()
        full_curve = torch.cat((curve[[0]], curve_interior, curve[[-1]]))
        subdiv_curve = subdivide_curve(full_curve, n_subdiv) if n_subdiv else full_curve
        curve_e = ebm_curve_energy(subdiv_curve, log_likelihood)
        if curve_e.item() < best_energy:
            best_curve = full_curve.detach()
            best_energy = curve_e.item()
        if abs(curve_e.item() - last_energy) < eps:
            pat -= 1
            if not patience:
                print('Early stopping!')
        else:
            pat = patience
        last_energy = curve_e.item()
        curve_e.backward()
        optimizer.step()
        pbar.set_postfix(loss=curve_e.item())
    
    return best_curve.to(og_device)

def optimize_ebm_mean(*points: torch.Tensor, log_likelihood, n_segments=16, n_iter=10000, lr=1e-3, n_subdiv=0, eps=1e-4, patience=5, device='cuda', verbose=True):
        """numerically optimize the Karcher mean of a given set of points on the manifold"""
        assert len(points) >= 2, "Need at least two points to compute Karcher mean"
        og_device = points[0].device
        xs = [x.to(device) for x in points]
        with torch.no_grad():
            mean: torch.Tensor = (sum(xs, torch.zeros(())) / len(xs))
            curves = [lerp(x, mean, n_segments).detach() for x in xs]
        curve_interiors = [curve[1:-1].requires_grad_() for curve in curves]
        mean = mean.requires_grad_()
        optimizer = torch.optim.Adam(curve_interiors + [mean], lr=lr)

        best_energy = np.inf
        last_energy = best_energy
        pat = patience
        best_mean = torch.empty_like(mean)

        pbar = trange(n_iter, desc='optimizing Karcher mean', disable = not verbose)
        for _ in pbar:
            optimizer.zero_grad()
            full_curves = [torch.cat((curve[[0]], curve_interior, mean.unsqueeze(0))) for curve, curve_interior in zip(curves, curve_interiors)]
            subdiv_curves = [subdivide_curve(full_curve, n_subdiv) if n_subdiv else full_curve for full_curve in full_curves]
            curves_e = sum((torch.square(ebm_curve_length(subdiv_curve, log_likelihood)) for subdiv_curve in subdiv_curves), torch.zeros(()))
            if curves_e.item() < best_energy:
                best_mean = mean.detach()
            if abs(curves_e.item() - last_energy) < eps:
                pat -= 1
                if not patience:
                    print('Early stopping!')
            else:
                pat = patience
            last_energy = curves_e.item()
            curves_e.backward()
            optimizer.step()
            pbar.set_postfix(loss=curves_e.item())
        
        return best_mean.to(og_device)