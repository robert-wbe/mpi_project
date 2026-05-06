import numpy as np
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Callable
from tqdm import trange
from geometry_tools import lerp, subdivide_curve
from typing import Iterable
from string import ascii_lowercase as alphabet

class RiemannianManifold(ABC):
    @abstractmethod
    def g(self, coords: torch.Tensor) -> torch.Tensor:
        """return the metric tensor g at the given extrinsic coordinates"""
        pass

    def discrete_curve_length(self, curve: torch.Tensor) -> torch.Tensor:
        x = curve[:-1]
        dx = curve[1:] - curve[:-1]
        forward = torch.sum(torch.sqrt(torch.einsum('tmv,tm,tv->t', self.g(x), dx, dx)))
        x = curve[1:]
        dx = curve[:-1] - curve[1:]
        backward = torch.sum(torch.sqrt(torch.einsum('tmv,tm,tv->t', self.g(x), dx, dx)))
        return (forward + backward) / 2.0
    
    def discrete_curve_kinetic_energy(self, curve: torch.Tensor) -> torch.Tensor:
        x = curve[:-1]
        dx = curve[1:] - curve[:-1]
        forward = torch.einsum('tmv,tm,tv->', self.g(x), dx, dx)
        x = curve[1:]
        dx = curve[:-1] - curve[1:]
        backward = torch.einsum('tmv,tm,tv->', self.g(x), dx, dx)
        return (forward + backward) / 2.0
    
    def Gamma(self, coords: torch.Tensor) -> torch.Tensor:
        """return the Christoffel symbols at the given extrinsic coordinates"""
        g_batched = lambda x: torch.einsum('...ij->ij', self.g(x))
        d_g: torch.Tensor = torch.autograd.functional.jacobian(g_batched, coords, create_graph=True)
        sum_deriv = ( torch.einsum('mk...l->...mkl', d_g)
                    + torch.einsum('ml...k->...mkl', d_g)
                    - torch.einsum('kl...m->...mkl', d_g))
        
        g_inverse = torch.inverse(self.g(coords))
        return 0.5 * torch.einsum('...im, ...mkl->...ikl', g_inverse, sum_deriv)
    
    def R(self, coords: torch.Tensor) -> torch.Tensor:
        """return the Riemannian curvature tensor at the given exirinsic coordinates"""
        Gamma = self.Gamma(coords)
        Gamma_batched = lambda x: torch.einsum('...ilk->ilk', self.Gamma(x))
        d_Gamma = torch.autograd.functional.jacobian(Gamma_batched, coords, create_graph=True) # ...rvsm

        return (
            torch.einsum('rvs...m->...rsmv', d_Gamma)
          - torch.einsum('rms...v->...rsmv', d_Gamma)
          + torch.einsum('...rml,...lvs->...rsmv', Gamma, Gamma)
          - torch.einsum('...rvl,...lms->...rsmv', Gamma, Gamma)
        )
    
    def Ric(self, coords: torch.Tensor) -> torch.Tensor:
        """return the Ricci tensor at the given exirinsic coordinates"""
        return torch.einsum('...cacb->...ab', self.R(coords))

    def ric(self, coords: torch.Tensor) -> torch.Tensor:
        """return the Ricci scalar at the given exirinsic coordinates"""
        g_inverse = torch.inverse(self.g(coords))
        return torch.einsum('...ij,...ji->...', g_inverse, self.Ric(coords))
    
    def vol(self, coords: torch.Tensor) -> torch.Tensor:
        """return the Riemannian volume form at the given extrinsic coordinates"""
        return torch.sqrt(torch.linalg.det(self.g(coords)))
    
    def log_vol(self, coords: torch.Tensor) -> torch.Tensor:
        """return the logarithm of the Riemannian volume form at the given extrinsic coordinates"""
        return torch.logdet(self.g(coords)) / 2.0
    
    def flat(self, tensor: torch.Tensor, coords: torch.Tensor, dim: int | Iterable[int] = -1) -> torch.Tensor:
        """lower one or multiple indices of a tensor at the specified extrinsic coordinates"""
        if type(dim) == int:
            fibered = tensor.movedim(dim, -1)
            n_extra_dims = tensor.ndim-coords.ndim
            s = alphabet[:n_extra_dims]
            flat = torch.einsum(f'...ij,...{s}j->...{s}i', self.g(coords), fibered)
            return flat.movedim(-1, dim)
        assert isinstance(dim, Iterable), "dim must be either int or sequence of int"
        for d in dim:
            tensor = self.flat(tensor, coords, d)
        return tensor
    
    def sharp(self, tensor: torch.Tensor, coords: torch.Tensor, dim: int | Iterable[int] = -1) -> torch.Tensor:
        """raise one or multiple indices of a tensor at the specified extrinsic coordinates"""
        if type(dim) == int:
            fibered = tensor.movedim(dim, -1)
            n_extra_dims = tensor.ndim-coords.ndim
            *batch_d, d = coords.shape
            sharp: torch.Tensor = torch.linalg.solve(
                self.g(coords).view(*batch_d, *(1,)*n_extra_dims, d, d).expand(*tensor.shape, d),
                fibered
            )
            return sharp.movedim(-1, dim)
        assert isinstance(dim, Iterable), "dim must be either int or sequence of int"
        for d in dim:
            tensor = self.sharp(tensor, coords, d)
        return tensor
        
    # TODO: grad

    
    def optimize_geodesic(self, x1: torch.Tensor, x2: torch.Tensor, n_segments=16, n_iter=10000, lr=1e-3, n_subdiv=0, eps=1e-4, patience=5, device='cuda', verbose=True) -> torch.Tensor:
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
        pbar = trange(n_iter, desc='optimizing geodesic', disable = not verbose)
        for _ in pbar:
            optimizer.zero_grad()
            full_curve = torch.cat((curve[[0]], curve_interior, curve[[-1]]))
            subdiv_curve = subdivide_curve(full_curve, n_subdiv) if n_subdiv else full_curve
            curve_e = self.discrete_curve_kinetic_energy(subdiv_curve)
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
    
    def optimize_Karcher_mean(self, *points: torch.Tensor, n_segments=16, n_iter=10000, lr=1e-3, n_subdiv=0, eps=1e-4, patience=5, device='cuda', verbose=True) -> torch.Tensor:
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
            curves_e = sum((torch.square(self.discrete_curve_length(subdiv_curve)) for subdiv_curve in subdiv_curves), torch.zeros(()))
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
    
    # def exponential_map(self, source: torch.Tensor, direction: torch.Tensor, factor, resolution: int, device='cuda', verbose=True) -> torch.Tensor:
    #     og_device = source.device
    #     x, v = source.to(device), direction.to(device)
    #     curve = [x.clone()]
    #     for _ in trange(factor * resolution, desc='Computing exponential map', disable = not verbose):
    #         v -= torch.einsum('kij, i, j -> k', self.Gamma(x), v, v) / resolution
    #         x += v / resolution
    #         curve.append(x.clone())
    #     return torch.stack(curve).to(og_device)

    def Hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        try:
            v = torch.linalg.solve(self.g(q), p)
        except torch.linalg.LinAlgError:
            raise Exception(f"g is singular at {q=}")
        return 0.5 * torch.einsum('...v,...v->...', p, v)

    def exponential_map(self, source: torch.Tensor, direction: torch.Tensor, factor, resolution: int, omega=1.0, return_curve=True, differentiable=False, device='cuda', verbose=True) -> torch.Tensor:
        """numerically compute the exponential map via the explicit symplectic FANTASY integration scheme of second order, 
        source: http://arxiv.org/abs/2010.02237 \\
        [written with help from Diaaeldin (Dia) Taha]"""
        og_device = source.device
        q1 = source.to(device)
        p1 = torch.einsum("...ij,...j->...i", self.g(q1), direction.to(device))
        q2, p2 = q1.clone(), p1.clone()
        curve = [q1.clone()]
        dt = 1.0 / resolution
        half_dt = 0.5 * dt
        angle = 2.0 * omega * dt
        c = torch.cos(torch.as_tensor(angle, device=q1.device, dtype=q1.dtype))
        s = torch.sin(torch.as_tensor(angle, device=q1.device, dtype=q1.dtype))

        def half_step(q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            dH_dq, dH_dp = torch.autograd.functional.jacobian(
                self.Hamiltonian, (q, p), create_graph=differentiable
            )

            dq = half_dt * dH_dp
            dp = - half_dt * dH_dq
            return dq, dp
        
        def c_step(q1, q2, p1, p2):
            dq = q1 - q2
            dp = p1 - p2
            sq = q1 + q2
            sp = p1 + p2
            return (
                0.5 * (sq + dq * c + dp * s),
                0.5 * (sq - dq * c - dp * s),
                0.5 * (sp + dp * c - dq * s),
                0.5 * (sp - dp * c + dq * s)
            )

        for _ in trange(int(factor * resolution), desc='Computing exponential map', disable = not verbose):
            dq, dp = half_step(q1, p2) # H_A half step
            q2 += dq; p1 += dp

            dq, dp = half_step(q2, p1) # H_B half step
            q1 += dq; p2 += dp
            
            q1, q2, p1, p2 = c_step(q1, q2, p1, p2) # H_C full step

            dq, dp = half_step(q1, p2) # H_B half step
            q1 += dq; p2 += dp

            dq, dp = half_step(q1, p2) # H_A half step
            q2 += dq; p1 += dp

            if return_curve:
                curve.append(q1.clone())


        result = torch.stack(curve) if return_curve else q1
        return result.to(og_device)
    
    def geodesic_deviation(self, curve: torch.Tensor, device='cuda'):
        curve = curve.to(device)
        total_deviation = torch.tensor(0., device=device)
        for x1, x2, x3 in curve.unfold(dimension=0, size=3, step=1).swapdims(-2, -1):
            dx1, dx2 = x2 - x1, x3 - x2
            d2x = dx2 - dx1
            a = d2x + torch.einsum('kij, i, j -> k', self.Gamma(x2), dx1, dx1)
            a_norm = torch.einsum('mv, m, v ->', self.g(x2), a, a)
            total_deviation += a_norm
        return total_deviation

    
class PullbackRiemannianManifold(RiemannianManifold):
    @abstractmethod
    def coordinate_map(self, coords: torch.Tensor) -> torch.Tensor:
        pass

    def g(self, coords: torch.Tensor) -> torch.Tensor:
        Phi_batched = lambda x: torch.einsum('...i->i', self.coordinate_map(x))
        d_Phi = torch.autograd.functional.jacobian(Phi_batched, coords, create_graph=True)
        return torch.einsum('k...m,k...v->...mv', d_Phi, d_Phi)
    
    def discrete_curve_length(self, curve: torch.Tensor) -> torch.Tensor:
        curve_Phi = self.coordinate_map(curve)
        return torch.square((curve_Phi[1:] - curve_Phi[:-1])).sum(1).sqrt().sum()

    def discrete_curve_kinetic_energy(self, curve: torch.Tensor) -> torch.Tensor:
        curve_Phi = self.coordinate_map(curve)
        return torch.square((curve_Phi[1:] - curve_Phi[:-1])).sum()

@dataclass
class NestedPullbackManifold(PullbackRiemannianManifold):
    source_manifold: PullbackRiemannianManifold
    coordinate_transform: Callable[[torch.Tensor], torch.Tensor]

    def coordinate_map(self, coords: torch.Tensor) -> torch.Tensor:
        return self.source_manifold.coordinate_map(self.coordinate_transform(coords))

@dataclass
class EuclidianPullbackmanifold(PullbackRiemannianManifold):
    coordinate_transform: Callable[[torch.Tensor], torch.Tensor]
    def coordinate_map(self, coords: torch.Tensor) -> torch.Tensor:
        return self.coordinate_transform(coords)