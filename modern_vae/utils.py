import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

def sample_differentiable(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    stdev = torch.exp(logvar * 0.5)
    return mu + stdev * torch.randn_like(mu)

ssim = StructuralSimilarityIndexMeasure()

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(
        torch.square(mu) + logvar.exp() - logvar - 1, dim=-1
    )

def vae_loss(
        input: torch.Tensor, 
        output: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        lambda_rec: float = 1.0,
        lambda_ssim: float = 0.2,
        lambda_kl: float = 1e-3,
        return_components=False
):
    reconstruction_loss = F.l1_loss(input, output)
    ssim_loss: torch.Tensor = 1 - ssim(input, output)
    kl_loss = kl_divergence(mu, logvar).mean()

    if return_components:
        return lambda_rec * reconstruction_loss + lambda_ssim * ssim_loss + lambda_kl * kl_loss, (lambda_rec * reconstruction_loss.item()), (lambda_ssim * ssim_loss.item()), (lambda_kl * kl_loss.item())
    else:
        return lambda_rec * reconstruction_loss + lambda_ssim * ssim_loss + lambda_kl * kl_loss
