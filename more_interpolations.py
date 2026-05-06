import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import trange, tqdm
from geometry_tools import lerp, slerp_curve
from utils import NestedDiagonalGMM
from posterior_manifold import FisherLogPosteriorManifold
from conformal_ebm_new import optimize_ebm_geodesic
from decoder_geometry import DecoderManifold
from pretrained_models import NestedVAE
from typing import NamedTuple

X, y = torch.load('datasets/celeba_deep_embeddings_w_y.pt')
best_fit_indices = torch.load('pickled_objects/top_indices_loss.pt')
celeba = TensorDataset(X, y)
celeba_subset = Subset(celeba, best_fit_indices)
celeba_loader = DataLoader(celeba_subset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)

vae = NestedVAE()
nested_gmm_params = torch.load('trained_checkpoints/celeba_nested_nested_gmm_4.pt')
nested_gmm = NestedDiagonalGMM(
    means=nested_gmm_params['means'],
    variances=nested_gmm_params['variances'],
    subweights=nested_gmm_params['subweights'],
    supweights=nested_gmm_params['supweights'],
)
M = FisherLogPosteriorManifold(nested_gmm.log_posterior)
M_Dec = DecoderManifold(vae.decode)

class InterpolationGroupExtra(NamedTuple):
    slinline: torch.Tensor
    ebm_geodesic_2: torch.Tensor

interpolations = []
for (images, labels) in tqdm(celeba_loader, desc='Computing interpolations'):
    images: torch.Tensor; labels: torch.Tensor
    images_cuda = images.cuda()

    slinline = slerp_curve(*images, n=8)
    ebm_geodesic_2 = optimize_ebm_geodesic(*images_cuda, log_likelihood=nested_gmm.log_prob, n_segments=8, n_subdiv=3, n_iter=1000).cpu()

    interpolations.append(InterpolationGroupExtra(slinline, ebm_geodesic_2))

torch.save(interpolations, 'pickled_objects/interpolations_extra.pt')