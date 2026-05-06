import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import trange, tqdm
from geometry_tools import lerp, slerp_curve
from utils import NestedDiagonalGMM
from posterior_manifold import FisherLogPosteriorManifold
from riemannian_geometry import EuclidianPullbackmanifold
from latent_attribute_prediction.model import MLP_ELU_convex
from attribute_prediction_model_new.model import model as attribute_model_new
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

attribute_model = MLP_ELU_convex()
attribute_model.load_state_dict(torch.load('latent_attribute_prediction/checkpoints/best_checkpoint.pt')['model_state_dict'])
attribute_model = attribute_model.cuda()

attribute_model_new.load_state_dict(torch.load('attribute_prediction_model_new/checkpoints/best_checkpoint.pt')['model_state_dict'])
attribute_model_new = attribute_model_new.cuda()

M_attr_latent = EuclidianPullbackmanifold(lambda x: attribute_model(x))
M_attr_outer = EuclidianPullbackmanifold(lambda x: attribute_model_new(vae.decode(x)))

class InterpolationGroupExtra2(NamedTuple):
    interp_latent: torch.Tensor
    interp_outer: torch.Tensor

interpolations = []
for (images, labels) in tqdm(celeba_loader, desc='Computing interpolations'):
    images: torch.Tensor; labels: torch.Tensor
    images_cuda = images.cuda()

    interp_latent = M_attr_latent.optimize_geodesic(*images, n_segments=8, n_iter=10000, n_subdiv=3)
    interp_outer = M_attr_outer.optimize_geodesic(*images, n_segments=8, n_iter=100, n_subdiv=3, lr=0.1)

    interpolations.append(InterpolationGroupExtra2(interp_latent, interp_outer))

torch.save(interpolations, 'pickled_objects/interpolations_extra2.pt')