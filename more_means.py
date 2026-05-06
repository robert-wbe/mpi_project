import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import trange, tqdm
from geometry_tools import lerp, slerp_curve
from utils import NestedDiagonalGMM
from posterior_manifold import FisherLogPosteriorManifold
from latent_attribute_prediction.model import MLP_ELU_convex
from attribute_prediction_model_new.model import model as attribute_model_new
from conformal_ebm import optimize_ebm_mean as optimize_ebm_mean1
from conformal_ebm_new import optimize_ebm_mean as optimize_ebm_mean2
from decoder_geometry import DecoderManifold
from pretrained_models import NestedVAE
from typing import NamedTuple
from riemannian_geometry import EuclidianPullbackmanifold

X, y = torch.load('datasets/celeba_deep_embeddings_w_y.pt')
best_fit_indices = torch.load('pickled_objects/top_indices_loss.pt')
celeba = TensorDataset(X, y)
celeba_subset = Subset(celeba, best_fit_indices)
celeba_loader = DataLoader(celeba_subset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)

vae = NestedVAE()
nested_gmm_params = torch.load('trained_checkpoints/celeba_nested_nested_gmm_4.pt')
nested_gmm = NestedDiagonalGMM(
    means=nested_gmm_params['means'],
    variances=nested_gmm_params['variances'],
    subweights=nested_gmm_params['subweights'],
    supweights=nested_gmm_params['supweights'],
)

M_Dec = DecoderManifold(vae.decode)

attribute_model = MLP_ELU_convex()
attribute_model.load_state_dict(torch.load('latent_attribute_prediction/checkpoints/best_checkpoint.pt')['model_state_dict'])
attribute_model = attribute_model.cuda()

attribute_model_new.load_state_dict(torch.load('attribute_prediction_model_new/checkpoints/best_checkpoint.pt')['model_state_dict'])
attribute_model_new = attribute_model_new.cuda()

M_attr_latent = EuclidianPullbackmanifold(lambda x: attribute_model(x))
M_attr_outer = EuclidianPullbackmanifold(lambda x: attribute_model_new(vae.decode(x)))

class MeansGroup2(NamedTuple):
    dec_mean: torch.Tensor
    ebm_mean1: torch.Tensor
    ebm_mean2: torch.Tensor
    mean_latent: torch.Tensor
    mean_outer: torch.Tensor

means = []
for (images, _) in tqdm(celeba_loader, desc='Computing means'):
    images: torch.Tensor
    dec_mean = M_Dec.optimize_Karcher_mean(*images, n_segments=8, n_iter=100).cpu()

    ebm_mean1 = optimize_ebm_mean1(*images, log_likelihood=nested_gmm.log_prob, n_segments=8, n_subdiv=3, n_iter=1000).cpu()
    ebm_mean2 = optimize_ebm_mean2(*images, log_likelihood=nested_gmm.log_prob, n_segments=8, n_subdiv=3, n_iter=1000).cpu()

    mean_latent = M_attr_latent.optimize_Karcher_mean(*images, n_segments=8, n_iter=10000, n_subdiv=3)
    mean_outer = M_attr_outer.optimize_Karcher_mean(*images, n_segments=8, n_iter=100, lr=0.1)

    means.append(MeansGroup2(dec_mean, ebm_mean1, ebm_mean2, mean_latent, mean_outer))

torch.save(means, 'pickled_objects/more_means.pt')