import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import trange, tqdm
from utils import NestedDiagonalGMM
from posterior_manifold import FisherLogPosteriorManifold
from pretrained_models import NestedVAE
from typing import NamedTuple

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
M = FisherLogPosteriorManifold(nested_gmm.log_posterior)

class MeanGroup(NamedTuple):
    embeds: torch.Tensor
    eucl_mean: torch.Tensor
    karcher_mean: torch.Tensor
    labels: torch.Tensor

means = []
for embeds, labels in tqdm(celeba_loader, desc='Computing means'):
    embeds: torch.Tensor; labels: torch.Tensor
    embeds_cuda = embeds.cuda()

    eucl_mean = embeds.mean(0)
    karcher_mean = M.optimize_Karcher_mean(*embeds_cuda, n_segments=8, n_iter=5000, n_subdiv=3).cpu()

    means.append(MeanGroup(embeds, eucl_mean, karcher_mean, labels))

torch.save(means, 'pickled_objects/means.pt')
