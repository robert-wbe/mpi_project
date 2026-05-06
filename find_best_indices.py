import torch
import torchvision
from torch.nn.functional import cross_entropy
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import trange, tqdm
from geometry_tools import lerp
from utils import NestedDiagonalGMM

TOP_N = 8000

X, y = torch.load('datasets/celeba_deep_embeddings_w_y.pt')
bright_indices = torch.load('pickled_objects/bright_indices.pt')
celeba = Subset(TensorDataset(X, y), bright_indices)
celeba_loader = DataLoader(celeba, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)

nested_gmm_params = torch.load('trained_checkpoints/celeba_nested_nested_gmm_4.pt')
nested_gmm = NestedDiagonalGMM(
    means=nested_gmm_params['means'],
    variances=nested_gmm_params['variances'],
    subweights=nested_gmm_params['subweights'],
    supweights=nested_gmm_params['supweights'],
)


batch_results = []
for embeds, labels in tqdm(celeba_loader, desc='Computing best fit indices'):
    embeds: torch.Tensor; labels: torch.Tensor
    embeds = embeds.cuda()
    labels = labels.cuda()

    gmm_posterior = nested_gmm.posterior(embeds)
    target_dist = labels / labels.sum(-1, keepdim=True)
    # batch_results.append(torch.sum(gmm_posterior * labels, dim=-1).cpu())
    batch_results.append(cross_entropy(gmm_posterior, target_dist, reduction='none').cpu())


results = torch.cat(batch_results)
assert len(results) == len(celeba)
top_n_indices = results.sort().indices[:TOP_N]
top_n_indices = bright_indices[top_n_indices]

torch.save(top_n_indices, 'pickled_objects/top_indices_8k.pt')

# print(results.min())