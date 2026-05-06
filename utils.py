import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import VisionDataset
import torch.distributions as D

def categorical_entropy(probs: torch.Tensor, dim=-1):
    return -torch.sum(probs * torch.log(probs), dim=dim)

def extended_batching(func):
    def new_func(x: torch.Tensor) -> torch.Tensor:
        return func(x.reshape(-1, x.shape[-1])).reshape_as(x)
    return new_func

def get_random_images(dataset: VisionDataset, n=3):
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    imgs, _ = next(iter(loader))
    return imgs

def get_random_images_w_labels(dataset: VisionDataset, n=3):
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    imgs, labels = next(iter(loader))
    return imgs, labels

def get_random_image(dataset: VisionDataset):
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    imgs, _ = next(iter(loader))
    return imgs[0]

def get_random_image_of_class(dataset: VisionDataset, label: int):
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    itr = iter(loader)
    while True:
        imgs, lb = next(itr)
        if torch.any(lb == label):
            return imgs[lb == label][0]

def get_random_images_of_classes(dataset: VisionDataset, labels: list[int]):
    return torch.stack([get_random_image_of_class(dataset, lb) for lb in labels])

class NestedDiagonalGMM(torch.distributions.Distribution):
    def __init__(self, means: torch.Tensor, variances: torch.Tensor, supweights: torch.Tensor | None = None, suplogits: torch.Tensor | None = None, subweights: torch.Tensor | None = None, sublogits: torch.Tensor | None = None):
        if not suplogits:
            assert supweights is not None, "Must pass either weights or logits"
            suplogits = torch.log(supweights / supweights.sum())
        if not sublogits:
            assert subweights is not None, "Must pass either weights or logits"
            sublogits = torch.log(subweights / subweights.sum(dim=-1, keepdim=True))
        self.dist = D.MixtureSameFamily(
            D.Categorical(logits=suplogits, validate_args=True),
            D.MixtureSameFamily(
                D.Categorical(logits=sublogits, validate_args=True),
                D.Independent(D.Normal(means, variances, validate_args=True), 1, validate_args=True),
                validate_args=True
            )
        , validate_args=True)
        super().__init__(batch_shape=self.dist.batch_shape, event_shape=self.dist.event_shape, validate_args=False)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(value)
    
    def log_posterior(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(
            self.dist.component_distribution.log_prob(value[..., None, :])
            + self.dist.mixture_distribution.logits,
            dim=-1
        )

    def posterior(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(
            self.dist.component_distribution.log_prob(value[..., None, :])
            + self.dist.mixture_distribution.logits,
            dim=-1            
        )

