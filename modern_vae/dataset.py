import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset

transform = v2.Compose([
    v2.Resize((128, 128), interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

celeba = torchvision.datasets.CelebA(root='../datasets', split='all', transform=transform, download=False)
LATENTS_PATH = '/usr/people/robertwiebe/ml_geometry_project/datasets/celeba_embeddings_normalized.pt'
X, mean, std = torch.load(LATENTS_PATH).values()
latents = torch.utils.data.TensorDataset(X)
mean = mean[..., None, None].cuda()
std = std[..., None, None].cuda()

dataset = torch.utils.data.StackDataset(celeba, latents)
# dataset = Subset(dataset, range(1000)) # temporary experiment
dataloader = DataLoader(dataset, batch_size=24, shuffle=True, pin_memory=True, num_workers=4)
dataloader_latents_only = DataLoader(latents, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)