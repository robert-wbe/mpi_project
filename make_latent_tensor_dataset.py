import torch
import torchvision
from torchvision.transforms import v2
from pretrained_models import StableDiffusionVAE
from torch.utils.data import DataLoader
from tqdm import tqdm

IMG_SIZE = 128

transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

celeba = torchvision.datasets.CelebA(root='datasets', split='all', transform=transform, download=False)
celeba_loader = DataLoader(celeba, batch_size=64, shuffle=False, num_workers=4)

vae = StableDiffusionVAE()
embeddings = torch.empty(0, 4*(IMG_SIZE//8)**2)
labels = torch.empty(0, 40)

for X, y in tqdm(celeba_loader, desc='Computing CelebA embeddings'):
    with torch.no_grad():
        latent = vae.encode(X)
        embeddings = torch.cat((embeddings, latent))
        del latent
    labels = torch.cat((labels, y))
    torch.cuda.empty_cache()

torch.save((embeddings, labels), 'datasets/celeba_embeddings.pt')