import torch
from pretrained_models import ModernLatentVAE
from torch.utils.data import DataLoader
from tqdm import tqdm

LATENTS_PATH = 'datasets/celeba_embeddings_normalized_w_y.pt'
SAVE_PATH = 'datasets/celeba_deep_embeddings_w_y'

X, y = torch.load(LATENTS_PATH)
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=8)

latent_vae = ModernLatentVAE()


embedded_batches = []
with torch.no_grad():
    for images, _ in tqdm(dataloader, desc='Computing and saving deep embeddings'):
        embedded_batches.append(latent_vae.encode(images))

deep_embeddings = torch.cat(embedded_batches)

torch.save((deep_embeddings, y), SAVE_PATH)