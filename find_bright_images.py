import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pretrained_models import NestedVAE

BRIGHTNESS_THRESHOLD = 0.5
BATCH_SIZE = 64

X, y = torch.load('datasets/celeba_deep_embeddings_w_y.pt')
celeba = TensorDataset(X, y)
celeba_loader = DataLoader(celeba, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=8)

vae = NestedVAE()

batch_results = []
for batch_idx, (embeds, _) in tqdm(enumerate(celeba_loader), desc='Finding bright images'):
    with torch.no_grad():
        decoded = vae.decode(embeds.cuda())
    batch_results.append(torch.argwhere(decoded.mean(dim=(1, 2, 3)) >= 0.5)[:, 0].cpu() + batch_idx * BATCH_SIZE)

results = torch.cat(batch_results)
print(f'Filtered {len(results)} / {len(celeba)} images')

torch.save(results, 'pickled_objects/bright_indices')