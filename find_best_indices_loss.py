import torch
import torchvision
from torch.nn.functional import cross_entropy, l1_loss, mse_loss
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import trange, tqdm
from geometry_tools import lerp
from utils import NestedDiagonalGMM
from pretrained_models import NestedVAE
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

TOP_N = 200
lambda_rec = 1.0
lambda_ssim = 0.2
lambda_brightness = 0.25

transform = v2.Compose([
    v2.Resize((128, 128), interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

top_indices = torch.load('pickled_objects/top_indices_8k.pt')
celeba = torchvision.datasets.CelebA(root='datasets', split='all', transform=transform, download=False)
celeba = Subset(celeba, top_indices)
celeba_loader = DataLoader(celeba, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=8)

vae = NestedVAE()
ssim = SSIM(reduction='none').eval().cuda()

batch_results = []
with torch.inference_mode():
    for images, _ in tqdm(celeba_loader, desc='Computing best fit indices'):
        images: torch.Tensor
        images = images.cuda()
        
        reconstructed = vae.reconstruct(images)
        loss = (
            lambda_rec * l1_loss(reconstructed, images, reduction='none').mean((1, 2, 3))
          + lambda_ssim * ssim(reconstructed, images)
          - lambda_brightness * (reconstructed.mean((1, 2, 3)) + reconstructed.std((1, 2, 3)))
        )
        batch_results.append(loss.cpu())


results = torch.cat(batch_results)
assert len(results) == len(celeba)
top_n_indices = results.sort().indices[:TOP_N]
top_n_indices = top_indices[top_n_indices]

torch.save(top_n_indices, 'pickled_objects/top_indices_loss.pt')

# print(results.min())