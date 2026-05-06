import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from tqdm import trange, tqdm
from sklearn.mixture import GaussianMixture
from pretrained_models import ModernLatentVAE

LATENTS_PATH = 'datasets/celeba_embeddings_normalized_w_y.pt'

X, y = torch.load(LATENTS_PATH)
dataset = TensorDataset(X, y)
bright_indices = torch.load('pickled_objects/bright_indices.pt')
dataloader = DataLoader(Subset(dataset, bright_indices), batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

latent_vae = ModernLatentVAE()

attribute_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

N_COMPONENTS = [4]
gmms = [[] for _ in N_COMPONENTS]

for i, attr_name in tqdm(enumerate(attribute_names), desc='Fitting atribute GMMS', position=0):
    digit_samples = np.empty((0, 64))
    with torch.no_grad():
        for images, class_labels in tqdm(dataloader, desc=attr_name, position=1, leave=False):
            if torch.any(class_labels[:, i]):
                digit_samples = np.concatenate((digit_samples, latent_vae.encode(images[class_labels[:, i].bool()]).numpy()))
    for collector, n_comps in zip(gmms, N_COMPONENTS):
        new_gmm = GaussianMixture(n_components=n_comps, covariance_type='diag',verbose=True, init_params='k-means++', max_iter=10000).fit(digit_samples)
        collector.append(new_gmm)

# SAVE
supweights = torch.load('trained_checkpoints/celeba_stablediff_nested_gmm.pt')['supweights']

for collector, n_comps in zip(gmms, N_COMPONENTS):

    subweights = torch.stack([
        torch.as_tensor(
            gmm.weights_, dtype=torch.float32, device='cuda'
        ) for gmm in collector
    ])

    means = torch.stack([
        torch.as_tensor(
            gmm.means_, dtype=torch.float32, device='cuda'
        ) for gmm in collector
    ])

    variances = torch.stack([
        torch.as_tensor(
            gmm.covariances_, dtype=torch.float32, device='cuda'
        ) for gmm in collector
    ])

    torch.save({
        'supweights' : supweights,
        'subweights': subweights,
        'means': means,
        'variances': variances,
    }, f'trained_checkpoints/celeba_nested_nested_gmm_{n_comps}.pt')