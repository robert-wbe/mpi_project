import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from tqdm import trange, tqdm
from sklearn.mixture import GaussianMixture

LATENTS_PATH = 'datasets/celeba_deep_embeddings_w_y.pt'

X, y = torch.load(LATENTS_PATH)
dataset = TensorDataset(X, y)
# indices = torch.load('pickled_objects/top_indices_8k.pt')
# X: torch.Tensor = X[indices]
# y: torch.Tensor = y[indices]

attribute_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

N_COMPONENTS = 4
gmms = []

supweights = (y.sum(0) / y.sum()).cuda()

for i, attr_name in enumerate(tqdm(attribute_names, desc='Fitting atribute GMMS')):
    digit_samples = X[[y[:, i].bool()]].numpy()
    new_gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='diag', verbose=True, init_params='k-means++', max_iter=10000).fit(digit_samples)
    gmms.append(new_gmm)

# SAVE

subweights = torch.stack([
    torch.as_tensor(
        gmm.weights_, dtype=torch.float32, device='cuda'
    ) for gmm in gmms
])

means = torch.stack([
    torch.as_tensor(
        gmm.means_, dtype=torch.float32, device='cuda'
    ) for gmm in gmms
])

variances = torch.stack([
    torch.as_tensor(
        gmm.covariances_, dtype=torch.float32, device='cuda'
    ) for gmm in gmms
])

torch.save({
    'supweights' : supweights,
    'subweights': subweights,
    'means': means,
    'variances': variances,
}, f'trained_checkpoints/celeba_nested_nested_gmm_4_v2.pt')