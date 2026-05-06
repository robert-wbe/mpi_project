import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from tqdm import trange, tqdm
from pretrained_models import StableDiffusionVAE
from sklearn.mixture import GaussianMixture

transform = v2.Compose([
    v2.Resize((128, 128), interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

celeba = torchvision.datasets.CelebA(root='datasets', split='all', transform=transform, download=False)
dataloader = DataLoader(celeba, batch_size=64, shuffle=False, num_workers=4)
vae = StableDiffusionVAE()

attribute_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
gmms = []
for i, attr_name in tqdm(enumerate(attribute_names), desc='Fitting atribute GMMS', position=0):
    digit_samples = np.empty((0, 1024))
    with torch.no_grad():
        for images, class_labels in tqdm(dataloader, desc=attr_name, position=1, leave=False):
            if torch.any(class_labels[:, i]):
                digit_samples = np.concatenate((digit_samples, vae.encode(images[class_labels[:, i].bool()]).numpy()))
    new_gmm = GaussianMixture(n_components=16, covariance_type='diag',verbose=True, init_params='k-means++', max_iter=10000).fit(digit_samples)
    gmms.append(new_gmm)

torch.save(gmms, 'trained_checkpoints/celeba_stablediff_nested_gmm.pt')