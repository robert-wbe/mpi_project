import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from repos.torch_vae.model import VAE as PRE_MNIST_VAE
import numpy as np
from repos.torch_vae_2.model import VAE as VAE_2
from modern_vae.model import VAE as ModernVAE

class MNIST_VAE:
    def __init__(self, n_latent: int, device='cuda'):
        self.model = PRE_MNIST_VAE(num_latent_dims=n_latent, num_img_channels=1, max_num_filters=128, device=device)
        self.device = device
        self.model.load(f'repos/torch_vae/models/mnist/vae_filters_0128_dims_{n_latent:04d}.pth')
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.model.encode(data)
        if is_single_image:
            latent = latent[0]
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        latent = latent.to(self.device)
        is_single_image = latent.ndim == 1
        if is_single_image:
            latent = latent.unsqueeze(0)
        data = self.model.decode(latent)
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)

def MNIST_VAE_DIM2(device='cuda') -> nn.Module:
    model = PRE_MNIST_VAE(num_latent_dims=2, num_img_channels=1, max_num_filters=128, device=device)
    model.load('repos/torch_vae/models/mnist/vae_filters_0128_dims_0002.pth')
    model.to(device)
    model.eval()
    return model

# def MNIST_VAE(n_latent: int, device='cuda') -> PRE_MNIST_VAE:
#     model = PRE_MNIST_VAE(num_latent_dims=n_latent, num_img_channels=1, max_num_filters=128, device=device)
#     model.load(f'repos/torch_vae/models/mnist/vae_filters_0128_dims_{n_latent:04d}.pth')
#     model.to(device)
#     model.eval()
#     return model

class FashionMNIST_VAE:
    def __init__(self, n_latent: int, device='cuda'):
        self.model = PRE_MNIST_VAE(num_latent_dims=n_latent, num_img_channels=1, max_num_filters=128, device=device)
        self.device = device
        self.model.load(f'repos/torch_vae/models/fashion-mnist/vae_filters_0128_dims_{n_latent:04d}.pth')
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.model.encode(data)
        if is_single_image:
            latent = latent[0]
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        latent = latent.to(self.device)
        is_single_image = latent.ndim == 1
        if is_single_image:
            latent = latent.unsqueeze(0)
        data = self.model.decode(latent)
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)
    
class CelebAVAE:
    def __init__(self, n_latent: int, device='cuda'):
        self.model = PRE_MNIST_VAE(num_latent_dims=n_latent, num_img_channels=3, max_num_filters=128, device=device)
        self.device = device
        self.model.load(f'repos/torch_vae/models/celeb-a/vae_filters_0128_dims_{n_latent:04d}.pth')
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.model.encode(data)
        if is_single_image:
            latent = latent[0]
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        latent = latent.to(self.device)
        is_single_image = latent.ndim == 1
        if is_single_image:
            latent = latent.unsqueeze(0)
        data = self.model.decode(latent)
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)

class CelebALatentVAE:
    def __init__(self, n_latent: int, device='cuda'):
        self.model = VAE_2(num_latent_dims=n_latent, num_img_channels=4, max_num_filters=256, device=device, img_size=16)
        self.device = device
        self.model.load(f'repos/torch_vae_2/models/celeba-latent/vae_filters_0256_dims_{n_latent:04d}_299.pth')
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.model.encode(data)
        if is_single_image:
            latent = latent[0]
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        latent = latent.to(self.device)
        is_single_image = latent.ndim == 1
        if is_single_image:
            latent = latent.unsqueeze(0)
        data = self.model.decode(latent)
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)

class CIFAR100VAE:
    def __init__(self, n_latent: int, device='cuda'):
        self.model = PRE_MNIST_VAE(num_latent_dims=n_latent, num_img_channels=3, max_num_filters=128, device=device)
        self.device = device
        self.model.load(f'repos/torch_vae/models/cifar-100/vae_filters_0128_dims_{n_latent:04d}.pth')
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.model.encode(data)
        if is_single_image:
            latent = latent[0]
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        latent = latent.to(self.device)
        is_single_image = latent.ndim == 1
        if is_single_image:
            latent = latent.unsqueeze(0)
        data = self.model.decode(latent)
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)

class CIFAR10VAE:
    def __init__(self, n_latent: int, device='cuda'):
        self.model = PRE_MNIST_VAE(num_latent_dims=n_latent, num_img_channels=3, max_num_filters=128, device=device)
        self.device = device
        self.model.load(f'repos/torch_vae/models/cifar-10/vae_filters_0128_dims_{n_latent:04d}.pth')
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.model.encode(data)
        if is_single_image:
            latent = latent[0]
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        latent = latent.to(self.device)
        is_single_image = latent.ndim == 1
        if is_single_image:
            latent = latent.unsqueeze(0)
        data = self.model.decode(latent)
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)

class StableDiffusionVAE:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device) #type:ignore
        self.model.eval()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.model.encode(data)[0].mode()
        if is_single_image:
            latent = latent[0]
        # return latent.flatten(start_dim=-3).to(og_device)
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        # width = int(np.sqrt(latent.shape[-1] // 4))
        is_single_image = latent.ndim == 3
        # latent = latent.reshape(-1, 4, width, width).to(self.device)
        if is_single_image:
            latent = latent.unsqueeze(0)
        latent = latent.to(self.device)
        data = self.model.decode(latent)[0].clamp(0, 1) #type:ignore
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)

class ModernLatentVAE:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = ModernVAE(latent_features=64, img_channels=4)
        self.model.load_state_dict(torch.load('modern_vae/checkpoints/continued/latent_vae_epoch_150.pt')['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.model.encode(data)
        if is_single_image:
            latent = latent[0]
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        latent = latent.to(self.device)
        is_single_image = latent.ndim == 1
        if is_single_image:
            latent = latent.unsqueeze(0)
        data = self.model.decode(latent)
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.to(self.device)
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)
    
class NestedVAE:
    def __init__(self):
        self.outer = StableDiffusionVAE()
        self.inner = ModernLatentVAE()
        _, mean, std = torch.load('datasets/celeba_embeddings_normalized.pt').values()
        self.mean = mean[..., None, None].cuda()
        self.std = std[..., None, None].cuda()
        self.denorm = lambda x: x * self.std + self.mean
        self.norm = lambda x: (x - self.mean) / self.std

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.cuda()
        is_single_image = data.ndim == 3
        if is_single_image:
            data = data.unsqueeze(0)
        latent = self.inner.encode(self.norm(self.outer.encode(data)))
        if is_single_image:
            latent = latent[0]
        return latent.to(og_device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        og_device = latent.device
        latent = latent.cuda()
        is_single_image = latent.ndim == 1
        if is_single_image:
            latent = latent.unsqueeze(0)
        data = self.outer.decode(self.denorm(self.inner.decode(latent)))
        if is_single_image:
            data = data[0]
        return data.to(og_device)
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        og_device = data.device
        data = data.cuda()
        reconstructed = self.decode(self.encode(data))
        return reconstructed.to(og_device)