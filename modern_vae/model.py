import torch
import torch.nn as nn

def sample_differentiable(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    stdev = torch.exp(logvar * 0.5)
    return mu + stdev * torch.randn_like(mu)

class ResNetBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=n_channels),
            nn.SiLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same'),

            nn.GroupNorm(num_groups=8, num_channels=n_channels),
            nn.SiLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)

class Encoder(nn.Module):
    def __init__(self, latent_features: int, in_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding='same'),
            ResNetBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            ResNetBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ResNetBlock(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 2*latent_features, kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = torch.chunk(self.layers(x).squeeze(-2, -1), 2, dim=-1)
        return mu, logvar
    
class ResNetBlockTranspose(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=n_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=3, padding=1),

            nn.GroupNorm(num_groups=8, num_channels=n_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)

class Decoder(nn.Module):
    def __init__(self, latent_features: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_features, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            ResNetBlockTranspose(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            ResNetBlockTranspose(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            ResNetBlockTranspose(32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x[..., None, None])

class VAE(nn.Module):
    def __init__(self, latent_features: int, img_channels: int):
        super().__init__()
        self.encoder = Encoder(latent_features, img_channels)
        self.decoder = Decoder(latent_features, img_channels)
    
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = sample_differentiable(mu, logvar)
        reconstr = self.decoder(z)
        return reconstr, mu, logvar

    def encode(self, x: torch.Tensor, sample_posterior=False) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        if sample_posterior:
            return sample_differentiable(mu, logvar)
        else:
            return mu

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
    def reconstruct(self, x: torch.Tensor):
        return self.decode(self.encode(x))
    
    @staticmethod
    def load_from(f: str):
        return NotImplementedError() #TODO