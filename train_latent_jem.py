import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
import argparse
from models.latent_JEM import MLP_ELU_convex
from pretrained_models import MNIST_VAE, PRE_MNIST_VAE
from tqdm import tqdm
from pathlib import Path

BATCH_SIZE = 64
SGLD_STEPS = 100
SGLD_STD = 1e-2
SGLD_LR = 1e-2

def get_vae(dataset: str, vae_dim: int) -> PRE_MNIST_VAE:
    match dataset:
        case "MNIST":
            return MNIST_VAE(vae_dim)
        case _: raise NotImplementedError()

def get_dataset(dataset_name: str, transform) -> VisionDataset:
    match dataset_name:
        case "MNIST":
            return torchvision.datasets.MNIST(root='datasets', transform=transform, download=True, train=True)
        case _: raise NotImplementedError()

dataset_n_classes = {
    "MNIST": 10,
}

def get_model(architecture: str, n_in: int, n_out: int) -> MLP_ELU_convex:
    match architecture:
        case "mlp_elu":
            return MLP_ELU_convex(n_dim=n_in, n_feat=8, n_classes=n_out).cuda()
        case _: raise NotImplementedError()

def sgld(en, x_i, n_steps=20, sgld_lr=100.0, sgld_std=1e-2):
    x_s = x_i.clone()
    #x_s = 5*torch.randn_like(x) + torch.tensor([0, 5]).unsqueeze(0).to(x.device)
    x_s.requires_grad_(True)
    for i in range(n_steps):
        e = en(x_s)
        grad_x = torch.autograd.grad(
            outputs=e, 
            inputs=x_s,
            grad_outputs=torch.ones_like(e),  # same shape as f
            create_graph=True,
            retain_graph=True
            )[0]
        
        x_s.data = x_s.data - sgld_lr * grad_x + sgld_std * torch.randn_like(x_s)
    final_samples = x_s.detach()
    return final_samples

def main(args):
    transform = v2.Compose([
        v2.Resize((args.image_size, args.image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    dataset = get_dataset(args.dataset, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    vae = get_vae(args.dataset, args.vae_dim)
    vae.eval()
    jem = get_model(args.architecture, args.vae_dim, dataset_n_classes[args.dataset])

    optimizer = torch.optim.Adam(jem.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=1e-6)
    cce_loss = nn.CrossEntropyLoss()

    best_loss = np.inf
    save_path = Path(args.save_root) / f'{args.model_name}.pt'

    for epoch in range(args.n_epochs):
        jem.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.n_epochs}')
        losses = []
        for imgs, labels in pbar:
            with torch.no_grad():
                latents = vae.encode(imgs.cuda())
                labels = labels.cuda()
                # latents = torch.stack((
                #     torch.cos(labels / 10),
                #     torch.sin(labels / 10)
                # ), dim=-1)

            L: torch.Tensor = torch.tensor(0.0, device='cuda')

            # 1. Categorical Cross Entropy (CCE) loss
            logits = jem(latents)
            L += args.cce_weight * cce_loss(logits, labels)

            # 2. Denoising Score Matching (DSM) loss
            if args.dsm_weight > 0.0:
                sigma = (torch.randint(1, 200, (latents.shape[0],)).float()/100.0).unsqueeze(1).cuda()
                latents_noisy = latents + torch.randn_like(latents) * sigma
                latents_noisy.requires_grad_(True)
                score_estimate = -torch.autograd.grad(jem.energy(latents_noisy).sum(), latents_noisy, create_graph=True)[0]
                score_target = (latents - latents_noisy) / sigma**2

                L += args.dsm_weight * ((score_estimate - score_target) ** 2).sum(dim=-1).mean()

            # 3. Contrastive Divergence (CD) loss
            if args.cd_weight > 0.0:
                latent_i = torch.randn_like(latents)
                latent_s = sgld(jem.energy, latent_i, n_steps=SGLD_STEPS, sgld_lr=SGLD_LR, sgld_std=SGLD_STD)
                fp_all = jem.energy(latents)
                fq_all = jem.energy(latent_s)
                if args.reg_weight > 0.0:
                    L += args.reg_weight * ((fp_all ** 2).mean() + (fq_all ** 2).mean())

                L += args.cd_weight * (fp_all.mean() - fq_all.mean())

            optimizer.zero_grad()
            L.backward()
            # torch.nn.utils.clip_grad_norm_(jem.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=L.item())

            losses.append(L.item())
            if L.item() < best_loss:
                # save model
                torch.save(jem.state_dict(), save_path)
        
        pbar.set_postfix(loss=sum(losses)/len(losses))




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Latent JEMs and stuff")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "cifar10"])
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
    # parser.add_argument("--vae_path", type=str, default="repos/torch_vae/models/mnist/vae_filters_0128_dims_0002.pth")
    
    ## MODEL args
    parser.add_argument("--vae_dim", type=int, default=2)
    parser.add_argument("--architecture", choices=["mlp_elu", "vanillanet"], default="mlp_elu")

    ## LOSS args
    parser.add_argument("--cce_weight", type=float, default=1.0)
    parser.add_argument("--dsm_weight", type=float, default=0.0)
    parser.add_argument("--cd_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.01)

    ## SAVE args
    parser.add_argument('--save_root', type=str, default="trained_checkpoints",
                        help='path to save the checkpoint')
    parser.add_argument('--model_name', type=str, default='latent_jem',
                        help='name of the model to save')

    args = parser.parse_args()
    main(args)