import torch
import numpy as np
from model import VAE
from dataset import dataloader_latents_only as dataloader, mean as latents_mean, std as latents_std
from tqdm import tqdm, trange
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from torch.nn.functional import l1_loss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils import kl_divergence
from torchinfo import summary
import time

# lambda_rec = 1.0
# lambda_ssim = 0.2
# lambda_kl = 0.00001
save_path_best = 'checkpoints/continued/latent_vae_best.pt'
# save_path_last = 'checkpoints/continued/latent_vae_last.pt'


ssim = StructuralSimilarityIndexMeasure().eval().cuda()
diff_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", device_map='cuda').eval()

vae = VAE(latent_features=64, img_channels=4).cuda()
vae.encoder.eval()
vae.decoder.train()
vae.load_state_dict(torch.load('checkpoints/continued/latent_vae_epoch_110.pt')['state_dict']) # CONTINUE FROM CHECKPOINT
# optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-5) # 1e-4
optimizer = torch.optim.AdamW(vae.decoder.parameters(), lr=1e-5) # 1e-4

model_stats = summary(
    vae,
    input_size=(1, 4, 16, 16),
    row_settings=["var_names"],
)
print("", flush=True)
time.sleep(1)

best_loss = np.inf
pbar = trange(110, 150, desc='Training Latent VAE', position=0)
for epoch in pbar:
    rec_losses, ssim_losses, kl_losses, total_losses = [], [], [], []
    epoch_pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/150', position=1, leave=False)
    # for (_, _), (latents, ) in epoch_pbar:
    for (latents, ) in epoch_pbar:
        # if epoch >= 100:
        #     images = images.cuda()
        latents = latents.cuda()

        reconstr, mu, logvar = vae(latents)
        # with torch.no_grad():
        #     small_latents = vae.encode(latents, sample_posterior=False)
        # reconstr = vae.decode(small_latents)

        # if epoch < 100:
        if True:
            reconstr_loss = l1_loss(reconstr, latents)
            # ssim_loss = 1 - ssim(reconstr, latents)
        else:
            denormalized = reconstr * latents_std + latents_mean
            full_reconstr = diff_vae.decode(denormalized)[0]
            # if epoch < 100:
            #     w1, w2 = (100-epoch)/50, (epoch-50)/50
            #     reconstr_loss = w1*l1_loss(reconstr, latents) + w2*l1_loss(full_reconstr, images)
            #     ssim_loss = w1*(1-ssim(reconstr, latents)) + w2*(1-ssim(full_reconstr, images))
            # else:
            reconstr_loss = l1_loss(full_reconstr, images)
            ssim_loss = 1 - ssim(full_reconstr, images)
        
        # kl_loss = kl_divergence(mu, logvar).mean()

        # total_loss = lambda_rec * reconstr_loss + lambda_ssim * ssim_loss + lambda_kl * kl_loss
        total_loss = reconstr_loss
        # total_loss = lambda_rec * reconstr_loss + lambda_ssim * ssim_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
        optimizer.step()

        # epoch_pbar.set_postfix(rec_loss=(lambda_rec * reconstr_loss.item()), ssim_loss=(lambda_ssim * ssim_loss.item()), kl_loss=(lambda_kl * kl_loss.item()), total_loss=total_loss.item())
        # epoch_pbar.set_postfix(rec_loss=(lambda_rec * reconstr_loss.item()), ssim_loss=(lambda_ssim * ssim_loss.item()), total_loss=total_loss.item())
        epoch_pbar.set_postfix(loss=total_loss.item())
        # rec_losses.append(lambda_rec * reconstr_loss.item())
        # ssim_losses.append(lambda_ssim * ssim_loss.item())
        # kl_losses.append(lambda_kl * kl_loss.item())
        total_losses.append(total_loss.item())
    
    # rec_loss = sum(rec_losses) / len(rec_losses)
    # ssim_loss = sum(ssim_losses) / len(ssim_losses)
    # kl_loss = sum(kl_losses) / len(kl_losses)
    total_loss = sum(total_losses) / len(total_losses)
    # pbar.set_postfix(rec_loss=rec_loss, ssim_loss=ssim_loss, kl_loss=kl_loss, total_loss=total_loss)
    # pbar.set_postfix(rec_loss=rec_loss, ssim_loss=ssim_loss, total_loss=total_loss)
    pbar.set_postfix(loss=total_loss)

    if not ((epoch+1) % 10):
    # if True:
        torch.save(
            {
                'state_dict' : vae.state_dict(),
                # 'rec_loss' : rec_loss,
                # 'ssim_loss' : ssim_loss,
                # 'kl_loss' : kl_loss,
                'total_loss' : total_loss,
            },
            f'checkpoints/continued/latent_vae_epoch_{epoch + 1}.pt'
        )

    if total_loss < best_loss:
        best_loss = total_loss
        torch.save({'state_dict' : vae.state_dict(), 'total_loss' : total_loss}, save_path_best)
    
# torch.save(vae.state_dict(), save_path_last)