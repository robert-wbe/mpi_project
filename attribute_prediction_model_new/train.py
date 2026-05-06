import torch
import torchvision
from torch.nn.functional import binary_cross_entropy as bce
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from tqdm import tqdm, trange
from model import model
import numpy as np
from pathlib import Path
from torchinfo import summary
import time

# ------ Parameters --------
OUTPUT_ROOT = '/usr/people/robertwiebe/ml_geometry_project/attribute_prediction_model_new/checkpoints'
RESUME_CHECKPOINT = None

TRAIN_BATCH_SIZE = 64 # 16
EVAL_BATCH_SIZE = 128 # 64
NUM_EPOCHS = 300
LEARNING_RATE = 3e-5
# VAL_RATIO = 0.2
# --------------------------

device = torch.device('cuda')
model.to(device)

transform = v2.Compose([
    v2.Resize((128, 128), interpolation=v2.InterpolationMode.BICUBIC),
    v2.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

eval_transform = v2.Compose([
    v2.Resize((128, 128)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

dataset = torchvision.datasets.CelebA(root='../datasets', split='train', transform=transform, download=False)
eval_dataset = torchvision.datasets.CelebA(root='../datasets', split='valid', transform=eval_transform, download=False)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-3) # weight decay was 1e-4
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)


train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=8)
val_loader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=8)

out_dir = Path(OUTPUT_ROOT)
print(f'Will save model checkpoint(s) to {str(out_dir)}')

if RESUME_CHECKPOINT:
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    print(f"Continuing from saved checkpoint at epoch {start_epoch + 1} with current best val loss of {best_val_loss}.")
else:
    start_epoch = 0
    best_val_loss = float('inf')

model_stats = summary(
    model,
    input_size=(1, 3, 128, 128),
    row_settings=["var_names"],
)
print("", flush=True)
time.sleep(1)

epoch_bar = trange(start_epoch, NUM_EPOCHS, desc='Training attribute prediction model', position=0)
for epoch in epoch_bar:

    model.train() # Fit the train set
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} | Train', position=1, leave=False)
    correct = 0
    total = 0
    train_acc: float
    for inputs, labels in train_bar:
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = bce(outputs, labels.float(), weight=torch.where(labels.bool(), 5, 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += ((outputs.detach() >= .5) == labels).sum().item()
        total += labels.numel()
        train_acc = correct / total

        train_bar.set_postfix({'loss': f'{loss.detach().item():.5f}', 'correct': f'{correct}/{total}', 'accuracy': f'{train_acc:.2f}'})
    
    # lr_scheduler.step()
    

    model.eval() # Validate the model
    val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1} | Val', position=1, leave=False)
    val_losses = []
    correct = 0
    total = 0
    val_acc: float = 0.
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = bce(outputs, labels.float(), weight=torch.where(labels.bool(), 5, 2)).item()
            val_losses.append(loss)
            correct += ((outputs >= .5) == labels).sum().item()
            total += labels.numel()
            val_acc = correct / total
            val_bar.set_postfix({'val loss': f'{loss:.5f}', 'correct': f'{correct}/{total}', 'accuracy': f'{val_acc:.2f}'})

    val_loss = sum(val_losses) / len(val_losses)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'val_acc': val_acc
        }
        # print(f'Epoch {epoch}- Saving best model checkpoint with val loss {best_val_loss:.5f} and val accuracy {val_acc:.2f}')
        epoch_bar.set_postfix_str(f'Current best validation loss {val_loss:.4f} w/ {val_acc:.2%} accuracy (reached at epoch {epoch + 1})')
        torch.save(checkpoint, out_dir / 'best_checkpoint.pt')

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    torch.save(checkpoint, out_dir / 'last_checkpoint.pt')
       