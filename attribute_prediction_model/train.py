import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from tqdm import tqdm
from model import model
import numpy as np
from pathlib import Path

# ------ Parameters --------
OUTPUT_ROOT = '/usr/people/robertwiebe/ml_geometry_project/attribute_prediction_model/checkpoints'
RESUME_CHECKPOINT = None

TRAIN_BATCH_SIZE = 64 # 16
EVAL_BATCH_SIZE = 128 # 64
NUM_EPOCHS = 300
LEARNING_RATE = 0.1
# VAL_RATIO = 0.2
# --------------------------

device = torch.device('cuda')
model.to(device)

transform = v2.Compose([
    v2.Resize((128, 128), interpolation=v2.InterpolationMode.BICUBIC),
    v2.ColorJitter(brightness=0.1, saturation=0.1, contrast=0.1),
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

criterion = nn.BCELoss()
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-3) # weight decay was 1e-4
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)


train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=8)
val_loader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=8)

out_dir = Path(OUTPUT_ROOT)
print(f'Will save model checkpoint(s) to {str(out_dir)}')

start_epoch = 0
if RESUME_CHECKPOINT:
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']

    print(f"Continuing from saved checkpoint at epoch {start_epoch} with current best val loss of {best_val_loss}.")
else:
    best_val_loss = float('inf')

for epoch in range(start_epoch, NUM_EPOCHS):

    model.train() # Fit the train set
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch} | Train')
    correct = 0
    total = 0
    train_acc: float
    for inputs, labels in train_bar:
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += ((outputs.detach() >= .5) == labels).sum().item()
        total += labels.numel()
        train_acc = correct / total

        train_bar.set_postfix({'loss': f'{loss.detach().item():.5f}', 'correct': f'{correct}/{total}', 'accuracy': f'{train_acc:.2f}'})
    
    lr_scheduler.step()
    

    model.eval() # Validate the model
    val_bar = tqdm(val_loader, desc=f'Epoch {epoch} | Val')
    val_losses = []
    correct = 0
    total = 0
    val_acc: float = 0.
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            val_losses.append(loss)
            correct += ((outputs >= .5) == labels).sum().item()
            total += labels.numel()
            val_acc = correct / total
            val_bar.set_postfix({'val loss': f'{loss:.5f}', 'correct': f'{correct}/{total}', 'accuracy': f'{val_acc:.2f}'})

    val_loss = sum(val_losses) / len(val_losses)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'val_acc': val_acc
        }
        print(f'Epoch {epoch}- Saving best model checkpoint with val loss {best_val_loss:.5f} and val accuracy {val_acc:.2f}')
        torch.save(checkpoint, out_dir / 'best_checkpoint.pt')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    torch.save(checkpoint, out_dir / 'last_checkpoint.pt')
       