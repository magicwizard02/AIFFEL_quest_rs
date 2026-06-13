# utils/trainer.py

import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """
    Performs one full pass through the training dataset for Image Classification.
    """
    model.train()
    batch_losses = []
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels, _ in progress_bar:
        # Move data to target device (GPU/MPS/CPU)
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # --- FORWARD PASS ---
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # --- BACKWARD PASS ---
        loss.backward()
        
        # Gradient Clipping (Useful for preventing spikes even in CNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # --- METRICS CALCULATION ---
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        batch_losses.append(loss.item())
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_acc = (correct / total) * 100 if total > 0 else 0
    return batch_losses, epoch_acc

def validate(model, loader, criterion, device):
    """
    Evaluates the CNN model performance on the validation set.
    """
    model.eval()
    val_losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
    val_acc = (100. * correct / total) if total > 0 else 0
    return avg_val_loss, val_acc