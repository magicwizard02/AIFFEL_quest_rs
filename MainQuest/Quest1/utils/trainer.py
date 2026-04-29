import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """
    Executes a single training epoch with Causal Shift Alignment [32, 127].
    """
    model.train()
    batch_losses = []
    correct_tokens = 0
    total_tokens = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        # DATA UNPACKING: Convert to long() for Embedding/Loss compatibility
        input_ids, labels = [t.to(device).long() for t in batch]
        
        optimizer.zero_grad()
        
        # --- PHASE 1: FORWARD PASS ---
        outputs = model(input_ids) 
        
        # --- PHASE 2: CAUSAL LOSS COMPUTATION (SHIFTED) ---
        # Aligning outputs (t) with labels (t+1) to ensure Next-Token Prediction logic.
        vocab_size = outputs.shape[-1]

        # Shift Logic: Results in [Batch, Seq_Len - 1] -> e.g., [32, 127]
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        outputs_flattened = shift_logits.view(-1, vocab_size)
        targets_flattened = shift_labels.view(-1)

        # Loss handles 'ignore_index=-100' automatically
        loss = criterion(outputs_flattened, targets_flattened)
        
        # --- PHASE 3: OPTIMIZATION ---
        loss.backward()
        
        # Gradient Clipping to stabilize Attention layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # --- PHASE 4: ACCURACY CALCULATION ---
        _, predicted = torch.max(outputs_flattened, dim=1)
        
        # MASKING: Exclude both [PAD](0) and SFT-Mask(-100) for clean metrics
        mask = (targets_flattened != 0) & (targets_flattened != -100)
        correct_tokens += (predicted[mask] == targets_flattened[mask]).sum().item()
        total_tokens += mask.sum().item()
        
        batch_losses.append(loss.item())
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_acc = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    return batch_losses, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates model performance using the same Causal Shift Logic as training.
    """
    model.eval()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids, labels = [t.to(device).long() for t in batch]
            
            # Forward pass
            outputs = model(input_ids)
            
            # --- CRITICAL: Apply the same Shift Logic as train_one_epoch ---
            vocab_size = outputs.shape[-1]
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            outputs_flattened = shift_logits.view(-1, vocab_size)
            targets_flattened = shift_labels.view(-1)
            
            # Compute Val Loss
            loss = criterion(outputs_flattened, targets_flattened)
            total_loss += loss.item()
            
            # Compute Val Accuracy (excluding PAD and SFT masks)
            _, predicted = torch.max(outputs_flattened, dim=1)
            mask = (targets_flattened != 0) & (targets_flattened != -100)
            
            correct_tokens += (predicted[mask] == targets_flattened[mask]).sum().item()
            total_tokens += mask.sum().item()

    avg_val_loss = total_loss / len(loader)
    val_acc = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    
    return avg_val_loss, val_acc