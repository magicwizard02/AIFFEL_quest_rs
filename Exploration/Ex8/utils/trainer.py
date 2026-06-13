# ===========================================================================
# utils/trainer.py
# Training and Validation Logic for Transformer Architecture
# ===========================================================================

import torch
import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """
    Performs one full pass through the training dataset.
    
    Detailed Breakdown:
    1. Data Transfer: Moving tensors to MPS (Mac) or CPU.
    2. Forward Pass: Calculating the model's prediction.
    3. Loss Calculation: Comparing predictions vs. targets (ignoring PAD).
    4. Optimization: Updating weights and adjusting the Learning Rate (Warmup).
    """
    model.train()
    batch_losses = []
    correct_tokens = 0
    total_tokens = 0
    
    # Progress bar for visual feedback in Jupyter
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        # --- PREPARATION ---
        # src: Question (Encoder Input)
        # tgt: Answer with [BOS] (Decoder Input)
        # tgt_y: Answer with [EOS] (The 'Ground Truth' we want to predict)
        src, tgt, tgt_y = [t.to(device).long() for t in batch]
        
        # Clear previous gradients before starting the new pass
        optimizer.zero_grad()
        
        # --- STEP 1: FORWARD PASS ---
        # The Transformer processes the entire sequence in parallel using Attention.
        # It produces a probability distribution for every word in the sequence.
        outputs = model(src, tgt)
        
        # --- STEP 2: RESHAPE FOR LOSS ---
        # CrossEntropyLoss requires a 2D input [Total_Tokens, Vocab_Size].
        # We flatten the Batch and Sequence dimensions together.
        vocab_size = outputs.shape[-1]
        outputs_flattened = outputs.view(-1, vocab_size)
        targets_flattened = tgt_y.view(-1)
        
        # --- STEP 3: COMPUTE LOSS ---
        # The 'criterion' uses ignore_index=0 to make sure we don't calculate 
        # loss for the [PAD] tokens.
        loss = criterion(outputs_flattened, targets_flattened)
        
        # --- STEP 4: BACKWARD PASS ---
        # Calculate the gradients (how much each weight contributed to the error)
        loss.backward()
        
        # --- STEP 5: GRADIENT CLIPPING ---
        # Critical for Transformers! Attention layers can create massive gradients.
        # Clipping at 1.0 prevents "Exploding Gradients" which would ruin the model.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # --- STEP 6: OPTIMIZER & SCHEDULER STEP ---
        # Update the actual weights of the model
        optimizer.step()
        
        # If a scheduler is provided (for Warmup), update the Learning Rate 
        # after EVERY batch. This ensures a smooth learning curve.
        if scheduler is not None:
            scheduler.step()
        
        # --- STEP 7: CALCULATE ACCURACY (EXCLUDING PADDING) ---
        # We only care if the model predicted the actual words correctly.
        _, predicted = torch.max(outputs_flattened, dim=1)
        
        # Create a mask to ignore index 0 (PAD) in accuracy calculation
        non_pad_mask = (targets_flattened != 0)
        correct_tokens += (predicted[non_pad_mask] == targets_flattened[non_pad_mask]).sum().item()
        total_tokens += non_pad_mask.sum().item()
        
        batch_losses.append(loss.item())
        
        # Update progress bar description with current loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Calculate average accuracy for the whole epoch
    epoch_acc = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    
    return batch_losses, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates model performance on the validation set.
    Uses 'torch.no_grad()' to save memory and compute time.
    """
    model.eval()
    val_losses = []
    correct, total = 0, 0
    
    with torch.no_grad():
        for src, tgt, tgt_y in loader:
            src = src.to(device).long()
            tgt = tgt.to(device).long()
            tgt_y = tgt_y.to(device).long()
            
            # Forward pass only, no backpropagation
            outputs = model(src, tgt)
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = tgt_y.view(-1)
            
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
            
            # Metrics calculation (Same logic as Training)
            _, predicted = torch.max(outputs, 1)
            non_pad_mask = (targets != 0)
            total += non_pad_mask.sum().item()
            correct += (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
            
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_acc = 100. * correct / total if total > 0 else 0
    return avg_val_loss, val_acc


def get_lr_lambda(d_model, warmup_steps=4000):
    """
    Creates a learning rate scheduling function based on the 'Attention Is All You Need' paper.
    
    The learning rate increases linearly for the first 'warmup_steps' and then 
    decreases proportional to the inverse square root of the step number.
    
    Args:
        d_model (int): The dimensionality of the model (embedding size).
        warmup_steps (int): The number of steps to increase the LR linearly.
        
    Returns:
        A lambda function that calculates the scaling factor for the base learning rate.
    """
    d_model = float(d_model)
    
    def lr_lambda(step):
        # Correct step indexing (start from 1 instead of 0 to avoid division by zero)
        step = step + 1
        
        # Original formula: d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        
    return lr_lambda