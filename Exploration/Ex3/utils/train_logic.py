import torch

def train_one_epoch(model, loader, criterion, optimizer, device, aug_type=None):
    """
    Performs one full training pass (epoch) over the dataset.
    Supports Standard training, Mixup, and CutMix augmentation.
    Returns:
        - batch_losses (list): A list containing the loss value of every batch.
        - epoch_acc (float): The final accuracy percentage for this epoch.
    """
    model.train()  # Set the model to training mode (activates Dropout, Batchnorm, etc.)
    
    # --- STORAGE FOR LOGGER ---
    # We use a list to store every individual batch loss for detailed tracking
    batch_losses = [] 
    
    correct = 0
    total = 0
    
    # Iterate over batches of images and labels
    for inputs, labels in loader:
        # Move data to the active device (MPS for Mac or CPU)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Reset gradients from the previous step to prevent accumulation
        optimizer.zero_grad()

        # --- AUGMENTATION LOGIC BRANCHING ---
        # CASE A: Mixup Augmentation
        if aug_type == 'mixup':
            # Blends two images and provides two sets of labels with a mixing ratio (lam)
            inputs, targets_a, targets_b, lam = mixup(inputs, labels)
            outputs = model(inputs)
            # Calculate a weighted loss based on the mixing ratio
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
        # CASE B: CutMix Augmentation
        elif aug_type == 'cutmix':
            # Patches a portion of one image onto another
            inputs, targets_a, targets_b, lam = cutmix(inputs, labels)
            outputs = model(inputs)
            # Loss is weighted by the area of the patch
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
        # CASE C: Standard Training (No-Aug or Basic-Aug)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # --- BACKPROPAGATION ---
        loss.backward()      # Compute gradients
        optimizer.step()     # Update model weights

        # --- METRICS CALCULATION ---
        # Save the current batch loss into our list for the CSV logger
        batch_losses.append(loss.item())
        
        # Calculate Accuracy: 
        # Note: For Mixup/CutMix, we measure accuracy against the dominant class (targets_a)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # --- FINAL CALCULATIONS ---
    # Calculate the total accuracy percentage for the entire epoch
    epoch_acc = 100. * correct / total
    
    # IMPORTANT: We return the LIST of losses, not the average, to satisfy update_results_refined()
    return batch_losses, epoch_acc


@torch.no_grad() # Disables gradient calculation to save memory and speed up inference
def validate(model, loader, device):
    """
    Evaluates the model's performance on the validation/test dataset.
    This function measures how well the model generalizes to unseen data.
    """
    model.eval() # Set the model to evaluation mode (deactivates Dropout, freezes Batchnorm)
    
    correct = 0
    total = 0
    
    # Iterate through the validation dataset
    for inputs, labels in loader:
        # Transfer tensors to the active device (MPS or CPU)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # FORWARD PASS ONLY
        # We do not compute loss or gradients here to keep the process "Pure"
        outputs = model(inputs)
        
        # PREDICTION LOGIC
        # Find the index of the highest logit value (the model's most confident guess)
        _, predicted = outputs.max(1)
        
        # TALLY RESULTS
        total += labels.size(0)
        # Compare the predicted index with the ground-truth label
        correct += predicted.eq(labels).sum().item()
        
    # Calculate Final Accuracy Percentage
    val_acc = 100. * correct / total
    
    return val_acc






def mixup(inputs, labels, alpha=1.0):
    """
    Applies Mixup augmentation by blending two images and their labels.
    
    Args:
        inputs (Tensor): Batch of images [Batch, C, H, W]
        labels (Tensor): Batch of class indices
        alpha (float): Beta distribution parameter (controls mixing strength)
        
    Returns:
        mixed_x (Tensor): Blended images
        y_a (Tensor): Original ground truth labels
        y_b (Tensor): Shuffled partner labels
        lam (float): Mixing ratio (lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size(0)
    # Randomly shuffle the batch to pick 'partner' images for mixing
    index = torch.randperm(batch_size).to(inputs.device)

    # Blend the two images: (lam * Image_A) + (1 - lam) * Image_B
    mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
    y_a, y_b = labels, labels[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix(inputs, labels, alpha=1.0):
    """
    Applies CutMix augmentation by pasting a random patch of image B onto image A.
    
    Args:
        inputs (Tensor): Batch of images [Batch, C, H, W]
        labels (Tensor): Batch of class indices
        alpha (float): Beta distribution parameter
        
    Returns:
        mixed_x (Tensor): Images with patches applied
        y_a (Tensor): Original ground truth labels
        y_b (Tensor): Shuffled partner labels (source of the patch)
        lam (float): Adjusted area ratio of the original image
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)

    W, H = inputs.size(2), inputs.size(3)
    
    # Calculate patch dimensions based on the sampled ratio
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Select a random center point for the patch
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Define the bounding box and clip it to image boundaries
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply the patch from the shuffled images
    mixed_x = inputs.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
    
    # ADJUST LAMBDA: Recalculate based on the actual pixel count replaced
    # This ensures the loss weight matches the visual content exactly.
    actual_patch_area = (bbx2 - bbx1) * (bby2 - bby1)
    lam = 1 - (actual_patch_area / (W * H))
    
    y_a, y_b = labels, labels[index]
    
    return mixed_x, y_a, y_b, lam