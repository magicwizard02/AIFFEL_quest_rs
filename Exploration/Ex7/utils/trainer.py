import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one full training pass.
    1. Sets model to train mode.
    2. Iterates through batches, calculating loss and gradients.
    3. Updates model weights via optimizer.
    """
    model.train()
    batch_losses = []
    correct, total = 0, 0
    
    for inputs, decoder_input, targets in loader:
        inputs = inputs.to(device).long()
        decoder_input = decoder_input.to(device).long()
        targets = targets.to(device).long()
        
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(inputs, decoder_input)
        
        # Reshape for CrossEntropyLoss (Batch*Seq, Vocab)
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
        
        # Metrics Calculation
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        # Ignore padding (0) in accuracy if necessary
        correct += (predicted == targets).sum().item()
        
    epoch_acc = 100. * correct / total
    return batch_losses, epoch_acc

def validate(model, loader, criterion, device):
    """
    Evaluates the model on test data without calculating gradients.
    """
    model.eval()
    val_losses = []
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, decoder_input, targets in loader:
            inputs = inputs.to(device).long()
            decoder_input = decoder_input.to(device).long()
            targets = targets.to(device).long()
            
            outputs = model(inputs, decoder_input)
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
            
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_acc = 100. * correct / total
    return avg_val_loss, val_acc