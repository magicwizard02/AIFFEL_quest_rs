import torch

def create_look_ahead_mask(x, pad_id=0):
    """
    Unified masking for GPT Decoder-only logic.
    Combines:
    1. Causal Mask (Look-ahead): Prevents attending to future tokens.
    2. Padding Mask: Prevents attending to [PAD] tokens.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch, seq_len).
        pad_id (int): The ID used for padding in your tokenizer.
        
    Returns:
        torch.Tensor: A combined boolean mask.
    """
    seq_len = x.size(1)
    
    # 1. Causal Mask: A lower triangular matrix of ones.
    # Elements at (i, j) where i >= j are 1 (keep), others are 0 (mask).
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
    
    # 2. Padding Mask: Marks where the input is NOT the pad_id.
    # Shape: (batch, 1, 1, seq_len) to allow broadcasting across heads.
    padding_mask = (x != pad_id).unsqueeze(1).unsqueeze(2)
    
    # 3. Combined Mask: Logic AND
    # The model only attends if a token is NOT padding AND is NOT in the future.
    return padding_mask & causal_mask