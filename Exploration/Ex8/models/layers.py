# ===========================================================================
# models/layers.py
# Shared Building Blocks for the Transformer Architecture
# ===========================================================================
# This module contains essential components used across both Encoder and Decoder:
# 1. FeedForward Network: Provides non-linearity and token-wise processing.
# 2. Positional Encoding: Injects sequence order into the attention mechanism.
# 3. Masking Utilities: Controls information flow (Padding & Causality).
# 4. LR Scheduler: Implements the "Warmup" learning strategy.
# ===========================================================================

import torch
import torch.nn as nn
import math

# ===========================================================================
# 1. Position-wise Feed Forward Network (FFN)
# ===========================================================================

class FeedForward(nn.Module):
    """
    Implementation of the Two-layer Linear Transformation with ReLU.
    
    While Attention focuses on 'relationships' between tokens, the FFN
    processes each token 'individually' to extract deeper features.
    """
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        # STEP 1: Expand dimension to d_ff (usually 4x d_model)
        # This creates a high-dimensional space for non-linear mapping.
        self.fc1 = nn.Linear(d_model, d_ff)
        
        # STEP 2: Project back to original d_model dimension
        # Ensures the output shape remains compatible with the next block.
        self.fc2 = nn.Linear(d_ff, d_model)
        
        # Non-linear activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Logic: Linear -> ReLU -> Linear
        return self.fc2(self.relu(self.fc1(x)))


# ===========================================================================
# 2. Positional Encoding (PE)
# ===========================================================================

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of tokens.
    
    Since Transformers do not use recurrence (RNN), they have no inherent 
    sense of order. Sinusoidal functions are used to provide this signal.
    """
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        # Generate the static PE matrix once during initialization
        self.pos_encoding = self._build_pos_encoding(position, d_model)

    def _get_angles(self, position, i, d_model):
        # Formula: pos / 10000^(2i/d_model)
        # Determines the frequency of the sine/cosine waves for each dimension.
        return 1.0 / (10000.0 ** ((2.0 * (i // 2)) / d_model)) * position

    def _build_pos_encoding(self, position, d_model):
        # Create a range of positions [0...max_len] and dimensions [0...d_model]
        pos = torch.arange(position, dtype=torch.float32).unsqueeze(1) # [pos, 1]
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)   # [1, d_model]
        
        angle_rads = self._get_angles(pos, i, d_model)

        # Apply sine to even indices (0, 2, ...)
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        # Apply cosine to odd indices (1, 3, ...)
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        # Add batch dimension: [1, position, d_model]
        return angle_rads.unsqueeze(0)

    def forward(self, x):
        # Add PE to input embeddings
        # We slice the PE matrix to match the actual input sequence length
        # .to(x.device) ensures computation happens on GPU if x is on GPU
        return x + self.pos_encoding[:, :x.size(1), :].to(x.device)


# ===========================================================================
# 3. Masking Utilities
# ===========================================================================

def create_padding_mask(x):
    """
    Prevents the model from attending to PAD tokens (0).
    Returns a Boolean tensor for logical operations.
    """
    # Changed .float() to .bool() or simply kept as boolean
    # Logic: True for real tokens, False for padding (0)
    mask = (x != 0) 
    
    # Reshape for multi-head attention: [batch, 1, 1, seq_len]
    return mask.unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(x):
    """
    Used in the Decoder to prevent attending to future tokens.
    Combines Padding Mask and Causal Mask using Boolean logic.
    """
    seq_len = x.size(1)
    
    # Create a lower triangular matrix (Causal Mask) as Boolean
    look_ahead_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
    
    # Get the padding mask as Boolean
    padding_mask = create_padding_mask(x)
    
    # Now both are Boolean, so '&' will work perfectly!
    return padding_mask & look_ahead_mask


# ===========================================================================
# 4. Warmup Learning Rate Scheduler
# ===========================================================================

def get_lr_lambda(d_model, warmup_steps=4000):
    """
    Implements the learning rate schedule from the original paper.
    
    Increases LR linearly for 'warmup_steps', then decreases it 
    proportional to the inverse square root of the step number.
    """
    def lr_lambda(step):
        # Step + 1 to avoid division by zero at step 0
        step = step + 1
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return lr_lambda