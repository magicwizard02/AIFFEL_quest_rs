import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN).
    
    While the Attention mechanism captures relationships between tokens, the FFN 
    processes each token position independently to extract higher-level features.
    """
    def __init__(self, d_model, d_ff=2048):
        super(FeedForward, self).__init__()
        # 1. First Linear Transformation: Expands the dimension from d_model to d_ff.
        # This expansion allows the model to learn more complex patterns.
        self.fc1 = nn.Linear(d_model, d_ff)
        
        # 2. Second Linear Transformation: Projects back to the original d_model.
        self.fc2 = nn.Linear(d_ff, d_model)
        
        # Non-linear activation function to introduce complexity beyond simple linear mapping.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Processes input x of shape (batch_size, seq_len, d_model).
        Logic: Linear -> ReLU -> Linear
        """
        return self.fc2(self.relu(self.fc1(x)))


class DecoderLayer(nn.Module):
    """
    Represents a single block in the GPT (Decoder-only) architecture.
    
    Each layer consists of two main sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    
    Residual connections and Layer Normalization are applied to both sub-layers 
     to ensure stable gradient flow and deeper training.
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Sub-layer 1: Self-Attention mechanism
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 2: Feature extraction network
        self.ffn = FeedForward(d_model, ff_dim)
        
        # Normalization layers to stabilize hidden state distributions
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Dropout for regularization to prevent overfitting during fine-tuning
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, look_ahead_mask):
        """
        Forward pass for a single Decoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            look_ahead_mask: Causal mask for the self-attention mechanism.
        """
        
        # --- Sub-layer 1: Multi-Head Self-Attention ---
        # Note: In self-attention, Query, Key, and Value all come from the same input 'x'.
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        
        # Residual Connection & Layer Normalization
        # Form: Norm(x + Dropout(Sublayer(x)))
        x = self.norm1(x + self.dropout1(attn_output))

        # --- Sub-layer 2: Feed-Forward Network ---
        ffn_output = self.ffn(x)
        
        # Second Residual Connection & Layer Normalization
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x