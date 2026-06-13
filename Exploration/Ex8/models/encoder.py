# ===========================================================================
# STEP: Transformer Encoder Block
# ===========================================================================
# The Encoder processes the input sequence (questions) and generates
# a high-dimensional representation that captures the context of the input.
#
# Structure:
# 1. Embedding + Positional Encoding
# 2. N x Encoder Layers (Self-Attention + FFN)
#
# This is the fundamental building block of the Transformer encoder.
# ===========================================================================


import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention
from .layers import PositionalEncoding, FeedForward

# ===========================================================================
# 1. Encoder Layer (Individual Block)
# ===========================================================================

class EncoderLayer(nn.Module):
    """
    A single unit of the Encoder. It consists of two sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed Forward Network
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Self-attention mechanism to find relationships between input tokens
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        # Feed Forward Network to process token features individually
        self.ffn = FeedForward(d_model, ff_dim)
        
        # Layer Normalization to stabilize the hidden state distributions
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Dropout for regularization to prevent overfitting
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # -------------------------------------------------------------------
        # Sub-layer 1: Multi-Head Self-Attention
        # -------------------------------------------------------------------
        # x is used as Query, Key, and Value (Self-Attention)
        attn_out = self.mha(x, x, x, mask)
        
        # Residual Connection + Dropout + LayerNorm
        # Formula: LayerNorm(x + Dropout(SubLayer(x)))
        x = self.norm1(x + self.dropout1(attn_out))

        # -------------------------------------------------------------------
        # Sub-layer 2: Feed Forward Network
        # -------------------------------------------------------------------
        ffn_out = self.ffn(x)
        
        # Second Residual Connection + Dropout + LayerNorm
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x


# ===========================================================================
# 2. Encoder (The Full Stack)
# ===========================================================================

class Encoder(nn.Module):
    """
    The full Encoder stack that repeats EncoderLayers N times.
    """
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, dropout=0.1, max_len=100):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        
        # Converts token IDs to continuous vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Adds spatial information to embeddings
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Creating multiple layers of the Encoder stack
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        # 1. Embedding lookup and scaling by sqrt(d_model)
        # Scaling helps stabilize the variance before adding Positional Encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 2. Add Positional Encoding and apply Dropout
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 3. Pass through the stack of N Encoder Layers
        for layer in self.enc_layers:
            x = layer(x, mask)
            
        return x  # Shape: [batch, seq_len, d_model]