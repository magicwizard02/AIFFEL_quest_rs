# ===========================================================================
# STEP: Transformer Decoder Block
# ===========================================================================
# The Decoder generates the target sequence token-by-token.
# It attends to its own previous tokens AND the Encoder's output.
#
# Structure:
# 1. Embedding + Positional Encoding
# 2. N x Decoder Layers (Self-Attention + Cross-Attention + FFN)
# ===========================================================================

import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention
from .layers import PositionalEncoding, FeedForward

# ===========================================================================
# 1. Decoder Layer (Individual Block)
# ===========================================================================

class DecoderLayer(nn.Module):
    """
    A single unit of the Decoder. It consists of three sub-layers:
    1. Masked Multi-Head Self-Attention (Causal)
    2. Multi-Head Cross-Attention (Attends to Encoder output)
    3. Position-wise Feed Forward Network
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Self-attention for target tokens
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        
        # Cross-attention to look at Encoder's context
        self.enc_dec_mha = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = FeedForward(d_model, ff_dim)
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, look_ahead_mask, padding_mask):
        # -------------------------------------------------------------------
        # Sub-layer 1: Masked Self-Attention
        # -------------------------------------------------------------------
        # Uses look_ahead_mask to prevent "cheating" from future tokens
        out1 = self.self_mha(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout1(out1))

        # -------------------------------------------------------------------
        # Sub-layer 2: Encoder-Decoder Cross-Attention
        # -------------------------------------------------------------------
        # Query comes from Decoder (x), Key & Value come from Encoder (enc_out)
        out2 = self.enc_dec_mha(x, enc_out, enc_out, padding_mask)
        x = self.norm2(x + self.dropout2(out2))

        # -------------------------------------------------------------------
        # Sub-layer 3: Feed Forward Network
        # -------------------------------------------------------------------
        out3 = self.ffn(x)
        x = self.norm3(x + self.dropout3(out3))
        
        return x


# ===========================================================================
# 2. Decoder (The Full Stack)
# ===========================================================================

class Decoder(nn.Module):
    """
    The full Decoder stack that repeats DecoderLayers N times.
    """
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, dropout=0.1, max_len=100):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of N Decoder Layers
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, look_ahead_mask, padding_mask):
        # 1. Embedding and scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 2. Add Positional Encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 3. Sequential processing through Decoder stack
        for layer in self.dec_layers:
            # Each layer requires look_ahead (self) and padding (cross) masks
            x = layer(x, enc_out, look_ahead_mask, padding_mask)
            
        return x  # Shape: [batch, seq_len, d_model]