# ===========================================================================
# models/attention.py
# Core Attention Mechanisms for Transformer
# ===========================================================================

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention Function (Core Transformer Mechanism)
# ===========================================================================
# This module implements the core idea of Transformer attention:
#
# 1. Convert input embeddings into Q, K, V representations
# 2. Compute attention scores using dot product between Q and K
# 3. Apply scaling to stabilize gradients
# 4. Apply masks (padding + causal) BEFORE softmax
# 5. Normalize with softmax
# 6. Multiply by V to get context-aware representations
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes similarity scores and context-aware values.
    
    Logic:
    1. Dot product of Q and K.
    2. Scale by sqrt(d_k) for stability.
    3. Apply mask (fill with -1e9 for zeroed positions).
    4. Softmax to get weights.
    5. Multiply by V.

    Args:
        Q: Query tensor  [batch, heads, seq_len, d_k]
        K: Key tensor    [batch, heads, seq_len, d_k]
        V: Value tensor  [batch, heads, seq_len, d_v]
        mask: optional attention mask

    Returns:
        output: context-aware representation
        attention_weights: attention distribution matrix
    """
    # -----------------------------------------------------------------------
    # STEP 1: Compute raw attention scores
    # Q * K^T: Similarity between all tokens
    # Shape: [batch, heads, seq, seq]
    # -----------------------------------------------------------------------
    matmul_qk = torch.matmul(query, key.transpose(-1, -2))

    # -----------------------------------------------------------------------
    # STEP 2: Scale scores
    # Prevents large dot-products from saturating softmax 
    # Prevents gradients from vanishing/exploding during softmax
    # -----------------------------------------------------------------------
    depth = key.size(-1)
    logits = matmul_qk / math.sqrt(depth)

    # -----------------------------------------------------------------------
    # STEP 3: Apply mask (VERY IMPORTANT)
    # - Padding mask: ignore PAD tokens
    # - Causal mask: prevent future token leakage
    #
    # Masked positions are set to -inf so softmax becomes 0 (i.e. ignored)
    # -----------------------------------------------------------------------
    if mask is not None:
        # mask expected to be 1 for valid tokens, 0 for masked
        logits = logits.masked_fill(mask == 0, -1e9)

    # -----------------------------------------------------------------------
    # STEP 4: Softmax normalization
    # Converts scores into probability distribution (sum to 1)
    # -----------------------------------------------------------------------
    attention_weights = F.softmax(logits, dim=-1)

    # -----------------------------------------------------------------------
    # STEP 5: Weighted sum of values
    # Produces final context-aware representation
    # -----------------------------------------------------------------------
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention Module
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Basic Multi-Head Attention implementation for learning purposes.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Ensure embedding dimension is divisible by number of heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        # Linear layers for Q, K, V projections
        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Splits embedding into multiple attention heads.

        Input:
            x: [batch, seq_len, d_model]

        Output:
            [batch, heads, seq_len, d_k]

        Reshapes (batch, seq, d_model) -> (batch, heads, seq, depth)      
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        """
        batch_size = query.size(0)
        # -------------------------------------------------------------------
        # STEP 1: Linear projections (Linear transformations)
        # -------------------------------------------------------------------
        # 1. Linear transformations
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # -------------------------------------------------------------------
        # STEP 2: Split into multiple heads (Split heads for parallel processing)
        # -------------------------------------------------------------------
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # -------------------------------------------------------------------
        # STEP 3: Apply attention (Scaled Dot-Product Attention)
        # -------------------------------------------------------------------
        scaled_attention, weights = scaled_dot_product_attention(query, key, value, mask)

        # -------------------------------------------------------------------
        # STEP 4: Concatenate heads back to original d_model shape
        # -------------------------------------------------------------------
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        # -------------------------------------------------------------------
        # STEP 5: Final linear projection
        # -------------------------------------------------------------------
        output = self.out_dense(concat_attention)
        return output