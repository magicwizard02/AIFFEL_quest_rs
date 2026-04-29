import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes the context-aware representation of the input sequences.
    
    The attention mechanism allows the model to assign different 'importance' 
    weights to different tokens in the sequence relative to a specific token.
    
    Args:
        query (torch.Tensor): Representation of the current token. Shape: (batch, heads, seq_len, depth)
        key (torch.Tensor): Representation of all tokens to compare against. Shape: (batch, heads, seq_len, depth)
        value (torch.Tensor): The actual information to be extracted. Shape: (batch, heads, seq_len, depth)
        mask (torch.Tensor, optional): A mask to ignore specific tokens (Padding or Causal). 
                                       Expected shape: (batch, 1, 1, seq_len) or (1, 1, seq_len, seq_len)
    
    Returns:
        output (torch.Tensor): Weighted sum of the 'value' vectors.
        attention_weights (torch.Tensor): The probability distribution of attention scores.
    """
    
    # 1. Compute Raw Attention Scores (Similarity)
    # Perform matrix multiplication between Query and the transpose of Key.
    # Resulting shape: (batch, heads, seq_len, seq_len)
    matmul_qk = torch.matmul(query, key.transpose(-1, -2))

    # 2. Apply Scaling
    # Divide by the square root of the head dimension (d_k) to stabilize gradients.
    # Large dot products can push the softmax into regions with extremely small gradients.
    depth = key.size(-1)
    logits = matmul_qk / math.sqrt(depth)

    # 3. Apply Masking
    # If a mask is provided, fill masked positions with a very small negative value (-1e9).
    # This ensures that after softmax, these positions contribute 0 to the output.
    if mask is not None:
        # Positions where mask == 0 are ignored.
        logits = logits.masked_fill(mask == 0, -1e9)

    # 4. Softmax Normalization
    # Convert raw scores into a probability distribution that sums to 1.0.
    attention_weights = F.softmax(logits, dim=-1)

    # 5. Compute Weighted Sum
    # Multiply attention weights by the Value tensor to get the final context vector.
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention to allow the model to attend to information 
    from different representation subspaces simultaneously.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Ensure the model dimension is divisible by the number of heads.
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        # Linear projections for Query, Key, and Value.
        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)

        # Final linear layer to project the concatenated heads back to d_model.
        self.out_dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Reshapes the input tensor to separate the multiple attention heads.
        
        Input: (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, depth)
        """
        # 1. Split d_model into (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # 2. Transpose to move num_heads to the second dimension
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        """
        Defines the data flow for the Multi-Head Attention block.
        """
        batch_size = query.size(0)

        # STEP 1: Linear Transformation
        # Project inputs into Q, K, and V spaces.
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # STEP 2: Split Heads
        # Reshape for parallel attention computation across multiple heads.
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # STEP 3: Scaled Dot-Product Attention
        # Compute individual head attention.
        scaled_attention, weights = scaled_dot_product_attention(query, key, value, mask)

        # STEP 4: Concatenation
        # Merge all heads back into a single tensor.
        # Shape: (batch, seq_len, num_heads * depth) -> (batch, seq_len, d_model)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        # STEP 5: Final Projection
        # Apply the final output linear layer.
        output = self.out_dense(concat_attention)
        
        return output