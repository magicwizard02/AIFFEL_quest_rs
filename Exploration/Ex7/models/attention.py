import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionDot(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionDot, self).__init__()
        # Dot product attention often uses no learnable parameters, 
        # but focuses on the similarity between vectors.

    def forward(self, decoder_output, encoder_outputs):
        # decoder_output: (batch, dec_seq_len, hidden_size)
        # encoder_outputs: (batch, enc_seq_len, hidden_size)
        
        # Calculate alignment scores using Batch Matrix Multiplication (BMM)
        # We transpose encoder_outputs to (batch, hidden_size, enc_seq_len) for dot product
        # result shape: (batch, dec_seq_len, enc_seq_len)
        attn_weights = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        
        # Apply Softmax to get probabilities (weights sum to 1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Multiply weights by encoder outputs to get the "Context Vector"
        # result shape: (batch, dec_seq_len, hidden_size)
        context_vector = torch.bmm(attn_weights, encoder_outputs)
        
        return context_vector