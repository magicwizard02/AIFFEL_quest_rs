import torch
import torch.nn as nn
from .layers import DecoderLayer
from utils import create_look_ahead_mask

class Decoder(nn.Module):
    """
    The full Decoder stack consisting of multiple Decoder layers.
    It handles token embeddings, positional encoding, and the sequential
    processing of transformer blocks.
    """
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, dropout=0.1, max_len=100):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        
        # 1. Token Embedding: Converts token IDs into continuous vectors.
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Embedding: Learns the relative or absolute position of tokens.
        # Since Transformers have no inherent sense of order, this is vital.
        self.pos_embedding = nn.Embedding(max_len, d_model) 
        
        self.dropout = nn.Dropout(dropout)
        
        # 3. Stack of Decoder Layers: The main processing units.
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])

        # 4. Final Projection: Maps the d_model hidden states back to vocabulary size.
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, look_ahead_mask=None):
        """
        Processes the input sequence through the embedding and decoder stack.
        """
        batch_size, seq_len = x.size()
        
        # Automatically generate causal mask if not provided during the forward pass.
        if look_ahead_mask is None:
            look_ahead_mask = create_look_ahead_mask(x, pad_id=0)
        
        # Generate position indices for the positional embedding.
        # Shape: (1, seq_len)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        # Combine Token and Positional Embeddings
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Sequential processing through the stack of transformer blocks.
        for layer in self.dec_layers:
            x = layer(x, look_ahead_mask)
        
        # Project to logits (raw scores for each word in the vocabulary).
        logits = self.fc_out(x)

        return logits


class GPTModel(nn.Module):
    """
    Top-level wrapper for the GPT (Decoder-only) architecture.
    Designed for Causal Language Modeling (CLM) and Supervised Fine-Tuning (SFT).
    """
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, dropout=0.1, max_len=100):
        super(GPTModel, self).__init__()
        
        # Initialize the Decoder stack.
        self.decoder = Decoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            ff_dim=ff_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len
        )

    def forward(self, x, look_ahead_mask=None):
        """
        Defines the data flow for autoregressive training.
        
        Args:
            x: Input token IDs [batch, seq_len]
            look_ahead_mask: Optional mask to enforce causality.
        
        Returns:
            logits: Predicted scores for the next token in the sequence.
        """
        
        # In this GPT architecture, we only utilize the Decoder pass.
        # There is no Encoder-Decoder cross-attention.
        logits = self.decoder(
            x=x, 
            look_ahead_mask=look_ahead_mask
        )
        
        return logits