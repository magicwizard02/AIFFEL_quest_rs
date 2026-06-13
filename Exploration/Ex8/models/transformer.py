# ===========================================================================
# models/transformer.py
# Full Transformer Model Architecture (End-to-End)
# ===========================================================================
# This class orchestrates the entire flow:
# 1. Dynamically creates masks for the specific input batch.
# 2. Passes input through the Encoder to get context vectors.
# 3. Passes target through the Decoder using Encoder context.
# 4. Maps the final hidden states to the vocabulary size (Logits).
# ===========================================================================

import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .layers import create_padding_mask, create_look_ahead_mask

class Transformer(nn.Module):
    """
    The complete Transformer model for Sequence-to-Sequence tasks.
    """
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, dropout=0.1, max_len=100):
        super(Transformer, self).__init__()
        
        # -------------------------------------------------------------------
        # 1. Initialize Encoder and Decoder stacks
        # -------------------------------------------------------------------
        self.encoder = Encoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            ff_dim=ff_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len
        )
        
        self.decoder = Decoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            ff_dim=ff_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len
        )
        
        # -------------------------------------------------------------------
        # 2. Final Output Layer
        # Maps the Decoder output (d_model) to the vocabulary size
        # -------------------------------------------------------------------
        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        Defines the data flow during training.
        
        Args:
            src: Source sequence IDs [batch, src_seq_len]
            tgt: Target sequence IDs (Decoder Input) [batch, tgt_seq_len]
        """
        
        # -------------------------------------------------------------------
        # STEP 1: Generate Masks dynamically based on current batch
        # -------------------------------------------------------------------
        # enc_mask: Hides padding in the source sentence
        enc_mask = create_padding_mask(src)
        
        # dec_padding_mask: Hides padding in the source sentence 
        # specifically for the Decoder's Cross-Attention layers
        dec_padding_mask = create_padding_mask(src)
        
        # look_ahead_mask: Hides future tokens + padding in the target sentence
        look_ahead_mask = create_look_ahead_mask(tgt)

        # -------------------------------------------------------------------
        # STEP 2: Encoder pass
        # Transforms source IDs into continuous contextual representations
        # -------------------------------------------------------------------
        enc_out = self.encoder(src, enc_mask)

        # -------------------------------------------------------------------
        # STEP 3: Decoder pass
        # Generates output representations using target input and encoder context
        # -------------------------------------------------------------------
        dec_out = self.decoder(
            x=tgt, 
            enc_out=enc_out, 
            look_ahead_mask=look_ahead_mask, 
            padding_mask=dec_padding_mask
        )

        # -------------------------------------------------------------------
        # STEP 4: Vocabulary Projection
        # Logic: [batch, tgt_seq, d_model] -> [batch, tgt_seq, vocab_size]
        # -------------------------------------------------------------------
        logits = self.final_linear(dec_out)
        
        return logits