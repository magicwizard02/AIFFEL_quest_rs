import torch
import torch.nn as nn
from .attention import AttentionDot

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, hidden_size):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = AttentionDot(hidden_size)
        
        # Linear layer to combine Decoder output and Attention context vector
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        
        # Final layer to project hidden states back to vocabulary size for prediction
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_input, decoder_input):
        # 1. Pass source text through Encoder
        encoder_outputs, hidden, cell = self.encoder(encoder_input)
        
        # 2. Pass target text through Decoder using Encoder's final states
        decoder_outputs, _, _ = self.decoder(decoder_input, hidden, cell)

        # 3. Apply Attention to find relevant parts of the source for each target word
        attn_out = self.attention(decoder_outputs, encoder_outputs)
        
        # 4. Concatenate (Stack) decoder hidden states and the context vector
        # Resulting shape: (batch, seq_len, hidden_size * 2)
        combined = torch.cat((decoder_outputs, attn_out), dim=-1)
        
        # 5. Compress back to hidden_size and apply non-linearity (tanh)
        combined = torch.tanh(self.concat(combined))
        
        # 6. Final prediction for every word in the sequence
        # Final shape: (batch, seq_len, vocab_size)
        output = self.output_layer(combined)
        
        return output