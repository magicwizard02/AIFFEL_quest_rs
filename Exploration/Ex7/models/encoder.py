import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=3, dropout=0.4):
        super(Encoder, self).__init__()
        # Lookup table that stores embeddings of a fixed dictionary and size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multi-layer Long Short-Term Memory (LSTM) network
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers,
            dropout=dropout, 
            batch_first=True
        )

    def forward(self, x):
        # x: (batch, seq_len) -> Convert word indices to dense vectors
        embedded = self.embedding(x)
        
        # Pass embeddings through LSTM. 
        # output: all hidden states for each time step (used for Attention)
        # hidden/cell: final states from the last time step (used to initialize Decoder)
        output, (hidden, cell) = self.lstm(embedded)
        
        return output, hidden, cell