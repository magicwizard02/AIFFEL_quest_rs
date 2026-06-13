import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=3, dropout=0.4):
        super(Decoder, self).__init__()
        # Map target word indices (headlines) to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer that takes target embeddings and Encoder's final (h, c)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )

    def forward(self, x, hidden, cell):
        # x: (batch, target_seq_len) -> Target word embeddings
        embedded = self.embedding(x)
        
        # Generate decoder hidden states based on input and previous states
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        return output, hidden, cell