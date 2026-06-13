import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super(BiLSTMClassifier, self).__init__()
        # Use padding_idx=0 to ensure the model ignores padding tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bi-directional LSTM
        # num_layers=2 adds depth to the model for better feature extraction
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=2, 
                           bidirectional=True, 
                           batch_first=True,
                           dropout=0.5)
        
        # Dense layers: hidden_dim * 2 because LSTM is bidirectional
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))
        
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        # hidden shape is [num_layers * num_directions, batch_size, hidden_dim]
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        dense = torch.relu(self.fc(hidden_cat))
        return self.sigmoid(self.out(self.dropout(dense)))