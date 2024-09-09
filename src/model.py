# src/model.py

import torch.nn as nn
import torch

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(20, input_size)  # 20 amino acids
        self.ss_embedding = nn.Embedding(4, input_size)  # 4 secondary structure types
        self.lstm = nn.LSTM(input_size * 2 + 1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        print(f"Initialized ProteinInteractionModel with:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Output size: {output_size}")

    def forward(self, x, rsa, ss):
        # x shape: (batch_size, seq_length)
        # ss shape: (batch_size, seq_length)
        # rsa shape: (batch_size)
        x = self.embedding(x)  # (batch_size, seq_length, input_size)
        ss = self.ss_embedding(ss)  # (batch_size, seq_length, input_size)
        
        # Repeat RSA for each position in the sequence
        rsa = rsa.unsqueeze(1).repeat(1, x.size(1)).unsqueeze(2)  # (batch_size, seq_length, 1)
        
        # Concatenate amino acid embeddings, secondary structure embeddings, and RSA
        combined = torch.cat([x, ss, rsa], dim=-1)  # (batch_size, seq_length, input_size*2 + 1)
        
        lstm_out, _ = self.lstm(combined)
        output = self.fc(lstm_out[:, -1, :])  # Use the last output of the LSTM
        return output.squeeze(-1)

    def __str__(self):
        return f"""ProteinInteractionModel(
    Embedding: {self.embedding}
    Secondary Structure Embedding: {self.ss_embedding}
    LSTM: {self.lstm}
    Fully Connected: {self.fc}
)"""