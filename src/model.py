# src/model.py

import torch
import torch.nn as nn

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, phys_prop_size):
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(20, input_size)  # 20 amino acids
        self.ss_embedding = nn.Embedding(4, input_size)  # 4 secondary structure types
        total_input_size = input_size * 2 + 1 + phys_prop_size  # AA embedding + SS embedding + RSA + phys_props
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x, rsa, ss, phys_props):
        x = self.embedding(x)  # (batch_size, seq_length, input_size)
        ss = self.ss_embedding(ss)  # (batch_size, seq_length, input_size)
        rsa = rsa.unsqueeze(2) if rsa.dim() == 2 else rsa  # (batch_size, seq_length, 1)
        combined = torch.cat([x, ss, rsa, phys_props], dim=-1)  # (batch_size, seq_length, total_input_size)
        lstm_out, _ = self.lstm(combined)  # (batch_size, seq_length, hidden_size*2)
        output = self.fc(lstm_out)  # (batch_size, seq_length, output_size)
        return torch.sigmoid(output).squeeze(-1)  # (batch_size, seq_length)

    def __str__(self):
        return f"""ProteinInteractionModel(
    Embedding: {self.embedding}
    Secondary Structure Embedding: {self.ss_embedding}
    LSTM: {self.lstm}
    Fully Connected: {self.fc}
)"""