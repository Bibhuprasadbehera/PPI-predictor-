# src/model.py
import torch.nn as nn
import torch

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(21, input_size)  # 20 amino acids + padding
        self.lstm = nn.LSTM(input_size + 1, hidden_size, num_layers, batch_first=True)  # +1 for RSA
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, rsa=None):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, input_size)
        
        if rsa is not None:
            # During training, concatenate RSA to each amino acid embedding
            rsa = rsa.unsqueeze(-1)
            x = torch.cat([x, rsa], dim=-1)
        
        lstm_out, _ = self.lstm(x)
        # Use the last output of the LSTM
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)

    def predict(self, x):
        # For prediction, we only use the AA sequence
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)