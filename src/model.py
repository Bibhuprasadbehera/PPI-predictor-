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
        
        # For prediction mode (amino acid sequence only)
        self.pred_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        print(f"Initialized ProteinInteractionModel with:")
        print(f" Input size: {input_size}")
        print(f" Hidden size: {hidden_size}")
        print(f" Number of layers: {num_layers}")
        print(f" Output size: {output_size}")

    def forward(self, x, rsa=None, ss=None):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, input_size)
        
        if rsa is not None and ss is not None:
            # Training mode
            ss = self.ss_embedding(ss)  # (batch_size, seq_length, input_size)
            rsa = rsa.unsqueeze(1).repeat(1, x.size(1)).unsqueeze(2)  # (batch_size, seq_length, 1)
            combined = torch.cat([x, ss, rsa], dim=-1)  # (batch_size, seq_length, input_size*2 + 1)
            lstm_out, _ = self.lstm(combined)
        else:
            # Prediction mode (amino acid sequence only)
            lstm_out, _ = self.pred_lstm(x)
        
        output = self.fc(lstm_out[:, -1, :])  # Use the last output of the LSTM
        return output.squeeze(-1)

    def __str__(self):
        return f"""ProteinInteractionModel(
            Embedding: {self.embedding}
            Secondary Structure Embedding: {self.ss_embedding}
            LSTM (training): {self.lstm}
            LSTM (prediction): {self.pred_lstm}
            Fully Connected: {self.fc}
        )"""