# src/model.py

import torch.nn as nn
import torch

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(21, input_size)  # 20 amino acids + padding
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + 1, output_size)  # +1 for RSA
        
        print(f"Initialized ProteinInteractionModel with:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Output size: {output_size}")
    
    def forward(self, x, rsa):
        # x shape: (batch_size, window_size)
        x = self.embedding(x)  # (batch_size, window_size, input_size)
        
        lstm_out, _ = self.lstm(x)
        # Use the output corresponding to the middle amino acid
        middle_output = lstm_out[:, 3, :]
        
        # Concatenate RSA to the middle output
        combined = torch.cat([middle_output, rsa.unsqueeze(1)], dim=1)
        
        output = self.fc(combined)
        return output.squeeze(-1)

    def __str__(self):
        return f"""ProteinInteractionModel(
  Embedding: {self.embedding}
  LSTM: {self.lstm}
  Fully Connected: {self.fc}
)"""