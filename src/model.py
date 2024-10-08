# src/model.py

import torch.nn as nn
import torch

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, phys_prop_size):
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(20, input_size)  # 20 amino acids
        self.ss_embedding = nn.Embedding(4, input_size)  # 4 secondary structure types
        total_input_size = input_size * 2 + 1 + phys_prop_size  # AA embedding + SS embedding + RSA + phys_props
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, rsa, ss, phys_props):
        # Embedding layers
        x = self.embedding(x)  # (batch_size, seq_length, 64)
        ss = self.ss_embedding(ss)  # (batch_size, seq_length, 64)
        
        # Ensure RSA has the correct shape: (batch_size, seq_length, 1)
        rsa = rsa.unsqueeze(2) if rsa.dim() == 2 else rsa  # (batch_size, seq_length, 1)

        # Concatenate along the last dimension (features)
        combined = torch.cat([x, ss, rsa, phys_props], dim=-1)  # (batch_size, seq_length, 139)
        
        assert combined.size(-1) == 139, f"Expected input size 139, but got {combined.size(-1)}"

        # Process the combined input through the LSTM
        lstm_out, _ = self.lstm(combined)  # (batch_size, seq_length, hidden_size)
        
        # Output the final time step (for each batch item)
        output = self.fc(lstm_out[:, -1, :])  # (batch_size, output_size)
        
        return torch.sigmoid(output).squeeze(-1)


    def __str__(self):
        return f"""ProteinInteractionModel(
    Embedding: {self.embedding}
    Secondary Structure Embedding: {self.ss_embedding}
    LSTM: {self.lstm}
    Fully Connected: {self.fc}
)"""