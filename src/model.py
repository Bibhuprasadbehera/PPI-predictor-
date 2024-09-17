# src/model.py

# src/model.py
import torch.nn as nn
import torch

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(20, input_size)  # 20 amino acids
        self.ss_embedding = nn.Embedding(4, input_size)  # 4 secondary structure types
        self.lstm = nn.LSTM(input_size * 2 + 1, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # For prediction mode (amino acid sequence only)
        self.pred_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
    def forward(self, x, rsa=None, ss=None):
        x = self.embedding(x)  # (batch_size, seq_length, input_size)
        
        if rsa is not None and ss is not None:
            # Training mode
            ss = self.ss_embedding(ss)
            rsa = rsa.unsqueeze(1).repeat(1, x.size(1)).unsqueeze(2)
            combined = torch.cat([x, ss, rsa], dim=-1)
            lstm_out, _ = self.lstm(combined)
        else:
            # Prediction mode
            lstm_out, _ = self.pred_lstm(x)
        
        # Apply sigmoid to constrain the output between 0 and 1
        output = self.fc(lstm_out[:, -1, :])
        return torch.sigmoid(output).squeeze(-1)

    def __str__(self):
        return f"""ProteinInteractionModel(
    Embedding: {self.embedding}
    Secondary Structure Embedding: {self.ss_embedding}
    LSTM: {self.lstm}
    Fully Connected: {self.fc}
)"""