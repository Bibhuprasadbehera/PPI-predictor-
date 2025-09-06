# src/model.py
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        context_vector = torch.bmm(attention_weights, values)

        return context_vector

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, phys_prop_size, num_chains=54):
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(20, input_size)
        self.ss_embedding = nn.Embedding(4, input_size)
        self.chain_embedding = nn.Embedding(num_chains, input_size)
        self.distance_mat_embedding = nn.Conv2d(in_channels=1, out_channels=input_size, kernel_size=1)
        total_input_size = input_size * 4 + 1 + phys_prop_size
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = SelfAttention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, rsa, ss, phys_props, chains, distance_mat):
        x = self.embedding(x)
        ss = self.ss_embedding(ss)
        chains = self.chain_embedding(chains)
        
        if distance_mat is not None:
            distance_mat = distance_mat.unsqueeze(1)  # Shape: (B, 1, L, L)
            distance_mat = self.distance_mat_embedding(distance_mat)  # Shape: (B, input_size, L, L)
            distance_mat = distance_mat.squeeze(1)  # Shape: (B, input_size, L, L)
            distance_mat = distance_mat.permute(0, 2, 3, 1)  # Shape: (B, L, L, input_size)
            distance_mat = distance_mat.mean(dim=2)  # Shape: (B, L, input_size)
        else:
            distance_mat = torch.zeros(x.size(0), x.size(1), self.distance_mat_embedding.out_channels, device=x.device)
            distance_mat = distance_mat.permute(0, 2, 1)  # Shape: (B, L, input_size)
        
        rsa = rsa.unsqueeze(2)  # Shape: (B, L, 1)
        phys_props = phys_props.unsqueeze(2) if phys_props.dim() == 2 else phys_props  # Shape: (B, L, phys_prop_size)
        
        combined = torch.cat([x, ss, rsa, phys_props, chains, distance_mat], dim=2)
        
        lstm_out, _ = self.lstm(combined)
        attention_out = self.attention(lstm_out)
        out = self.fc1(attention_out)
        out = self.dropout(out)
        output = self.fc2(out)
        
        return output  # Shape: (B, L, L)

    def __str__(self):
        return (f"ProteinInteractionModel(\n"
                f"  Embedding: {self.embedding}\n"
                f"  Secondary Structure Embedding: {self.ss_embedding}\n"
                f"  Chain Embedding: {self.chain_embedding}\n"  # Add chain embedding to string representation
                f"  LSTM: {self.lstm}\n"
                f"  Attention: {self.attention}\n"
                f"  Fully Connected 1: {self.fc1}\n"
                f"  Dropout: {self.dropout}\n"
                f"  Fully Connected 2: {self.fc2}\n)")
