import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: (batch_size, seq_length, hidden_size)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        context_vector = torch.bmm(attention_weights, values)

        return context_vector

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, phys_prop_size):
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(20, input_size)  # 20 amino acids
        self.ss_embedding = nn.Embedding(4, input_size)  # 4 secondary structure types
        total_input_size = input_size * 2 + 1 + phys_prop_size  # AA embedding + SS embedding + RSA + phys_props
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = SelfAttention(hidden_size * 2)  # Added attention layer
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # Added a fully connected layer
        self.dropout = nn.Dropout(0.3)  # Added dropout
        self.fc2 = nn.Linear(hidden_size, output_size)  # *2 for bidirectional

    def forward(self, x, rsa, ss, phys_props):
        x = self.embedding(x)  # (batch_size, seq_length, input_size)
        ss = self.ss_embedding(ss)  # (batch_size, seq_length, input_size)
        rsa = rsa.unsqueeze(2) if rsa.dim() == 2 else rsa  # (batch_size, seq_length, 1)
        combined = torch.cat([x, ss, rsa, phys_props], dim=-1)  # (batch_size, seq_length, total_input_size)
        lstm_out, _ = self.lstm(combined)  # (batch_size, seq_length, hidden_size*2)
        attention_out = self.attention(lstm_out)  # Pass through attention layer
        out = self.fc1(attention_out)  # Pass through the first fully connected layer
        out = self.dropout(out)  # Apply dropout
        output = self.fc2(out)  # Pass through the second fully connected layer
        return torch.sigmoid(output).squeeze(-1)  # (batch_size, seq_length)

    def __str__(self):
        return f"""ProteinInteractionModel(
    Embedding: {self.embedding}
    Secondary Structure Embedding: {self.ss_embedding}
    LSTM: {self.lstm}
    Attention: {self.attention}
    Fully Connected 1: {self.fc1}
    Dropout: {self.dropout}
    Fully Connected 2: {self.fc2}
)"""
