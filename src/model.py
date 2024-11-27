import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (x.size(-1) ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Mask padding tokens

        attention_weights = torch.softmax(scores, dim=-1)
        context_vector = torch.bmm(attention_weights, values)

        return context_vector

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, phys_prop_size, num_chains=26):  # Add num_chains parameter
        super(ProteinInteractionModel, self).__init__()
        self.embedding = nn.Embedding(20, input_size)
        self.ss_embedding = nn.Embedding(4, input_size)
        self.chain_embedding = nn.Embedding(num_chains, input_size)  # Add chain embedding layer
        total_input_size = input_size * 3 + 1 + phys_prop_size  # Adjust input size
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = SelfAttention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # Updated input size for fc1
        self.dropout = nn.Dropout(0.3)
        # self.fc2 = nn.Linear(hidden_size, output_size)  # Removed fc2

    def forward(self, x1, rsa1, ss1, phys_props1, chains1, x2, rsa2, ss2, phys_props2, chains2):  # Add parameters for the second sequence
        # Pad sequences to the same length
        max_len = max(x1.size(1), x2.size(1))
        x1 = nn.functional.pad(x1, (0, 0, 0, max_len - x1.size(1)))
        rsa1 = nn.functional.pad(rsa1, (0, max_len - rsa1.size(1)))
        ss1 = nn.functional.pad(ss1, (0, max_len - ss1.size(1)))
        phys_props1 = nn.functional.pad(phys_props1, (0, 0, 0, max_len - phys_props1.size(1)))
        chains1 = nn.functional.pad(chains1, (0, max_len - chains1.size(1)))

        x2 = nn.functional.pad(x2, (0, 0, 0, max_len - x2.size(1)))
        rsa2 = nn.functional.pad(rsa2, (0, max_len - rsa2.size(1)))
        ss2 = nn.functional.pad(ss2, (0, max_len - ss2.size(1)))
        phys_props2 = nn.functional.pad(phys_props2, (0, 0, 0, max_len - phys_props2.size(1)))
        chains2 = nn.functional.pad(chains2, (0, max_len - chains2.size(1)))

        # Create masks for padding tokens
        mask1 = (x1 != 0).unsqueeze(1).unsqueeze(2)
        mask2 = (x2 != 0).unsqueeze(1).unsqueeze(2)

        x1 = self.embedding(x1)
        ss1 = self.ss_embedding(ss1)
        chains1 = self.chain_embedding(chains1)  # Embed chain IDs
        rsa1 = rsa1.unsqueeze(2) if rsa1.dim() == 2 else rsa1
        combined1 = torch.cat([x1, ss1, rsa1, phys_props1, chains1], dim=-1)  # Concatenate chain embeddings

        x2 = self.embedding(x2)
        ss2 = self.ss_embedding(ss2)
        chains2 = self.chain_embedding(chains2)  # Embed chain IDs
        rsa2 = rsa2.unsqueeze(2) if rsa2.dim() == 2 else rsa2
        combined2 = torch.cat([x2, ss2, rsa2, phys_props2, chains2], dim=-1)  # Concatenate chain embeddings

        lstm_out1, _ = self.lstm(combined1)
        attention_out1 = self.attention(lstm_out1, mask1)
        
        lstm_out2, _ = self.lstm(combined2)
        attention_out2 = self.attention(lstm_out2, mask2)

        # Calculate interaction matrix using dot product
        interaction_matrix = torch.bmm(attention_out1, attention_out2.transpose(1, 2))

        # Return the individual predictions and the interaction matrix
        return torch.sigmoid(attention_out1).squeeze(-1), torch.sigmoid(attention_out2).squeeze(-1), interaction_matrix

    def __str__(self):
        return (f"ProteinInteractionModel(\n"
                f"  Embedding: {self.embedding}\n"
                f"  Secondary Structure Embedding: {self.ss_embedding}\n"
                f"  Chain Embedding: {self.chain_embedding}\n"  # Add chain embedding to string representation
                f"  LSTM: {self.lstm}\n"
                f"  Attention: {self.attention}\n"
                f"  Fully Connected 1: {self.fc1}\n"
                f"  Dropout: {self.dropout}\n"
                # f"  Fully Connected 2: {self.fc2}\n)")  # Removed fc2 from string representation
                f")")
