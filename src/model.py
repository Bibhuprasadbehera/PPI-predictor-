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

        # DSSP processing branches
        self.dssp_embedding = nn.Embedding(20, input_size)
        self.ss_embedding = nn.Embedding(4, input_size)
        self.chain_embedding = nn.Embedding(num_chains, input_size)
        self.dssp_lstm = nn.LSTM(input_size * 3 + 1 + phys_prop_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.dssp_attention = SelfAttention(hidden_size * 2)

        # Enhanced distance matrix processing
        self.distance_conv = nn.Conv2d(in_channels=phys_prop_size + 4, out_channels=hidden_size, kernel_size=3, padding=1)
        self.distance_relu = nn.ReLU()

        # Integration and output
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, dssp_data_1, dssp_data_2, enhanced_distance_matrix, interaction_matrix):
        # Process DSSP data for protein 1
        seq_1, rsa_1, ss_1, phys_prop_1, chain_1 = dssp_data_1
        x_1 = self.dssp_embedding(seq_1)
        ss_1 = self.ss_embedding(ss_1)
        chain_1 = self.chain_embedding(chain_1)
        rsa_1 = rsa_1.unsqueeze(2)
        combined_1 = torch.cat([x_1, ss_1, rsa_1, phys_prop_1, chain_1], dim=-1)
        lstm_out_1, _ = self.dssp_lstm(combined_1)
        attention_out_1 = self.dssp_attention(lstm_out_1)

        # Process DSSP data for protein 2
        seq_2, rsa_2, ss_2, phys_prop_2, chain_2 = dssp_data_2
        x_2 = self.dssp_embedding(seq_2)
        ss_2 = self.ss_embedding(ss_2)
        chain_2 = self.chain_embedding(chain_2)
        rsa_2 = rsa_2.unsqueeze(2)
        combined_2 = torch.cat([x_2, ss_2, rsa_2, phys_prop_2, chain_2], dim=-1)
        lstm_out_2, _ = self.dssp_lstm(combined_2)
        attention_out_2 = self.dssp_attention(lstm_out_2)

        # Process enhanced distance matrix
        distance_features = self.distance_conv(enhanced_distance_matrix.permute(0, 3, 1, 2))
        distance_features = self.distance_relu(distance_features)
        distance_features = torch.mean(distance_features, dim=(-1, -2))

        # Integrate features
        combined = torch.cat([attention_out_1.mean(dim=1), attention_out_2.mean(dim=1), distance_features], dim=-1)

        # Output layer
        out = self.fc1(combined)
        out = self.dropout(out)
        output = self.fc2(out)
        return torch.sigmoid(output)

    def __str__(self):
        return (f"ProteinInteractionModel(\n"
                f"  DSSP Embedding: {self.dssp_embedding}\n"
                f"  Secondary Structure Embedding: {self.ss_embedding}\n"
                f"  Chain Embedding: {self.chain_embedding}\n"
                f"  DSSP LSTM: {self.dssp_lstm}\n"
                f"  DSSP Attention: {self.dssp_attention}\n"
                f"  Distance Conv: {self.distance_conv}\n"
                f"  Distance ReLU: {self.distance_relu}\n"
                f"  Fully Connected 1: {self.fc1}\n"
                f"  Dropout: {self.dropout}\n"
                f"  Fully Connected 2: {self.fc2}\n)")
