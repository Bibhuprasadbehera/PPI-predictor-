# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_size, hidden_size, num_layers, phys_prop_size, num_chains=54, motif_feature_size=4):
        super(ProteinInteractionModel, self).__init__()
        # Embedding layers
        self.aa_embedding = nn.Embedding(22, input_size)  # 20 amino acids + J, B
        self.ss_embedding = nn.Embedding(4, input_size)   # C, H, E, P
        self.chain_embedding = nn.Embedding(num_chains, input_size)
        
        # Total input size after combining all features
        total_input_size = input_size + 1 + input_size + phys_prop_size + input_size + motif_feature_size
        
        # LSTM layer
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=0.3, bidirectional=True)
        
        # Attention mechanism
        self.attention = SelfAttention(hidden_size * 2)
        
        # Dense layers for processing individual sequence features
        self.interaction_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Output layer for individual position features: predicts 1 value per position
        self.individual_output = nn.Linear(hidden_size // 2, 1)  # Output 1 value per position
        
        # Output layer for pairwise interactions: takes concatenated features from pair
        self.pairwise_output = nn.Linear(3, 1)  # Takes 3 features (pos_i, pos_j, |pos_i - pos_j|)
        
        self.motif_feature_size = motif_feature_size
        self.hidden_size = hidden_size

    def forward(self, x, rsa, ss, phys_props, chains, motif_binary, motif_index, motif_position, motif_overlap, use_all_features=True, sequence2=None, rsa2=None, ss2=None, phys_props2=None, chains2=None, motif_binary2=None, motif_index2=None, motif_position2=None, motif_overlap2=None):
        batch_size, seq_len = x.size()
        
        # Process first sequence
        x_embed = self.aa_embedding(x)  # (B, L1, input_size)
        
        # Reshape other inputs to add dimension if needed for sequence 1
        rsa = rsa.unsqueeze(-1) if rsa.dim() == 2 else rsa  # (B, L1, 1)
        ss_embed = self.ss_embedding(ss)  # (B, L1, input_size)
        chains_embed = self.chain_embedding(chains)  # (B, L1, input_size)
        
        # Process motif features with normalization for sequence 1
        motif_binary = motif_binary.unsqueeze(-1)  # (B, L1, 1)
        
        # Normalize motif features separately for sequence 1
        motif_index_normalized = self.normalize_feature(motif_index.float())  # (B, L1)
        motif_position_normalized = self.normalize_feature(motif_position.float())  # (B, L1)
        motif_overlap_normalized = self.normalize_feature(motif_overlap.float())  # (B, L1)
        
        # Stack motif features for sequence 1
        motif_features = torch.stack([motif_binary.squeeze(-1), 
                                      motif_index_normalized, 
                                      motif_position_normalized, 
                                      motif_overlap_normalized], dim=-1)  # (B, L1, 4)
        
        # For inference, use only sequence features if specified for sequence 1
        if not use_all_features:
            # Use default/learned values for missing features for sequence 1
            combined1 = torch.cat([
                x_embed,
                rsa,
                torch.zeros_like(ss_embed),
                torch.zeros_like(phys_props),
                torch.zeros_like(chains_embed),
                torch.zeros_like(motif_features)
            ], dim=-1)
        else:
            combined1 = torch.cat([
                x_embed,        # (B, L1, input_size)
                rsa,            # (B, L1, 1)
                ss_embed,       # (B, L1, input_size)
                phys_props,     # (B, L1, phys_prop_size)
                chains_embed,   # (B, L1, input_size)
                motif_features  # (B, L1, motif_feature_size)
            ], dim=-1)  # (B, L1, total_input_size)
        
        # Process second sequence if provided
        if sequence2 is not None:
            x2_embed = self.aa_embedding(sequence2)  # (B, L2, input_size)
            
            # Reshape other inputs for sequence 2
            rsa2 = rsa2.unsqueeze(-1) if rsa2.dim() == 2 else rsa2  # (B, L2, 1)
            ss2_embed = self.ss_embedding(ss2)  # (B, L2, input_size)
            chains2_embed = self.chain_embedding(chains2)  # (B, L2, input_size)
            
            # Process motif features with normalization for sequence 2
            motif_binary2 = motif_binary2.unsqueeze(-1)  # (B, L2, 1)
            
            # Normalize motif features separately for sequence 2
            motif_index2_normalized = self.normalize_feature(motif_index2.float())  # (B, L2)
            motif_position2_normalized = self.normalize_feature(motif_position2.float())  # (B, L2)
            motif_overlap2_normalized = self.normalize_feature(motif_overlap2.float())  # (B, L2)
            
            # Stack motif features for sequence 2
            motif_features2 = torch.stack([motif_binary2.squeeze(-1), 
                                          motif_index2_normalized, 
                                          motif_position2_normalized, 
                                          motif_overlap2_normalized], dim=-1)  # (B, L2, 4)
            
            # For inference, use only sequence features if specified for sequence 2
            if not use_all_features:
                # Use default/learned values for missing features for sequence 2
                combined2 = torch.cat([
                    x2_embed,
                    rsa2,
                    torch.zeros_like(ss2_embed),
                    torch.zeros_like(phys_props2),
                    torch.zeros_like(chains2_embed),
                    torch.zeros_like(motif_features2)
                ], dim=-1)
            else:
                combined2 = torch.cat([
                    x2_embed,        # (B, L2, input_size)
                    rsa2,            # (B, L2, 1)
                    ss2_embed,       # (B, L2, input_size)
                    phys_props2,     # (B, L2, phys_prop_size)
                    chains2_embed,   # (B, L2, input_size)
                    motif_features2  # (B, L2, motif_feature_size)
                ], dim=-1)  # (B, L2, total_input_size)
        else:
            # If no second sequence, use the first sequence for both (intra-protein interactions)
            combined2 = combined1
        
        # Pass through LSTM for sequence 1
        lstm_out1, _ = self.lstm(combined1)
        attention_out1 = self.attention(lstm_out1)  # (B, L1, hidden_size*2)
        interaction_features1 = self.interaction_fc(attention_out1)  # (B, L1, hidden_size//2)
        pos_repr1 = self.individual_output(interaction_features1)  # (B, L1, 1)
        pos_repr1 = pos_repr1.squeeze(-1)  # (B, L1)
        
        # Pass through LSTM for sequence 2
        if sequence2 is not None:
            lstm_out2, _ = self.lstm(combined2)
            attention_out2 = self.attention(lstm_out2)  # (B, L2, hidden_size*2)
            interaction_features2 = self.interaction_fc(attention_out2)  # (B, L2, hidden_size//2)
            pos_repr2 = self.individual_output(interaction_features2)  # (B, L2, 1)
            pos_repr2 = pos_repr2.squeeze(-1)  # (B, L2)
        else:
            # If no second sequence, use same representations (for intra-protein)
            pos_repr2 = pos_repr1
        
        # Create cross-sequence interaction matrix
        pos_repr1_expanded = pos_repr1.unsqueeze(-1).expand(-1, -1, pos_repr2.size(-1))  # (B, L1, L2)
        pos_repr2_expanded = pos_repr2.unsqueeze(1).expand(-1, pos_repr1.size(-1), -1)   # (B, L1, L2)
        
        # Combine features for each pair (i from seq1, j from seq2) to create interaction prediction
        pair_features = torch.cat([
            pos_repr1_expanded.unsqueeze(-1),  # (B, L1, L2, 1)
            pos_repr2_expanded.unsqueeze(-1),  # (B, L1, L2, 1)
            torch.abs(pos_repr1_expanded - pos_repr2_expanded).unsqueeze(-1)  # (B, L1, L2, 1) - difference
        ], dim=-1)  # (B, L1, L2, 3)
        
        # Reshape for processing - flatten the spatial dimensions
        batch_size, seq_len1, seq_len2, feat_dim = pair_features.shape
        pair_features_reshaped = pair_features.view(-1, feat_dim)  # (B*L1*L2, 3)
        
        # Apply final transformation to get single value per pair
        interaction_values = self.pairwise_output(pair_features_reshaped).squeeze(-1)  # (B*L1*L2,)
        
        # Reshape back to (B, L1, L2)
        interaction_matrix = interaction_values.view(batch_size, seq_len1, seq_len2)  # (B, L1, L2)
        
        # Apply sigmoid to constrain output to [0,1] range for interaction probability
        interaction_matrix = torch.sigmoid(interaction_matrix)
        
        return interaction_matrix

    def normalize_feature(self, feature):
        """Normalize a feature tensor to [0, 1] range"""
        # Calculate min and max per sample in the batch (across sequence dimension)
        # feature shape is (B, L) where B is batch size and L is sequence length
        feature_min = torch.min(feature, dim=-1, keepdim=True)[0]  # (B, 1)
        feature_max = torch.max(feature, dim=-1, keepdim=True)[0]  # (B, 1)
        range_val = feature_max - feature_min  # (B, 1)
        # Avoid division by zero
        range_val = torch.clamp(range_val, min=1e-7)
        normalized = (feature - feature_min) / range_val
        return normalized

    def __str__(self):
        return (f"ProteinInteractionModel(\n"
                f"  AA Embedding: {self.aa_embedding}\n"
                f"  SS Embedding: {self.ss_embedding}\n"
                f"  Chain Embedding: {self.chain_embedding}\n"
                f"  LSTM: {self.lstm}\n"
                f"  Attention: {self.attention}\n"
                f"  Interaction FC: {self.interaction_fc}\n"
                f"  Individual Output: {self.individual_output}\n"
                f"  Pairwise Output: {self.pairwise_output}\n"
                f"  Motif Feature Size: {self.motif_feature_size}\n)")
