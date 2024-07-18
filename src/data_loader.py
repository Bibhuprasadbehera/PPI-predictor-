# src/data_loader.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class ProteinDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.sequences = self.data['sequence'].apply(self.one_hot_encode).values
        self.labels = self.data['label'].values

    def one_hot_encode(self, sequence):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
        encoding = np.zeros((len(sequence), 21))
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                encoding[i, aa_to_idx[aa]] = 1
            else:
                encoding[i, -1] = 1  # Padding for unknown amino acids
        return encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence, label

def get_data_loader(file_path, batch_size, num_workers):
    dataset = ProteinDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
