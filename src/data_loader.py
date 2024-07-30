#src/data_loader.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

class ProteinDataset(Dataset):
    def __init__(self, data_dir):
        self.data = self.load_dssp_files(data_dir)
        self.sequences = self.data['aa'].apply(self.one_hot_encode).values
        self.rsa = self.data['rsa'].values
        self.labels = self.calculate_labels()

    def load_dssp_files(self, data_dir):
        all_data = []
        for file in os.listdir(data_dir):
            if file.endswith('_dssp.csv'):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                all_data.append(df[['aa', 'rsa']])  # Only select 'aa' and 'rsa' columns
        return pd.concat(all_data, ignore_index=True)

    def one_hot_encode(self, sequence):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
        encoding = np.zeros(20)
        if sequence in aa_to_idx:
            encoding[aa_to_idx[sequence]] = 1
        return encoding

    def calculate_labels(self):
        rsa_threshold = np.percentile(self.rsa, 90)
        return (self.rsa > rsa_threshold).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        rsa = torch.tensor([self.rsa[idx]], dtype=torch.float32)
        feature = torch.cat([sequence, rsa])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature.unsqueeze(0), label  # Add batch dimension to feature

def get_data_loader(data_dir, batch_size, num_workers):
    dataset = ProteinDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)