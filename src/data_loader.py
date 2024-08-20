#src/data_loader.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

class ProteinDataset(Dataset):
    def __init__(self, data_dir):
        self.data = self.load_dssp_files(data_dir)
        self.sequences = self.data['aa'].apply(list).values
        self.rsa = self.process_rsa(self.data['rsa'])
        self.labels = self.calculate_labels()

    def load_dssp_files(self, data_dir):
        all_data = []
        for file in os.listdir(data_dir):
            if file.endswith('_dssp.csv'):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                all_data.append(df[['aa', 'rsa']])
        return pd.concat(all_data, ignore_index=True)

    def process_rsa(self, rsa_series):
        def process_single_rsa(rsa):
            if isinstance(rsa, (float, int)):
                return [float(rsa)]
            elif isinstance(rsa, str):
                try:
                    return [float(x) for x in rsa.strip('[]').split(',')]
                except ValueError:
                    return [float('nan')]
            else:
                return [float('nan')]
        
        return rsa_series.apply(process_single_rsa).values

    def one_hot_encode(self, aa):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        encoding = np.zeros(20)
        if aa in amino_acids:
            encoding[amino_acids.index(aa)] = 1
        return encoding

    def calculate_labels(self):
        all_rsa = np.concatenate([np.repeat(rsa, len(seq)) for rsa, seq in zip(self.rsa, self.sequences)])
        rsa_threshold = np.nanpercentile(all_rsa, 90)
        return [np.array(seq_rsa) > rsa_threshold for seq_rsa in self.rsa]

    def __len__(self):
        return sum(len(seq) for seq in self.sequences)

    def __getitem__(self, idx):
        for seq_idx, seq_len in enumerate(map(len, self.sequences)):
            if idx < seq_len:
                break
            idx -= seq_len

        sequence = torch.tensor(self.one_hot_encode(self.sequences[seq_idx][idx]), dtype=torch.float32)
        rsa = torch.tensor([self.rsa[seq_idx][0]], dtype=torch.float32)  # Always use the first (and possibly only) RSA value
        feature = torch.cat([sequence, rsa])
        label = torch.tensor(self.labels[seq_idx][0], dtype=torch.float32)  # Use the corresponding label
        return feature, label

def get_data_loader(data_dir, batch_size, num_workers):
    dataset = ProteinDataset(data_dir)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)