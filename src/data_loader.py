#src/data_loader.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os

class ProteinDataset(Dataset):
    def __init__(self, data_dir):
        self.data = self.load_dssp_files(data_dir)
        self.aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.aa_to_index['<PAD>'] = 20  # Padding token
        print(f"Loaded {len(self.data)} samples")

    def load_dssp_files(self, data_dir):
        all_data = []
        for file in os.listdir(data_dir):
            if file.endswith('_dssp.csv'):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                if 'aa' not in df.columns or 'rsa' not in df.columns or 'test_interaction_score' not in df.columns:
                    print(f"Warning: File {file} is missing required columns. Skipping.")
                    continue
                all_data.append(df[['aa', 'rsa', 'test_interaction_score']])
        if not all_data:
            raise ValueError(f"No valid data files found in {data_dir}")
        return pd.concat(all_data, ignore_index=True)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = torch.tensor([self.aa_to_index.get(aa, 20) for aa in str(row['aa'])], dtype=torch.long)
        rsa = torch.tensor(float(row['rsa']), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(float(row['test_interaction_score']), dtype=torch.float32)
        return sequence, rsa, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        sequences, rsas, labels = zip(*batch)
        sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=20)
        rsas_padded = nn.utils.rnn.pad_sequence(rsas, batch_first=True, padding_value=0)
        return sequences_padded, rsas_padded, torch.tensor(labels)

def get_data_loader(data_dir, batch_size, num_workers):
    dataset = ProteinDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ProteinDataset.collate_fn)