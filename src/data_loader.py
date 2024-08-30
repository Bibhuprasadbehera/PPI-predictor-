# src/data_loader.py

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ProteinDataset(Dataset):
    def __init__(self, data_dir):
        self.data = self.load_dssp_files(data_dir)
        self.aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.aa_to_index['<PAD>'] = 20  # Padding token
        print(f"Loaded {len(self.data)} samples")
        self.visualize_data_distribution()
        self.print_sample_data()

    def load_dssp_files(self, data_dir):
        all_data = []
        for file in tqdm(os.listdir(data_dir), desc="Loading DSSP files"):
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
        sequence = str(row['aa'])
        padded_sequence = '<PAD>' * 3 + sequence + '<PAD>' * 3
        
        sequences = []
        rsas = []
        labels = []
        
        for i in range(3, len(padded_sequence) - 3):
            window = padded_sequence[i-3:i+4]
            sequences.append(torch.tensor([self.aa_to_index.get(aa, 20) for aa in window], dtype=torch.long))
            rsas.append(torch.tensor(float(row['rsa']), dtype=torch.float32))
            labels.append(torch.tensor(float(row['test_interaction_score']), dtype=torch.float32))
        
        return sequences, rsas, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        sequences, rsas, labels = [], [], []
        for seq_batch, rsa_batch, label_batch in batch:
            sequences.extend(seq_batch)
            rsas.extend(rsa_batch)
            labels.extend(label_batch)
        return torch.stack(sequences), torch.stack(rsas), torch.stack(labels)
    
    
    def visualize_data_distribution(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['test_interaction_score'], kde=True)
        plt.title('Distribution of Interaction Scores')
        plt.xlabel('Interaction Score')
        plt.ylabel('Count')
        plt.savefig('interaction_score_distribution.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['rsa'], kde=True)
        plt.title('Distribution of RSA Values')
        plt.xlabel('RSA')
        plt.ylabel('Count')
        plt.savefig('rsa_distribution.png')
        plt.close()

    def print_sample_data(self):
        print("\nSample data:")
        print(self.data.head())
        print("\nData types:")
        print(self.data.dtypes)
        print("\nData statistics:")
        print(self.data.describe())

def get_data_loader(data_dir, batch_size, num_workers):
    dataset = ProteinDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ProteinDataset.collate_fn)

def visualize_batch(batch, num_samples=5):
    sequences, rsas, labels = batch
    fig, axs = plt.subplots(num_samples, 2, figsize=(20, 5*num_samples))
    for i in range(num_samples):
        seq = sequences[i].numpy()
        rsa = rsas[i].numpy()
        
        # Visualize one-hot encoded sequence
        axs[i, 0].imshow(np.eye(21)[seq], aspect='auto', cmap='viridis')
        axs[i, 0].set_title(f'Sample {i+1} - Sequence (One-hot encoded)')
        axs[i, 0].set_ylabel('AA Index')
        axs[i, 0].set_xlabel('Position')
        
        axs[i, 1].plot(rsa)
        axs[i, 1].set_title(f'Sample {i+1} - RSA Values')
        axs[i, 1].set_ylabel('RSA')
        axs[i, 1].set_xlabel('Position')
   
    plt.tight_layout()
    plt.savefig('batch_visualization.png')
    plt.close()
    print(f"Batch shape - Sequences: {sequences.shape}, RSAs: {rsas.shape}, Labels: {labels.shape}")
    print(f"Label values: {labels[:num_samples]}")