# src/data_loader.py

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class ProteinDataset(Dataset):
    def __init__(self, data_dir, phys_prop_file):
        self.data = self.load_dssp_files(data_dir)
        self.phys_props = self.load_phys_props(phys_prop_file)
        self.aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.ss_to_index = {'C': 0, 'H': 1, 'E': 2, 'P': 3}
        print(f"Loaded {len(self.data)} samples")
        self.visualize_data_distribution()
        self.print_sample_data()
        self.plot_physicochemical_properties_distribution(phys_prop_file)

    def load_dssp_files(self, data_dir):
        all_data = []
        for file in tqdm(os.listdir(data_dir), desc="Loading DSSP files"):
            if file.endswith('_dssp.csv'):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                required_columns = ['aa', 'rsa', 'three_hot_ss', 'test_interaction_score']
                if all(col in df.columns for col in required_columns):
                    all_data.append(df[required_columns])
                else:
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    print(f"Warning: File {file} is missing these required columns: {missing_columns}. Skipping.")
        if not all_data:
            raise ValueError(f"No valid data files found in {data_dir}. Please check your data files and ensure they contain the required columns.")
        return pd.concat(all_data, ignore_index=True)

    def load_phys_props(self, phys_prop_file):
        return pd.read_csv(phys_prop_file, index_col='amino acid')

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = str(row['aa'])

        phys_prop_list = [self.phys_props.loc[aa].values for aa in sequence]
        phys_prop_array = np.array(phys_prop_list, dtype=np.float32)
        phys_prop_tensor = torch.tensor(phys_prop_array, dtype=torch.float32)

        sequence_tensor = torch.tensor([self.aa_to_index[aa] for aa in sequence], dtype=torch.long)
        ss_tensor = torch.tensor([self.ss_to_index[s] for s in str(row['three_hot_ss'])], dtype=torch.long)
        rsa_tensor = torch.tensor([float(row['rsa'])] * len(sequence), dtype=torch.float32)
        label_tensor = torch.tensor(float(row['test_interaction_score']), dtype=torch.float32)

        return sequence_tensor, rsa_tensor, ss_tensor, phys_prop_tensor, label_tensor

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        sequences, rsas, secondary_structures, phys_props, labels = zip(*batch)
        max_len = max(seq.size(0) for seq in sequences)

        padded_sequences = torch.stack([torch.nn.functional.pad(seq, (0, max_len - seq.size(0))) for seq in sequences])
        padded_ss = torch.stack([torch.nn.functional.pad(ss, (0, max_len - ss.size(0))) for ss in secondary_structures])
        padded_phys_props = torch.stack([torch.nn.functional.pad(pp, (0, 0, 0, max_len - pp.size(0))) for pp in phys_props])
        padded_rsas = torch.stack([torch.nn.functional.pad(rsa, (0, max_len - rsa.size(0))) for rsa in rsas])

        labels = torch.stack(labels)

        return padded_sequences, padded_rsas, padded_ss, padded_phys_props, labels

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

        plt.figure(figsize=(12, 6))
        ss_counts = self.data['three_hot_ss'].apply(lambda x: ''.join(set(x))).value_counts()
        sns.barplot(x=ss_counts.index, y=ss_counts.values)
        plt.title('Distribution of Secondary Structures')
        plt.xlabel('Secondary Structure')
        plt.ylabel('Count')
        plt.savefig('secondary_structure_distribution.png')
        plt.close()

        self.plot_amino_acid_frequency()
        self.plot_sequence_length_distribution()

    def print_sample_data(self):
        print("\nSample data:")
        print(self.data.head())
        print("\nData types:")
        print(self.data.dtypes)
        print("\nData statistics:")
        print(self.data.describe())

    def plot_physicochemical_properties_distribution(self, phys_prop_file):
        # Load the physicochemical properties data
        data = pd.read_csv(phys_prop_file)

        # Create distribution plots for each property
        for property in data.columns[1:]:  # Skip the first column (amino acid)
            plt.figure(figsize=(8, 6))
            sns.histplot(data[property], kde=True)
            plt.title(f'Distribution of {property}')
            plt.xlabel(property)
            plt.ylabel('Frequency')
            plt.savefig(f'physicochemical_properties_distribution/{property}_distribution.png')
            plt.close()

        print("Distribution plots for physicochemical properties generated successfully.")

    def plot_amino_acid_frequency(self):
        plt.figure(figsize=(12, 6))
        amino_acid_counts = self.data['aa'].value_counts()
        sns.barplot(x=amino_acid_counts.index, y=amino_acid_counts.values)
        plt.title('Amino Acid Frequency')
        plt.xlabel('Amino Acid')
        plt.ylabel('Count')
        plt.savefig('amino_acid_frequency.png')
        plt.close()

    def plot_sequence_length_distribution(self):
        plt.figure(figsize=(12, 6))
        sequence_lengths = self.data['aa'].str.len()
        sns.histplot(sequence_lengths, kde=True)
        plt.title('Sequence Length Distribution')
        plt.xlabel('Sequence Length')
        plt.ylabel('Count')
        plt.savefig('sequence_length_distribution.png')
        plt.close()

def get_data_loader(data_dir, phys_prop_file, batch_size, num_workers):
    dataset = ProteinDataset(data_dir, phys_prop_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ProteinDataset.collate_fn)

def visualize_batch(batch, num_samples=5):
    sequences, rsas, secondary_structures, phys_props, labels = batch
    fig, axs = plt.subplots(num_samples, 3, figsize=(20, 5*num_samples))
    for i in range(num_samples):
        seq = sequences[i].numpy()
        rsa = rsas[i].numpy()
        ss = secondary_structures[i].numpy()

        axs[i, 0].imshow(np.eye(20)[seq], aspect='auto', cmap='viridis')
        axs[i, 0].set_title(f'Sample {i+1} - Sequence (One-hot encoded)')
        axs[i, 0].set_ylabel('AA Index')
        axs[i, 0].set_xlabel('Position')

        axs[i, 1].plot(rsa.repeat(len(seq)))
        axs[i, 1].set_title(f'Sample {i+1} - RSA Value')
        axs[i, 1].set_ylabel('RSA')
        axs[i, 1].set_xlabel('Position')

        axs[i, 2].imshow(np.eye(4)[ss], aspect='auto', cmap='viridis')
        axs[i, 2].set_title(f'Sample {i+1} - Secondary Structure')
        axs[i, 2].set_ylabel('SS Index')
        axs[i, 2].set_xlabel('Position')

    plt.tight_layout()
    plt.savefig('batch_visualization.png')
    plt.close()
    print(f"Batch shape - Sequences: {sequences.shape}, RSAs: {rsas.shape}, Secondary Structures: {secondary_structures.shape}, Physicochemical Properties: {phys_props.shape}, Labels: {labels.shape}")
    print(f"Label values: {labels[:num_samples]}")
