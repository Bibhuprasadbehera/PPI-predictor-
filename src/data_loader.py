import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from plots_dataloader import create_interaction_score_distribution_plot, create_rsa_distribution_plot, create_secondary_structure_distribution_plot, create_amino_acid_frequency_plot, create_sequence_length_distribution_plot, create_physicochemical_properties_distribution_plots, create_batch_visualization


class ProteinDataset(Dataset):
    def __init__(self, data_dir, phys_prop_file):
        self.data = self.load_dssp_files(data_dir)
        self.phys_props = self.load_phys_props(phys_prop_file)
        self.aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.ss_to_index = {'C': 0, 'H': 1, 'E': 2, 'P': 3}
        self.chain_to_index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'q': 50, 'r': 51, 's': 52, 'u': 53, 'w': 54}
        print(f"Loaded {len(self.data)} samples")
        self.print_sample_data()

    def load_dssp_files(self, data_dir):
        all_data = []
        for file in tqdm(os.listdir(data_dir), desc="Loading DSSP files"):
            if file.endswith('_dssp.csv'):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                required_columns = ['aa', 'rsa', 'three_hot_ss', 'test_interaction_score', 'chain']
                if all(col in df.columns for col in required_columns):
                    # Check if 'rsa' and 'test_interaction_score' can be converted to float
                    try:
                        df['rsa'] = df['rsa'].astype(float)
                        df['test_interaction_score'] = df['test_interaction_score'].astype(float)
                        all_data.append(df[required_columns])
                    except TypeError:
                        print(f"Error: Invalid data type in file {file}. 'rsa' and 'test_interaction_score' must be numeric. Skipping.")
                else:
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    print(f"Warning: File {file} is missing these required columns: {missing_columns}. Skipping.")
        if not all_data:
            raise ValueError(f"No valid data files found in {data_dir}. Please check your data files and ensure they contain the required columns.")
        return pd.concat(all_data, ignore_index=True)

    def load_phys_props(self, phys_prop_file):
        try:
            return pd.read_csv(phys_prop_file, index_col='amino acid')
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Physicochemical properties file not found: {phys_prop_file}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = str(row['aa'])

        phys_prop_list = [self.phys_props.loc[aa].values for aa in sequence]
        phys_prop_array = np.array(phys_prop_list, dtype=np.float32)
        phys_prop_tensor = torch.tensor(phys_prop_array, dtype=torch.float32)

        sequence_tensor = torch.tensor([self.aa_to_index[aa] for aa in sequence], dtype=torch.long)
        ss_tensor = torch.tensor([self.ss_to_index[s] for s in str(row['three_hot_ss'])], dtype=torch.long)
        rsa_tensor = torch.tensor([float(row['rsa'])] * len(sequence), dtype=torch.float32)
        chain_tensor = torch.tensor([self.chain_to_index[row['chain']]] * len(sequence), dtype=torch.long)  # Create chain tensor
        label_tensor = torch.tensor(float(row['test_interaction_score']), dtype=torch.float32)

        return sequence_tensor, rsa_tensor, ss_tensor, phys_prop_tensor, chain_tensor, label_tensor  # Include chain tensor

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        sequences, rsas, secondary_structures, phys_props, chains, labels = zip(*batch)  # Include chains
        max_len = max(seq.size(0) for seq in sequences)

        padded_sequences = torch.stack([torch.nn.functional.pad(seq, (0, max_len - seq.size(0))) for seq in sequences])
        padded_ss = torch.stack([torch.nn.functional.pad(ss, (0, max_len - ss.size(0))) for ss in secondary_structures])
        padded_phys_props = torch.stack([torch.nn.functional.pad(pp, (0, 0, 0, max_len - pp.size(0))) for pp in phys_props])
        padded_rsas = torch.stack([torch.nn.functional.pad(rsa, (0, max_len - rsa.size(0))) for rsa in rsas])
        padded_chains = torch.stack([torch.nn.functional.pad(chain, (0, max_len - chain.size(0))) for chain in chains])  # Pad chains

        labels = torch.stack(labels)

        return padded_sequences, padded_rsas, padded_ss, padded_phys_props, padded_chains, labels  # Include padded chains

    def print_sample_data(self):
        print("\nSample data:")
        print(self.data.head())
        print("\nData types:")
        print(self.data.dtypes)
        print("\nData statistics:")
        print(self.data.describe())

def get_data_loader(data_dir, phys_prop_file, batch_size, num_workers):
    dataset = ProteinDataset(data_dir, phys_prop_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ProteinDataset.collate_fn)

def visualize_batch(batch, num_samples=5):
    sequences, rsas, secondary_structures, phys_props, chains, labels = batch  # Unpack chains
    create_batch_visualization(batch, num_samples)
