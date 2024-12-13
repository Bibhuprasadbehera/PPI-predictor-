import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
from sklearn.model_selection import train_test_split

class ProteinDataset(Dataset):
    def __init__(self, csv_file, complex_dssp_dir, distance_matrix_dir, phys_prop_file):
        self.data = pd.read_csv(csv_file)
        self.complex_dssp_dir = complex_dssp_dir
        self.distance_matrix_dir = distance_matrix_dir
        self.phys_props = self.load_phys_props(phys_prop_file)
        self.aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.ss_to_index = {'C': 0, 'H': 1, 'E': 2, 'P': 3}
        self.chain_to_index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'q': 50, 'r': 51, 's': 52, 'u': 53, 'w': 54}

        # Filter out entries with missing files
        self.data = self.filter_missing_files()

    def filter_missing_files(self):
        filtered_data = []
        for index, row in self.data.iterrows():
            chain_1_id = row['chain_1_id']
            chain_2_id = row['chain_2_id']
            pdb_id_2 = row['pdb_id_2']

            dssp_1_path = os.path.join(self.complex_dssp_dir, 'protein_1', f"{chain_1_id}_dssp.csv")
            dssp_2_path = os.path.join(self.complex_dssp_dir, 'protein_2', f"{chain_2_id}_dssp.csv")
            distance_matrix_path = os.path.join(self.distance_matrix_dir, f"{pdb_id_2}_ca.tsv")

            if os.path.exists(dssp_1_path) and os.path.exists(dssp_2_path) and os.path.exists(distance_matrix_path):
                filtered_data.append(row)

        return pd.DataFrame(filtered_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            chain_1_id = row['chain_1_id']
            chain_2_id = row['chain_2_id']
            pdb_id_2 = row['pdb_id_2']

            # Load DSSP data for protein 1
            dssp_1_path = os.path.join(self.complex_dssp_dir, 'protein_1', f"{chain_1_id}_dssp.csv")
            dssp_1_data = self.load_dssp(dssp_1_path)

            # Load DSSP data for protein 2
            dssp_2_path = os.path.join(self.complex_dssp_dir, 'protein_2', f"{chain_2_id}_dssp.csv")
            dssp_2_data = self.load_dssp(dssp_2_path)

            # Load distance matrix
            distance_matrix_path = os.path.join(self.distance_matrix_dir, f"{pdb_id_2}_ca.tsv")
            distance_matrix = self.load_distance_matrix(distance_matrix_path)

            # Create binary interaction matrix
            interaction_matrix = (distance_matrix < 5).astype(int)

            # Map DSSP data to distance matrix
            enhanced_distance_matrix = self.map_dssp_to_distance_matrix(dssp_1_data, dssp_2_data, distance_matrix, interaction_matrix)

            if dssp_1_data is None or dssp_2_data is None or distance_matrix is None:
                return None

            return dssp_1_data, dssp_2_data, enhanced_distance_matrix, interaction_matrix
        except Exception as e:
            print(f"Error processing item at index {idx}: {e}")
            return None

    def load_dssp(self, dssp_path):
        # Load DSSP data from CSV
        try:
            dssp_data = pd.read_csv(dssp_path)
            # Ensure required columns are present
            required_columns = ['aa', 'rsa', 'three_hot_ss', 'chain']
            if not all(col in dssp_data.columns for col in required_columns):
                missing_columns = [col for col in required_columns if col not in dssp_data.columns]
                print(f"Error: DSSP file {dssp_path} is missing required columns: {missing_columns}")
                return None

            # Extract sequence, RSA, and secondary structure
            sequence = dssp_data['aa'].apply(lambda aa: self.aa_to_index.get(aa, -1)).tolist()  # Use -1 for unknown amino acids
            rsa = dssp_data['rsa'].apply(float).tolist()
            ss = dssp_data['three_hot_ss'].apply(lambda s: self.ss_to_index.get(s, -1)).tolist()  # Use -1 for unknown secondary structures
            chain = dssp_data['chain'].apply(lambda c: self.chain_to_index.get(c, -1)).tolist()

            # Convert to PyTorch tensors
            sequence_tensor = torch.tensor(sequence, dtype=torch.long)
            rsa_tensor = torch.tensor(rsa, dtype=torch.float32)
            ss_tensor = torch.tensor(ss, dtype=torch.long)
            chain_tensor = torch.tensor(chain, dtype=torch.long)

            # Get physicochemical properties
            phys_prop_list = [self.phys_props.loc[aa].values for aa in dssp_data['aa'] if aa in self.phys_props.index]
            phys_prop_array = np.array(phys_prop_list, dtype=np.float32)
            phys_prop_tensor = torch.tensor(phys_prop_array, dtype=torch.float32)

            return sequence_tensor, rsa_tensor, ss_tensor, phys_prop_tensor, chain_tensor

        except FileNotFoundError:
            print(f"Error: DSSP file not found at {dssp_path}")
            return None
        except pd.errors.ParserError:
            print(f"Error: Could not parse DSSP file at {dssp_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while reading {dssp_path}: {e}")
            return None

    def load_distance_matrix(self, distance_matrix_path):
        # Load distance matrix from TSV
        try:
            distance_matrix = pd.read_csv(distance_matrix_path, sep='\t', header=None)
            return distance_matrix.values
        except FileNotFoundError:
            print(f"Error: Distance matrix file not found at {distance_matrix_path}")
            return None
        except pd.errors.ParserError:
            print(f"Error: Could not parse distance matrix file at {distance_matrix_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while reading {distance_matrix_path}: {e}")
            return None
    
    def load_phys_props(self, phys_prop_file):
        try:
            return pd.read_csv(phys_prop_file, index_col='amino acid')
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Physicochemical properties file not found at {phys_prop_file}")

    def map_dssp_to_distance_matrix(self, dssp_1_data, dssp_2_data, distance_matrix, interaction_matrix):
        """
        Maps DSSP data (including physicochemical properties, RSA, and SS) to the distance matrix.
        """
        if dssp_1_data is None or dssp_2_data is None:
            print("Error: DSSP data is None. Cannot map to distance matrix.")
            return None

        # Unpack DSSP data
        sequence_1, rsa_1, ss_1, phys_prop_1, chain_1 = dssp_1_data
        sequence_2, rsa_2, ss_2, phys_prop_2, chain_2 = dssp_2_data

        # Placeholder for the enhanced matrix
        num_features = phys_prop_1.shape[1] + 3 + 1  # phys_prop + rsa + ss + chain + distance
        enhanced_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1], num_features))

        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[1]):
                if interaction_matrix[i, j] == 1:
                    # Concatenate features for interacting pairs
                    enhanced_matrix[i, j, :] = np.concatenate((phys_prop_1[i, :], [rsa_1[i]], [ss_1[i]], [chain_1[i]],
                                                              phys_prop_2[j, :], [rsa_2[j]], [ss_2[j]], [chain_2[j]],
                                                              [distance_matrix[i, j]]))
                else:
                    # Use a zero vector for non-interacting pairs
                    enhanced_matrix[i, j, :] = np.zeros(num_features)

        return enhanced_matrix

def get_data_loader(csv_file, complex_dssp_dir, distance_matrix_dir, phys_prop_file, batch_size=32, shuffle=True, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, num_workers=0):
    dataset = ProteinDataset(csv_file, complex_dssp_dir, distance_matrix_dir, phys_prop_file)

    # Filter out None values (due to missing files)
    dataset = [item for item in dataset if item is not None]

    # Split dataset into training, validation, and test sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
