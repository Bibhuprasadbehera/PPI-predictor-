# src/data_loader.py
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import logging
from motif_loader import MotifLoader
from plots_dataloader import (
    create_rsa_distribution_plot,
    create_secondary_structure_distribution_plot,
    create_amino_acid_frequency_plot,
    create_sequence_length_distribution_plot,
    create_physicochemical_properties_distribution_plots,
    create_batch_visualization,
    create_rsa_vs_ss_plot,
    create_chain_distribution_plot,
    create_physicochemical_properties_correlation_plot
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProteinDataset(Dataset):
    def __init__(self, data_dir, phys_prop_file, motif_csv_path=None, normalize_distance=True):
        self.data_dir = data_dir
        self.normalize_distance = normalize_distance
        self.data = self.load_dssp_files(self.data_dir)
        self.phys_props = self.load_phys_props(phys_prop_file)
        
        # Include J and B in amino acid mapping (J -> index 20, B -> index 21)
        self.aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWYJB')}
        self.ss_to_index = {'C': 0, 'H': 1, 'E': 2, 'P': 3}
        self.chain_to_index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'q': 50, 'r': 51, 's': 52, 'u': 53, 'w': 54}
        
        # Initialize motif loader with proper error handling
        self.motif_loader = None
        if motif_csv_path:
            try:
                self.motif_loader = MotifLoader(motif_csv_path)
                logger.info("Successfully initialized motif loader")
            except Exception as e:
                logger.warning(f"Failed to initialize MotifLoader for {motif_csv_path}: {e}. Motif features will be disabled.")

        logger.info(f"Loaded {len(self.data)} samples")
        self.print_sample_data()
        self.generate_plots()

    def generate_plots(self):
        os.makedirs("plots", exist_ok=True)
        create_rsa_distribution_plot(self.data)
        create_secondary_structure_distribution_plot(self.data)
        create_amino_acid_frequency_plot(self.data)
        create_sequence_length_distribution_plot(self.data)
        create_physicochemical_properties_distribution_plots(self.phys_props)
        create_rsa_vs_ss_plot(self.data)
        create_chain_distribution_plot(self.data)
        create_physicochemical_properties_correlation_plot(self.phys_props)
    
    def load_dssp_files(self, data_dir):
        all_data = []
        dssp_dir = os.path.join(data_dir, 'dssp')
        for file in tqdm(os.listdir(dssp_dir), desc="Loading DSSP files"):
            # Accept both _dssp.csv and _dssp.tsv files
            if file.endswith('_dssp.csv') or file.endswith('_dssp.tsv'):
                # Determine protein_id by stripping the correct suffix
                if file.endswith('_dssp.csv'):
                    protein_id = file[:-len('_dssp.csv')]
                else:
                    protein_id = file[:-len('_dssp.tsv')]

                distance_mat_file = os.path.join(data_dir, 'distance_mat', f'{protein_id}_ca.tsv')

                if os.path.exists(distance_mat_file):
                    file_path = os.path.join(dssp_dir, file)
                    try:
                        # Use appropriate separator based on extension
                        if file.endswith('_dssp.tsv'):
                            df = pd.read_csv(file_path, sep='\t')
                        else:
                            df = pd.read_csv(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to read file {file_path}: {e}. Skipping.")
                        continue

                    required_columns = ['aa', 'rsa', 'three_hot_ss', 'chain']
                    
                    if all(col in df.columns for col in required_columns):
                        try:
                            # Aggregate features per file - store secondary structure as array, not concatenated string
                            sequence_str = ''.join(df['aa'].astype(str).tolist())
                            ss_array = df['three_hot_ss'].astype(str).tolist()  # Keep as list/array of per-residue values
                            rsa_vec = df['rsa'].astype(float).to_numpy()
                            chain = str(df['chain'].iloc[0])
                            
                            all_data.append({
                                'aa': sequence_str,
                                'three_hot_ss': ss_array,  # Store as array instead of concatenated string
                                'rsa': rsa_vec,
                                'chain': chain,
                                'protein_id': protein_id
                            })
                        except (TypeError, ValueError) as e:
                            logger.error(f"Error processing file {file}: {e}. Skipping.")
                    else:
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        logger.warning(f"File {file} is missing required columns: {missing_columns}. Skipping.")
                else:
                    logger.warning(f"Missing distance matrix file for {protein_id}")
        
        if not all_data:
            raise ValueError(f"No valid data files found in {data_dir}. Please check your data files and ensure they contain the required columns.")
        
        return pd.DataFrame(all_data)
            
    def load_distance_mat(self, protein_id, sequence):
        distance_mat_file = os.path.join(self.data_dir, 'distance_mat', f'{protein_id}_ca.tsv')
        if os.path.exists(distance_mat_file):
            try:
                distance_mat_df = pd.read_csv(distance_mat_file, sep='\t', header=0, index_col=0)
                distance_mat = distance_mat_df.values
                
                # Convert to interaction probability (inverse relationship to distance)
                # Lower distances = higher interaction probability
                if self.normalize_distance:
                    # Normalize distances to [0, 1] range and convert to probabilities
                    # Use inverse function: prob = 1/(1 + distance/10) to map distances to probabilities
                    distance_mat = np.nan_to_num(distance_mat, nan=20.0, posinf=20.0, neginf=0.0)
                    interaction_prob = 1.0 / (1.0 + distance_mat / 10.0)
                else:
                    interaction_prob = np.nan_to_num(distance_mat, nan=10.0, posinf=20.0, neginf=0.0)
                
                distance_mat_tensor = torch.tensor(interaction_prob, dtype=torch.float32)
                return distance_mat_tensor
            except Exception as e:
                logger.error(f"Error loading distance matrix for protein {protein_id} from {distance_mat_file}: {e}. Returning zero matrix.")
                return torch.zeros((len(sequence), len(sequence)), dtype=torch.float32)
        else:
            logger.warning(f"Distance matrix file not found for protein {protein_id}: {distance_mat_file}. Returning zero matrix.")
            return torch.zeros((len(sequence), len(sequence)), dtype=torch.float32)
        
    def load_phys_props(self, phys_prop_file):
        try:
            return pd.read_csv(phys_prop_file, index_col='amino acid')
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Physicochemical properties file not found: {phys_prop_file}")

    @staticmethod
    def _load_phys_props_static(phys_prop_file):
        """Static method to load physicochemical properties."""
        try:
            return pd.read_csv(phys_prop_file, index_col='amino acid')
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Physicochemical properties file not found: {phys_prop_file}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = str(row['aa'])
        protein_id = row['protein_id']

        # Handle non-standard amino acids for physicochemical lookup
        aa_standard_map = {'J': 'L', 'B': 'N', 'Z': 'Q'}
        phys_prop_list = []
        for aa in sequence:
            mapped_aa = aa_standard_map.get(aa, aa)
            try:
                phys_prop_list.append(self.phys_props.loc[mapped_aa].values)
            except KeyError:
                phys_prop_list.append(np.zeros(self.phys_props.shape[1], dtype=np.float32))
        phys_prop_tensor = torch.tensor(np.array(phys_prop_list), dtype=torch.float32)

        # Sequence tensor
        seq_indices = []
        for aa in sequence:
            if aa in self.aa_to_index:
                seq_indices.append(self.aa_to_index[aa])
            else:
                fallback = aa_standard_map.get(aa, 'A')
                seq_indices.append(self.aa_to_index.get(fallback, 0))
        sequence_tensor = torch.tensor(seq_indices, dtype=torch.long)

        # Secondary structure tensor - handling both the old string format and new list format
        ss_data = row['three_hot_ss']
        if isinstance(ss_data, list):
            # New format: stored as list of per-residue values
            ss_indices = [self.ss_to_index.get(str(s), 0) for s in ss_data]
        else:
            # Backward compatibility: old format was string
            ss_chars = str(ss_data)
            ss_indices = [self.ss_to_index.get(s, 0) for s in ss_chars]
        ss_tensor = torch.tensor(ss_indices, dtype=torch.long)

        # RSA tensor - use per-residue values when available
        rsa = row['rsa']
        if isinstance(rsa, (np.ndarray, list)):
            rsa_tensor = torch.tensor(rsa, dtype=torch.float32)
        else:
            try:
                rsa_tensor = torch.tensor([float(rsa)] * len(sequence), dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Invalid RSA value for {protein_id}: {e}. Using zeros.")
                rsa_tensor = torch.zeros(len(sequence), dtype=torch.float32)

        # Chain tensor
        chain_id = row.get('chain', '0')
        chain_idx = self.chain_to_index.get(chain_id, 0)
        chain_tensor = torch.tensor([chain_idx] * len(sequence), dtype=torch.long)

        # Distance matrix - now used as target
        distance_mat = self.load_distance_mat(protein_id, sequence)

        # Motif features with fallback
        if self.motif_loader is not None:
            try:
                motif_feats = self.motif_loader.get_motif_features_for_any(protein_id, len(sequence))
                motif_binary = motif_feats.get('motif_binary', np.zeros(len(sequence), dtype=np.float32))
                motif_index = motif_feats.get('motif_index', np.zeros(len(sequence), dtype=np.int32))
                motif_position = motif_feats.get('motif_position', np.zeros(len(sequence), dtype=np.int32))
                motif_overlap = motif_feats.get('motif_overlap_count', np.zeros(len(sequence), dtype=np.int32))
            except Exception as e:
                logger.warning(f"Failed to get motif features for {protein_id}: {e}")
                motif_binary = np.zeros(len(sequence), dtype=np.float32)
                motif_index = np.zeros(len(sequence), dtype=np.int32)
                motif_position = np.zeros(len(sequence), dtype=np.int32)
                motif_overlap = np.zeros(len(sequence), dtype=np.int32)
        else:
            motif_binary = np.zeros(len(sequence), dtype=np.float32)
            motif_index = np.zeros(len(sequence), dtype=np.int32)
            motif_position = np.zeros(len(sequence), dtype=np.int32)
            motif_overlap = np.zeros(len(sequence), dtype=np.int32)

        return {
            'sequence': sequence_tensor,
            'rsa': rsa_tensor,
            'ss': ss_tensor,
            'phys_props': phys_prop_tensor,
            'chain': chain_tensor,
            'motif_binary': torch.tensor(motif_binary, dtype=torch.float32),
            'motif_index': torch.tensor(motif_index, dtype=torch.long),
            'motif_position': torch.tensor(motif_position, dtype=torch.long),
            'motif_overlap': torch.tensor(motif_overlap, dtype=torch.long),
            'distance_mat': distance_mat,  # Now as target
            'protein_id': protein_id
        }

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        """Collate a batch of samples into padded tensors."""
        # Get max sequence length for padding
        max_len = max(sample['sequence'].size(0) for sample in batch)

        output = {}
        
        # Handle non-tensor items (e.g., protein_id)
        if 'protein_id' in batch[0]:
            output['protein_id'] = [sample['protein_id'] for sample in batch]
            
        # Pad sequence tensors
        for key in ['sequence', 'rsa', 'ss', 'chain', 'motif_binary', 'motif_index', 'motif_position', 'motif_overlap']:
            if key in batch[0] and isinstance(batch[0][key], torch.Tensor):
                output[key] = torch.stack([
                    F.pad(sample[key], (0, max_len - sample[key].size(0)))
                    for sample in batch
                ])
                
        # Special handling for phys_props (needs padding in specific dimension)
        if 'phys_props' in batch[0]:
            output['phys_props'] = torch.stack([
                F.pad(sample['phys_props'], (0, 0, 0, max_len - sample['phys_props'].size(0)))
                for sample in batch
            ])
            
        # Special handling for distance matrices - pad (left, right, top, bottom)
        if 'distance_mat' in batch[0]:
            output['distance_mat'] = torch.stack([
                F.pad(sample['distance_mat'], 
                     (0, max_len - sample['distance_mat'].size(1),  # pad columns (left=0, right)
                      0, max_len - sample['distance_mat'].size(0)))  # pad rows (top=0, bottom)
                for sample in batch
            ])
            
        return output
    
    def print_sample_data(self):
        """Log sample data information at debug level."""
        logger.debug("Sample data preview:")
        logger.debug("\n" + str(self.data.head()))
        logger.debug("Data types:")
        logger.debug("\n" + str(self.data.dtypes))
        logger.debug("Data statistics:")
        logger.debug("\n" + str(self.data.describe()))

def get_data_loader(data_dir, phys_prop_file, batch_size, num_workers, motif_csv_path=None, normalize_distance=True):
    dataset = ProteinDataset(data_dir, phys_prop_file, motif_csv_path=motif_csv_path, normalize_distance=normalize_distance)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ProteinDataset.collate_fn)

def visualize_batch(batch, num_samples=5):
    """Visualize a batch of samples using the dictionary format."""
    create_batch_visualization({
        'sequences': batch['sequence'],
        'rsas': batch['rsa'],
        'secondary_structures': batch['ss'],
        'phys_props': batch['phys_props'],
        'chains': batch['chain'],
        'labels': batch.get('motif_binary', torch.zeros_like(batch['sequence']))  # Use motif binary as labels if available
    }, num_samples)