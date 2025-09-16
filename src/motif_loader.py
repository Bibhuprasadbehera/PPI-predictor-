import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MotifLoader:
    """
    A class to load and process motif data from CSV files and generate per-residue motif features.
    
    This class handles:
    - Loading motif data from CSV files
    - Preprocessing motif sequences with non-standard amino acids
    - Mapping motif data to protein sequences
    - Generating per-residue motif features including binary flags, motif indices, and position encoding
    """
    
    def __init__(self, motif_csv_path: str):
        """
        Initialize the MotifLoader with the path to the motif CSV file.
        
        Args:
            motif_csv_path (str): Path to the motif CSV file
        """
        self.motif_csv_path = Path(motif_csv_path)
        self.motif_data = None
        self.protein_motifs = {}  # Dict to store motifs per protein
        self.motif_index_mapping = {}  # Maps original motif indices to offset indices
        self.max_motif_index = 0
        
        # Set up logging
        # self.logger = logging.getLogger(__name__)
        
        # Load and preprocess the motif data
        self._load_motif_data()
        self._preprocess_motif_data()
    
    def _load_motif_data(self) -> None:
        """
        Load motif data from the CSV file.
        
        Raises:
            FileNotFoundError: If the motif CSV file doesn't exist
            pd.errors.EmptyDataError: If the CSV file is empty
        """
        try:
            if not self.motif_csv_path.exists():
                raise FileNotFoundError(f"Motif CSV file not found: {self.motif_csv_path}")
            
            self.motif_data = pd.read_csv(self.motif_csv_path)
            logger.info(f"Loaded motif data with {len(self.motif_data)} rows from {self.motif_csv_path}")
            
            # Validate required columns
            required_columns = [
                'sequence_protein_id', 'motif_index', 'motif_sequence', 
                'start_position', 'end_position', 'motif_found'
            ]
            missing_columns = [col for col in required_columns if col not in self.motif_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in motif CSV: {missing_columns}")
                
        except Exception as e:
            logger.error(f"Error loading motif data: {e}")
            raise
    
    def _preprocess_motif_data(self) -> None:
        """
        Preprocess the motif data including:
        - Filtering out entries with no motifs found
        - Handling non-standard amino acids
        - Creating motif index mappings with +1 offset
        - Organizing data by protein ID
        """
        try:
            # Filter out entries where no motif was found
            valid_motifs = self.motif_data[
                (self.motif_data['motif_found'] == 'yes') & 
                (self.motif_data['motif_sequence'].notna()) &
                (self.motif_data['motif_sequence'] != 'empty')
            ].copy()
            
            logger.info(f"Found {len(valid_motifs)} valid motif entries")
            
            # Handle non-standard amino acids in motif sequences
            valid_motifs['processed_motif_sequence'] = valid_motifs['motif_sequence'].apply(
                self._process_motif_sequence
            )
            
            # Create motif index mapping with +1 offset (0 reserved for "no motif")
            unique_motif_indices = valid_motifs['motif_index'].dropna().unique()
            self.motif_index_mapping = {
                int(idx): int(idx) + 1 for idx in unique_motif_indices if pd.notna(idx)
            }
            self.max_motif_index = max(self.motif_index_mapping.values()) if self.motif_index_mapping else 0
            
            # Organize motifs by protein ID
            for _, row in valid_motifs.iterrows():
                protein_id = row['sequence_protein_id']
                if protein_id not in self.protein_motifs:
                    self.protein_motifs[protein_id] = []
                
                motif_info = {
                    'motif_index': int(row['motif_index']) if pd.notna(row['motif_index']) else None,
                    'offset_motif_index': self.motif_index_mapping.get(int(row['motif_index']), 0) if pd.notna(row['motif_index']) else 0,
                    'motif_sequence': row['motif_sequence'],
                    'processed_sequence': row['processed_motif_sequence'],
                    'start_position': int(row['start_position']) if pd.notna(row['start_position']) else None,
                    'end_position': int(row['end_position']) if pd.notna(row['end_position']) else None,
                    'motif_length': len(row['processed_motif_sequence']) if pd.notna(row['processed_motif_sequence']) else 0
                }
                self.protein_motifs[protein_id].append(motif_info)
            
            logger.info(f"Processed motifs for {len(self.protein_motifs)} proteins")
            logger.info(f"Max motif index (offset): {self.max_motif_index}")
            
        except Exception as e:
            logger.error(f"Error preprocessing motif data: {e}")
            raise
    
    def _process_motif_sequence(self, sequence: str) -> str:
        """
        Process motif sequences to handle non-standard amino acids.
        
        Non-standard amino acids are mapped as follows:
        - 'J' -> 'L' (Leucine, as J is sometimes used for Leucine/Isoleucine)
        - 'B' -> 'N' (Asparagine, as B represents Aspartic acid/Asparagine)
        - 'Z' -> 'Q' (Glutamine, as Z represents Glutamic acid/Glutamine)
        
        Args:
            sequence (str): Original motif sequence
            
        Returns:
            str: Processed motif sequence with standard amino acids
        """
        if pd.isna(sequence) or sequence == 'empty':
            return ''
        
        # Mapping for non-standard amino acids
        aa_mapping = {
            'J': 'L',  # Leucine/Isoleucine ambiguity
            'B': 'N',  # Aspartic acid/Asparagine ambiguity  
            'Z': 'Q'   # Glutamic acid/Glutamine ambiguity
        }
        
        processed_sequence = sequence.upper()
        for non_standard, standard in aa_mapping.items():
            processed_sequence = processed_sequence.replace(non_standard, standard)
        
        return processed_sequence
    
    def get_protein_motifs(self, protein_id: str) -> List[Dict]:
        """
        Get all motifs for a specific protein ID.
        
        Args:
            protein_id (str): Protein identifier
            
        Returns:
            List[Dict]: List of motif information dictionaries
        """
        return self.protein_motifs.get(protein_id, [])
    
    def has_motifs(self, protein_id: str) -> bool:
        """
        Check if a protein has any motifs.
        
        Args:
            protein_id (str): Protein identifier
            
        Returns:
            bool: True if protein has motifs, False otherwise
        """
        return protein_id in self.protein_motifs and len(self.protein_motifs[protein_id]) > 0
    
    def get_motif_features(self, protein_id: str, sequence_length: int) -> Dict[str, np.ndarray]:
        """
        Generate per-residue motif features for a protein sequence.
        
        Args:
            protein_id (str): Protein identifier
            sequence_length (int): Length of the protein sequence
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing motif features:
                - 'motif_binary': Binary flags for motif presence
                - 'motif_index': Motif index for each residue
                - 'motif_position': Position within motif for each residue
                - 'motif_overlap_count': Count of overlapping motifs per residue
        """
        # Initialize feature arrays with zeros
        motif_binary = np.zeros(sequence_length, dtype=np.float32)
        motif_index = np.zeros(sequence_length, dtype=np.int32)
        motif_position = np.zeros(sequence_length, dtype=np.int32)
        motif_overlap_count = np.zeros(sequence_length, dtype=np.int32)
        
        # Get motifs for this protein
        protein_motifs = self.get_protein_motifs(protein_id)
        
        if not protein_motifs:
            logger.debug(f"No motifs found for protein {protein_id}")
            return {
                'motif_binary': motif_binary,
                'motif_index': motif_index,
                'motif_position': motif_position,
                'motif_overlap_count': motif_overlap_count
            }
        
        # Process each motif
        for motif_info in protein_motifs:
            start_pos = motif_info['start_position']
            end_pos = motif_info['end_position']
            offset_motif_idx = motif_info['offset_motif_index']
            
            if start_pos is None or end_pos is None:
                logger.warning(f"Invalid motif positions for protein {protein_id}: start={start_pos}, end={end_pos}")
                continue
            
            # Convert to 0-based indexing
            start_idx = max(0, start_pos - 1)
            end_idx = min(sequence_length, end_pos)
            
            if start_idx >= sequence_length or end_idx <= 0:
                logger.warning(f"Motif position out of bounds for protein {protein_id}: start={start_idx}, end={end_idx}, seq_len={sequence_length}")
                continue
            
            # Check for overlaps before setting
            if motif_binary[start_idx:end_idx].any():
                logger.info(f"Motif overlap in {protein_id}: indices {start_idx}-{end_idx}, motif {offset_motif_idx}")
                motif_overlap_count[start_idx:end_idx] += 1
            
            # Set motif features for this region
            motif_binary[start_idx:end_idx] = 1.0
            motif_index[start_idx:end_idx] = offset_motif_idx
            
            # Set position within motif (1-based)
            motif_length = end_idx - start_idx
            motif_position[start_idx:end_idx] = np.arange(1, motif_length + 1)
        
        return {
            'motif_binary': motif_binary,
            'motif_index': motif_index,
            'motif_position': motif_position,
            'motif_overlap_count': motif_overlap_count
        }
    
    def get_motif_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the loaded motif data.
        
        Returns:
            Dict[str, any]: Dictionary containing motif statistics
        """
        total_proteins = len(self.protein_motifs)
        total_motifs = sum(len(motifs) for motifs in self.protein_motifs.values())
        
        motif_lengths = []
        motif_indices = set()
        
        for protein_motifs in self.protein_motifs.values():
            for motif in protein_motifs:
                if motif['motif_length'] > 0:
                    motif_lengths.append(motif['motif_length'])
                if motif['motif_index'] is not None:
                    motif_indices.add(motif['motif_index'])
        
        return {
            'total_proteins_with_motifs': total_proteins,
            'total_motif_instances': total_motifs,
            'unique_motif_indices': len(motif_indices),
            'max_offset_motif_index': self.max_motif_index,
            'motif_length_stats': {
                'min': min(motif_lengths) if motif_lengths else 0,
                'max': max(motif_lengths) if motif_lengths else 0,
                'mean': np.mean(motif_lengths) if motif_lengths else 0,
                'std': np.std(motif_lengths) if motif_lengths else 0
            }
        }
    
    def get_all_protein_ids(self) -> Set[str]:
        """
        Get all protein IDs that have motif data.
        
        Returns:
            Set[str]: Set of protein IDs with motif data
        """
        return set(self.protein_motifs.keys())
    
    def validate_protein_sequence(self, protein_id: str, sequence: str) -> bool:
        """
        Validate that motif positions are compatible with the given protein sequence.
        
        Args:
            protein_id (str): Protein identifier
            sequence (str): Protein sequence
            
        Returns:
            bool: True if all motifs are within sequence bounds, False otherwise
        """
        protein_motifs = self.get_protein_motifs(protein_id)
        sequence_length = len(sequence)
        
        for motif_info in protein_motifs:
            start_pos = motif_info['start_position']
            end_pos = motif_info['end_position']
            
            if start_pos is None or end_pos is None:
                continue
                
            # Convert to 0-based indexing
            start_idx = start_pos - 1
            end_idx = end_pos
            
            if start_idx < 0 or end_idx > sequence_length:
                logger.warning(
                    f"Motif position out of bounds for protein {protein_id}: "
                    f"motif range [{start_idx}:{end_idx}], sequence length {sequence_length}"
                )
                return False
        
        return True

    def get_motif_features_for_any(self, protein_id: str, sequence_length: int) -> Dict[str, np.ndarray]:
        """
        Try to get motif features for a protein ID with fallback to base ID.
        
        Args:
            protein_id (str): Full protein ID (e.g., '1BML_D_B')
            sequence_length (int): Length of the protein sequence
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing motif features
        """
        # Try with exact protein ID first
        features = self.get_motif_features(protein_id, sequence_length)
        if not features['motif_binary'].any():
            # Try with base ID if no motifs found
            base_id = protein_id.split('_')[0]
            logger.debug(f"No motifs found for {protein_id}, trying base ID {base_id}")
            features = self.get_motif_features(base_id, sequence_length)
        return features


def create_motif_loader(motif_csv_path: str) -> MotifLoader:
    """
    Factory function to create a MotifLoader instance.
    
    Args:
        motif_csv_path (str): Path to the motif CSV file
        
    Returns:
        MotifLoader: Initialized MotifLoader instance
    """
    return MotifLoader(motif_csv_path)


# Example usage and testing functions
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    motif_csv_path = "../data/parsed_motifs/self_match_motif_data_with_empty.csv"
    
    try:
        # Create motif loader
        motif_loader = create_motif_loader(motif_csv_path)
        
        # Get statistics
        stats = motif_loader.get_motif_statistics()
        print("Motif Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example: Get motif features for a protein
        protein_ids = list(motif_loader.get_all_protein_ids())[:5]  # First 5 proteins
        
        for protein_id in protein_ids:
            print(f"\nProtein: {protein_id}")
            motifs = motif_loader.get_protein_motifs(protein_id)
            print(f"  Number of motifs: {len(motifs)}")
            
            # Generate features for a hypothetical sequence of length 200
            features = motif_loader.get_motif_features(protein_id, 200)
            print(f"  Motif binary sum: {features['motif_binary'].sum()}")
            print(f"  Unique motif indices: {np.unique(features['motif_index'])}")
            
    except Exception as e:
        print(f"Error: {e}")