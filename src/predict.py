# src/predict.py
import torch
import numpy as np
import yaml
from model import ProteinInteractionModel
from data_loader import ProteinDataset
import logging

def setup_logger():
    """Set up logging for prediction."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

class SequencePredictor:
    """Class for predicting interaction matrices from amino acid sequences."""
    
    def __init__(self, model_path, config_path):
        """Initialize the predictor with trained model."""
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize model
        self.model = ProteinInteractionModel(
            self.cfg['model']['input_size'],
            self.cfg['model']['hidden_size'],
            self.cfg['model']['num_layers'],
            self.cfg['model']['phys_prop_size'],
            self.cfg['model']['num_chains'],
            self.cfg['model']['motif_feature_size']
        )
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
        
        # Load physicochemical properties for mapping
        self.phys_props = ProteinDataset._load_phys_props_static(self.cfg['data']['phys_prop_file'])
        
        # Amino acid mappings
        self.aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWYJB')}
        self.ss_to_index = {'C': 0, 'H': 1, 'E': 2, 'P': 3}
        self.chain_to_index = {'0': 0}  # Default chain
        
        # Non-standard amino acid mapping
        self.aa_standard_map = {'J': 'L', 'B': 'N', 'Z': 'Q'}
    
    def _process_sequence(self, sequence):
        """Process a single amino acid sequence into model inputs."""
        sequence = sequence.upper()
        
        # Sequence tensor
        seq_indices = []
        for aa in sequence:
            if aa in self.aa_to_index:
                seq_indices.append(self.aa_to_index[aa])
            else:
                fallback = self.aa_standard_map.get(aa, 'A')
                seq_indices.append(self.aa_to_index.get(fallback, 0))
        sequence_tensor = torch.tensor(seq_indices, dtype=torch.long).unsqueeze(0)  # Add batch dim
        
        # Default RSA (based on common values from training data, typically 0.2-0.8 range)
        rsa_tensor = torch.full((1, len(sequence)), 0.5, dtype=torch.float32)
        
        # Default secondary structure (coil)
        ss_tensor = torch.zeros((1, len(sequence)), dtype=torch.long)
        
        # Default physicochemical properties
        phys_prop_list = []
        for aa in sequence:
            mapped_aa = self.aa_standard_map.get(aa, aa)
            try:
                phys_prop_list.append(self.phys_props.loc[mapped_aa].values)
            except KeyError:
                phys_prop_list.append(np.zeros(self.phys_props.shape[1], dtype=np.float32))
        phys_prop_tensor = torch.tensor(np.array([phys_prop_list]), dtype=torch.float32)
        
        # Default chain (chain 0)
        chain_tensor = torch.zeros((1, len(sequence)), dtype=torch.long)
        
        # Default motif features (no motifs)
        motif_binary = torch.zeros((1, len(sequence)), dtype=torch.float32)
        motif_index = torch.zeros((1, len(sequence)), dtype=torch.long)
        motif_position = torch.zeros((1, len(sequence)), dtype=torch.long)
        motif_overlap = torch.zeros((1, len(sequence)), dtype=torch.long)
        
        return {
            'sequence': sequence_tensor,
            'rsa': rsa_tensor,
            'ss': ss_tensor,
            'phys_props': phys_prop_tensor,
            'chain': chain_tensor,
            'motif_binary': motif_binary,
            'motif_index': motif_index,
            'motif_position': motif_position,
            'motif_overlap': motif_overlap
        }
    
    def predict(self, sequence1, sequence2=None):
        """Predict interaction matrix between two amino acid sequences.
        
        Args:
            sequence1: First amino acid sequence
            sequence2: Second amino acid sequence (if None, predict intra-protein interactions)
        """
        inputs1 = self._process_sequence(sequence1)
        
        if sequence2 is not None:
            inputs2 = self._process_sequence(sequence2)
            
            with torch.no_grad():
                output = self.model(
                    inputs1['sequence'], 
                    inputs1['rsa'], 
                    inputs1['ss'],
                    inputs1['phys_props'], 
                    inputs1['chain'],
                    inputs1['motif_binary'], 
                    inputs1['motif_index'],
                    inputs1['motif_position'], 
                    inputs1['motif_overlap'],
                    use_all_features=False,  # Use minimal features for inference
                    sequence2=inputs2['sequence'],
                    rsa2=inputs2['rsa'],
                    ss2=inputs2['ss'],
                    phys_props2=inputs2['phys_props'],
                    chains2=inputs2['chain'],
                    motif_binary2=inputs2['motif_binary'],
                    motif_index2=inputs2['motif_index'],
                    motif_position2=inputs2['motif_position'],
                    motif_overlap2=inputs2['motif_overlap']
                )
        else:
            # For single sequence, predict intra-protein interactions
            with torch.no_grad():
                output = self.model(
                    inputs1['sequence'], 
                    inputs1['rsa'], 
                    inputs1['ss'],
                    inputs1['phys_props'], 
                    inputs1['chain'],
                    inputs1['motif_binary'], 
                    inputs1['motif_index'],
                    inputs1['motif_position'], 
                    inputs1['motif_overlap'],
                    use_all_features=False  # Use minimal features for inference
                )
        
        # Extract the interaction matrix (already in probability format)
        interaction_matrix = output.squeeze(0).numpy()  # Remove batch dimension
        
        # Calculate interaction probabilities (already done in model)
        interaction_prob = interaction_matrix
        
        return interaction_matrix, interaction_prob
    
    def predict_ppi(self, sequence1, sequence2):
        """Predict protein-protein interaction between two sequences."""
        return self.predict(sequence1, sequence2)

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict protein-protein interaction from sequences')
    parser.add_argument('--model', default='checkpoints/best_model.pth', help='Path to trained model')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--sequence', help='First amino acid sequence to predict (for intra-protein)')
    parser.add_argument('--sequence1', help='First amino acid sequence for PPI prediction')
    parser.add_argument('--sequence2', help='Second amino acid sequence for PPI prediction')
    parser.add_argument('--file', help='File containing sequences (one per line for intra-protein)')
    parser.add_argument('--file1', help='File containing first sequences for PPI prediction')
    parser.add_argument('--file2', help='File containing second sequences for PPI prediction')
    parser.add_argument('--save-matrix', help='Save the full interaction matrix to a specified file (npy format)')
    parser.add_argument('--save-format', choices=['npy', 'csv'], default='npy', help='Format for saving matrix (default: npy)')
    
    args = parser.parse_args()
    
    predictor = SequencePredictor(args.model, args.config)

    if args.sequence1 and args.sequence2:
        # Predict PPI between two sequences
        interaction_matrix, interaction_prob = predictor.predict(args.sequence1, args.sequence2)
        print(f"Sequence 1: {args.sequence1}")
        print(f"Sequence 2: {args.sequence2}")
        print(f"Interaction matrix shape: {interaction_matrix.shape}")
        print(f"Interaction probability range: {interaction_prob.min():.3f} - {interaction_prob.max():.3f}")
        
        # Save matrix to file if requested
        if args.save_matrix:
            import numpy as np
            if args.save_format == 'npy':
                np.save(args.save_matrix, interaction_prob)
                print(f"Full interaction matrix saved to {args.save_matrix}")
            elif args.save_format == 'csv':
                import pandas as pd
                df = pd.DataFrame(interaction_prob)
                df.to_csv(args.save_matrix, index=False)
                print(f"Full interaction matrix saved to {args.save_matrix}")
        
        # Print some sample interactions (first 5x5)
        # For very large matrices, only show sample; for manageable sizes, show more
        if interaction_prob.shape[0] <= 50 and interaction_prob.shape[1] <= 50:
            print(f"Full interaction matrix ({interaction_prob.shape[0]}x{interaction_prob.shape[1]}):")
            print(interaction_prob)
        elif args.save_matrix is None:  # Only show sample if not saving to file
            print(f"Sample interaction matrix (top-left 5x5 of {interaction_prob.shape[0]}x{interaction_prob.shape[1]}):")
            sample_size = min(5, interaction_prob.shape[0], interaction_prob.shape[1])
            print(interaction_prob[:sample_size, :sample_size])
            print(f"... (use --save-matrix to save full matrix or use smaller sequences for full display)")
        
    elif args.sequence:
        # Predict intra-protein interactions for single sequence
        interaction_matrix, interaction_prob = predictor.predict(args.sequence)
        print(f"Sequence: {args.sequence}")
        print(f"Interaction matrix shape: {interaction_matrix.shape}")
        print(f"Interaction probability range: {interaction_prob.min():.3f} - {interaction_prob.max():.3f}")
        
        # Save matrix to file if requested
        if args.save_matrix:
            import numpy as np
            if args.save_format == 'npy':
                np.save(args.save_matrix, interaction_prob)
                print(f"Full interaction matrix saved to {args.save_matrix}")
            elif args.save_format == 'csv':
                import pandas as pd
                df = pd.DataFrame(interaction_prob)
                df.to_csv(args.save_matrix, index=False)
                print(f"Full interaction matrix saved to {args.save_matrix}")
        
        # Print some sample interactions (first 5x5)
        # For very large matrices, only show sample; for manageable sizes, show more
        if interaction_prob.shape[0] <= 50 and interaction_prob.shape[1] <= 50:
            print(f"Full interaction matrix ({interaction_prob.shape[0]}x{interaction_prob.shape[1]}):")
            print(interaction_prob)
        elif args.save_matrix is None:  # Only show sample if not saving to file
            print(f"Sample interaction matrix (top-left 5x5 of {interaction_prob.shape[0]}x{interaction_prob.shape[1]}):")
            sample_size = min(5, interaction_prob.shape[0], interaction_prob.shape[1])
            print(interaction_prob[:sample_size, :sample_size])
            print(f"... (use --save-matrix to save full matrix or use smaller sequences for full display)")
        
    elif args.file1 and args.file2:
        # Predict PPI for multiple sequence pairs from files
        with open(args.file1, 'r') as f1, open(args.file2, 'r') as f2:
            sequences1 = [line.strip() for line in f1 if line.strip()]
            sequences2 = [line.strip() for line in f2 if line.strip()]
        
        # Pair up sequences (assuming same number in both files)
        min_len = min(len(sequences1), len(sequences2))
        for i in range(min_len):
            seq1 = sequences1[i]
            seq2 = sequences2[i]
            interaction_matrix, interaction_prob = predictor.predict(seq1, seq2)
            print(f"\nPPI between Sequence {i+1} pairs:")
            print(f"  Seq1: {seq1[:30]}{'...' if len(seq1) > 30 else ''}")
            print(f"  Seq2: {seq2[:30]}{'...' if len(seq2) > 30 else ''}")
            print(f"  Interaction matrix shape: {interaction_matrix.shape}")
            print(f"  Interaction probability range: {interaction_prob.min():.3f} - {interaction_prob.max():.3f}")
        
    elif args.file:
        # Predict intra-protein interactions for multiple sequences from file
        with open(args.file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]

        for i, seq in enumerate(sequences):
            interaction_matrix, interaction_prob = predictor.predict(seq)
            print(f"\nSequence {i+1}: {seq[:30]}{'...' if len(seq) > 30 else ''}")
            print(f"Interaction matrix shape: {interaction_matrix.shape}")
            print(f"Interaction probability range: {interaction_prob.min():.3f} - {interaction_prob.max():.3f}")
    else:
        print("Please provide either:")
        print("  --sequence for intra-protein prediction")
        print("  --sequence1 and --sequence2 for PPI prediction")
        print("  --file for multiple intra-protein predictions")
        print("  --file1 and --file2 for multiple PPI predictions")

if __name__ == '__main__':
    main()