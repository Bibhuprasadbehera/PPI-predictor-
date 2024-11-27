# src/predict.py

import torch
import yaml
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from data_loader import ProteinDataset
from model import ProteinInteractionModel

# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def predict(model_path, sequence1, sequence2, config, phys_prop_file):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'],
                                    cfg['model']['phys_prop_size'], cfg['model']['num_chains'])

    state_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()

    phys_props_df = pd.read_csv(phys_prop_file, index_col='amino acid')
    aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    chain_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}  # Add more as needed

    # Pad sequences to the same length
    max_len = max(len(sequence1), len(sequence2))
    sequence1_padded = sequence1 + '-' * (max_len - len(sequence1))
    sequence2_padded = sequence2 + '-' * (max_len - len(sequence2))

    # Create tensors for sequence1
    sequence1_tensor = torch.tensor([aa_to_index.get(aa, 0) for aa in sequence1_padded], dtype=torch.long).unsqueeze(0)
    rsa1_tensor = torch.tensor([0.5] * max_len, dtype=torch.float32).unsqueeze(0)  # Replace with actual RSA data if available
    ss1_tensor = torch.tensor([0] * max_len, dtype=torch.long).unsqueeze(0)  # Replace with actual secondary structure data if available
    chain1_tensor = torch.tensor([chain_to_index['A']] * max_len, dtype=torch.long).unsqueeze(0)  # Assuming chain ID is 'A'
    phys_props1_list = [phys_props_df.loc[aa].values if aa != '-' else np.zeros(phys_props_df.shape[1]) for aa in sequence1_padded]
    phys_props1_array = np.array(phys_props1_list)
    phys_props1_tensor = torch.tensor(phys_props1_array, dtype=torch.float32).unsqueeze(0)

    # Create tensors for sequence2
    sequence2_tensor = torch.tensor([aa_to_index.get(aa, 0) for aa in sequence2_padded], dtype=torch.long).unsqueeze(0)
    rsa2_tensor = torch.tensor([0.5] * max_len, dtype=torch.float32).unsqueeze(0)  # Replace with actual RSA data if available
    ss2_tensor = torch.tensor([0] * max_len, dtype=torch.long).unsqueeze(0)  # Replace with actual secondary structure data if available
    chain2_tensor = torch.tensor([chain_to_index['B']] * max_len, dtype=torch.long).unsqueeze(0)  # Assuming chain ID is 'B'
    phys_props2_list = [phys_props_df.loc[aa].values if aa != '-' else np.zeros(phys_props_df.shape[1]) for aa in sequence2_padded]
    phys_props2_array = np.array(phys_props2_list)
    phys_props2_tensor = torch.tensor(phys_props2_array, dtype=torch.float32).unsqueeze(0)


    with torch.no_grad():
        predictions, predictions2, interaction_matrix = model(sequence1_tensor, rsa1_tensor, ss1_tensor, phys_props1_tensor, chain1_tensor,
                           sequence2_tensor, rsa2_tensor, ss2_tensor, phys_props2_tensor, chain2_tensor)

    predictions = predictions.squeeze(0).cpu().numpy().tolist()
    predictions2 = predictions2.squeeze(0).cpu().numpy().tolist()
    interaction_matrix = interaction_matrix.squeeze(0).cpu().numpy()

    # Plot predicted interaction scores along the sequence1
    plt.figure(figsize=(12, 6))
    plt.plot(predictions[:len(sequence1)])  # Plot only for the original sequence length
    plt.title('Predicted Interaction Scores Along the Sequence 1')
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Interaction Score')
    plt.xticks(range(len(sequence1)), list(sequence1))
    plt.savefig('predicted_interaction_scores_sequence1.png')
    plt.close()

    # Plot predicted interaction scores along the sequence2
    plt.figure(figsize=(12, 6))
    plt.plot(predictions2[:len(sequence2)])  # Plot only for the original sequence length
    plt.title('Predicted Interaction Scores Along the Sequence 2')
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Interaction Score')
    plt.xticks(range(len(sequence2)), list(sequence2))
    plt.savefig('predicted_interaction_scores_sequence2.png')
    plt.close()

    # Save the interaction matrix to a CSV file
    df = pd.DataFrame(interaction_matrix)
    df.to_csv('predicted_interaction_matrix.csv', index=False, header=False)

    return interaction_matrix

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Predict protein interaction scores")
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--sequence1', required=True, help='First amino acid sequence to predict')
    parser.add_argument('--sequence2', required=True, help='Second amino acid sequence to predict')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--phys_prop_file', default='data/transformed_physicochemical_properties.csv', help='Path to physicochemical properties file')
    args = parser.parse_args()

    interaction_matrix = predict(args.model, args.sequence1, args.sequence2, args.config, args.phys_prop_file)
    print(f'Sequence 1: {args.sequence1}')
    print(f'Sequence 2: {args.sequence2}')
    print('Predicted Interaction Matrix:')
    print(interaction_matrix)
