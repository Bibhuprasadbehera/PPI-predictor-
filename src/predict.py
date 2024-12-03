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

def predict(model_path, sequence, config, phys_prop_file):
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
    chain_to_index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'q': 50, 'r': 51, 's': 52, 'u': 53, 'w': 54}  # Add more as needed

    sequence_tensor = torch.tensor([aa_to_index[aa] for aa in sequence], dtype=torch.long).unsqueeze(0)
    rsa_tensor = torch.tensor([0.5] * len(sequence), dtype=torch.float32).unsqueeze(0)  # Replace with actual RSA data
    ss_tensor = torch.tensor([0] * len(sequence), dtype=torch.long).unsqueeze(0)  # Replace with actual secondary structure data
    chain_tensor = torch.tensor([chain_to_index['A']] * len(sequence), dtype=torch.long).unsqueeze(0)  # Assuming chain ID is 'A'
    phys_props_list = [phys_props_df.loc[aa].values for aa in sequence]
    phys_props_array = np.array(phys_props_list)
    phys_props_tensor = torch.tensor(phys_props_array, dtype=torch.float32).unsqueeze(0)

    print(f"Sequence tensor shape: {sequence_tensor.shape}")
    print(f"RSA tensor shape: {rsa_tensor.shape}")
    print(f"Secondary structure tensor shape: {ss_tensor.shape}")
    print(f"Chain tensor shape: {chain_tensor.shape}")  # Print chain tensor shape
    print(f"Physicochemical properties tensor shape: {phys_props_tensor.shape}")

    with torch.no_grad():
        predictions = model(sequence_tensor, rsa_tensor, ss_tensor, phys_props_tensor, chain_tensor)  # Pass chain tensor to the model

    predictions = predictions.squeeze(0).cpu().numpy().tolist()

    # Plot predicted interaction scores along the sequence
    plt.figure(figsize=(12, 6))
    plt.plot(predictions)
    plt.title('Predicted Interaction Scores Along the Sequence')
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Interaction Score')
    plt.xticks(range(len(sequence)), list(sequence))
    plt.savefig('predicted_interaction_scores.png')
    plt.close()

    return predictions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Predict protein interaction scores")
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--sequence', required=True, help='Amino acid sequence to predict')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--phys_prop_file', default='data/transformed_physicochemical_properties.csv', help='Path to physicochemical properties file')
    args = parser.parse_args()

    predictions = predict(args.model, args.sequence, args.config, args.phys_prop_file)
    print(f'Sequence: {args.sequence}')
    print('Predicted Interaction Scores:')
    for i, (aa, score) in enumerate(zip(args.sequence, predictions)):
        print(f'Position {i+1} ({aa}): {score:.4f}')
