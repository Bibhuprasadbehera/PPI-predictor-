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
    chain_to_index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'q': 50, 'r': 51, 's': 52, 'u': 53, 'w': 54}

    # Prepare input tensors for sequence1 and sequence2
    sequence1_tensor = torch.tensor([aa_to_index[aa] for aa in sequence1], dtype=torch.long).unsqueeze(0)
    sequence2_tensor = torch.tensor([aa_to_index[aa] for aa in sequence2], dtype=torch.long).unsqueeze(0)
    rsa_tensor = torch.tensor([0.5] * len(sequence1), dtype=torch.float32).unsqueeze(0)  # Placeholder RSA
    ss_tensor = torch.tensor([0] * len(sequence1), dtype=torch.long).unsqueeze(0)  # Placeholder SS
    chain_tensor = torch.tensor([chain_to_index['A']] * len(sequence1), dtype=torch.long).unsqueeze(0)  # Placeholder chain
    phys_props_list = [phys_props_df.loc[aa].values for aa in sequence1]
    phys_props_array = np.array(phys_props_list)
    phys_props_tensor = torch.tensor(phys_props_array, dtype=torch.float32).unsqueeze(0)
    contact_map_tensor = torch.zeros((1, len(sequence1), len(sequence2)), dtype=torch.float32)  # Placeholder contact map

    print(f"Sequence tensor shape: {sequence_tensor.shape}")
    print(f"RSA tensor shape: {rsa_tensor.shape}")
    print(f"Secondary structure tensor shape: {ss_tensor.shape}")
    print(f"Chain tensor shape: {chain_tensor.shape}")
    print(f"Physicochemical properties tensor shape: {phys_props_tensor.shape}")
    print(f"Contact map tensor shape: {contact_map_tensor.shape}")

    with torch.no_grad():
        predictions = model(sequence1_tensor, rsa_tensor, ss_tensor, phys_props_tensor, chain_tensor, contact_map_tensor)

    predictions = predictions.squeeze(0).cpu().numpy()

    # Plot predicted contact map
    plt.figure(figsize=(12, 6))
    plt.imshow(predictions, cmap='viridis', aspect='auto')
    plt.title('Predicted Contact Map')
    plt.xlabel('Sequence 2 Position')
    plt.ylabel('Sequence 1 Position')
    plt.colorbar(label='Interaction Score')
    plt.savefig('predicted_contact_map.png')
    plt.close()

    return predictions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Predict protein contact map")
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--sequence1', required=True, help='First amino acid sequence')
    parser.add_argument('--sequence2', required=True, help='Second amino acid sequence')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--phys_prop_file', default='data/transformed_physicochemical_properties.csv', help='Path to physicochemical properties file')
    args = parser.parse_args()

    predictions = predict(args.model, args.sequence1, args.sequence2, args.config, args.phys_prop_file)
    print('Predicted Contact Map:')
    print(predictions)