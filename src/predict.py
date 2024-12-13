import torch
import yaml
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_dssp, load_distance_matrix, map_dssp_to_distance_matrix
from model import ProteinInteractionModel
from utils import setup_logger

# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def predict(model_path, sequence_1, sequence_2, dssp_dir_1, dssp_dir_2, distance_matrix_path, config, phys_prop_file):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'],
                                    cfg['model']['phys_prop_size']).to(device)

    # Load model state dict
    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()

    # Load DSSP data for both sequences
    dssp_data_1 = load_dssp(dssp_dir_1)
    dssp_data_2 = load_dssp(dssp_dir_2)

    if dssp_data_1 is None or dssp_data_2 is None:
        print("Error: Could not load DSSP data. Check file paths and format.")
        return

    # Load distance matrix
    distance_matrix = load_distance_matrix(distance_matrix_path)

    if distance_matrix is None:
        print("Error: Could not load distance matrix. Check file path and format.")
        return
    
    # Create interaction matrix from distance matrix
    interaction_matrix = (distance_matrix < 5).astype(int)

    # Map DSSP data to distance matrix
    enhanced_distance_matrix = map_dssp_to_distance_matrix(dssp_data_1, dssp_data_2, distance_matrix, interaction_matrix)

    # Convert data to tensors and move to device
    dssp_data_1 = tuple(d.to(device) for d in dssp_data_1)
    dssp_data_2 = tuple(d.to(device) for d in dssp_data_2)
    enhanced_distance_matrix = torch.tensor(enhanced_distance_matrix, dtype=torch.float32).unsqueeze(0).to(device)
    interaction_matrix = torch.tensor(interaction_matrix, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(dssp_data_1, dssp_data_2, enhanced_distance_matrix, interaction_matrix)

    predictions = predictions.squeeze(0).cpu().numpy()

    # Plot predicted interaction matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(predictions, cmap='viridis', origin='lower')
    plt.title('Predicted Interaction Matrix')
    plt.xlabel('Amino Acid Position (Protein 2)')
    plt.ylabel('Amino Acid Position (Protein 1)')
    plt.colorbar(label='Interaction Score')
    plt.savefig('predicted_interaction_matrix.png')
    plt.close()

    return predictions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Predict protein interaction scores")
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--sequence_1', required=True, help='Amino acid sequence for protein 1')
    parser.add_argument('--sequence_2', required=True, help='Amino acid sequence for protein 2')
    parser.add_argument('--dssp_dir_1', required=True, help='Path to the DSSP file for protein 1')
    parser.add_argument('--dssp_dir_2', required=True, help='Path to the DSSP file for protein 2')
    parser.add_argument('--distance_matrix_path', required=True, help='Path to the distance matrix file')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--phys_prop_file', default='data/transformed_physicochemical_properties.csv', help='Path to physicochemical properties file')
    args = parser.parse_args()

    predictions = predict(args.model, args.sequence_1, args.sequence_2, args.dssp_dir_1, args.dssp_dir_2, args.distance_matrix_path, args.config, args.phys_prop_file)
    print(f'Sequence 1: {args.sequence_1}')
    print(f'Sequence 2: {args.sequence_2}')
    print('Predicted Interaction Matrix saved to predicted_interaction_matrix.png')
