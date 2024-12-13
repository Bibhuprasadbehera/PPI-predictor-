import yaml
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import ProteinInteractionModel
from data_loader import get_data_loader
from plots_evaluate import create_evaluation_plots
from utils import calculate_metrics

def evaluate(model_path, csv_file, complex_dssp_dir, distance_matrix_dir, phys_prop_file, config):
    print("Loading configuration...")
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing model...")
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'],
                                    cfg['model']['phys_prop_size']).to(device)

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)

    # Filter out keys that are not in the current model
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # Update the current model state dict
    model_dict.update(state_dict)

    # Load the updated state dict
    model.load_state_dict(model_dict)

    model.eval()
    print(model)

    print("Loading test data...")
    test_loader = get_data_loader(csv_file, complex_dssp_dir, distance_matrix_dir, phys_prop_file,
                                  cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    all_interaction_matrices = []
    all_preds = []

    print("Running evaluation...")
    with torch.no_grad():
        for dssp_data_1, dssp_data_2, enhanced_distance_matrix, interaction_matrix in tqdm(test_loader, desc='Evaluating'):
            dssp_data_1 = tuple(d.to(device) for d in dssp_data_1)
            dssp_data_2 = tuple(d.to(device) for d in dssp_data_2)
            enhanced_distance_matrix = enhanced_distance_matrix.to(device)
            interaction_matrix = interaction_matrix.to(device)

            outputs = model(dssp_data_1, dssp_data_2, enhanced_distance_matrix, interaction_matrix)
            all_interaction_matrices.extend(interaction_matrix.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    all_interaction_matrices = np.array(all_interaction_matrices)
    all_preds = np.array(all_preds)

    # Calculate metrics using the function from utils.py
    # You might need to modify this depending on the new output format
    metrics = calculate_metrics(all_interaction_matrices.flatten(), all_preds.flatten())

    print(f'Mean Squared Error: {metrics["mse"]:.4f}')
    print(f'Root Mean Squared Error: {metrics["rmse"]:.4f}')
    print(f'Mean Absolute Error: {metrics["mae"]:.4f}')
    print(f'R2 Score: {metrics["r2"]:.4f}')
    print(f'Pearson Correlation Coefficient: {metrics["pearson_corr"]:.4f}')
    print(f'Spearman Correlation Coefficient: {metrics["spearman_corr"]:.4f}')

    create_evaluation_plots(all_interaction_matrices.flatten(), all_preds.flatten())

    # Return metrics for potential further use
    return metrics

if __name__ == '__main__':
    print("Starting evaluation...")
    metrics = evaluate('checkpoints/model_epoch_20.pth', 'data/final_protein_dataset2_modified.csv', 'data/complex_dssp', 'data/distance_matrix', 'data/transformed_physicochemical_properties.csv', 'config.yaml')
    print("Evaluation complete.")
    print("Summary of metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
