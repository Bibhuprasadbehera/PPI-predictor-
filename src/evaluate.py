# src/evaluate.py
import yaml
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import ProteinInteractionModel
from data_loader import ProteinDataset, visualize_batch
from plots_evaluate import create_evaluation_plots
from utils import calculate_metrics  # Import calculate_metrics from utils.py

def evaluate(model_path, test_data_dir, phys_prop_file, config):
    print("Loading configuration...")
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    print("Initializing model...")
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'],
                                    cfg['model']['phys_prop_size'], cfg['model']['num_chains'])

    # Load the state dict
    state_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    print(model)

    print("Loading test data...")
    test_dataset = ProteinDataset(test_data_dir, phys_prop_file)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'],
                             shuffle=False, num_workers=cfg['data']['num_workers'],
                             collate_fn=ProteinDataset.collate_fn)

    all_preds = []
    all_targets = []

    print("Running evaluation...")
    with torch.no_grad():
        for sequences, rsas, secondary_structures, phys_props, chains, contact_maps in tqdm(test_loader, desc='Evaluating'):
            outputs = model(sequences, rsas, secondary_structures, phys_props, chains, contact_maps)
            
            # Flatten the predictions and targets
            all_preds.extend(outputs.cpu().numpy().ravel())
            all_targets.extend(contact_maps.cpu().numpy().ravel())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics using the function from utils.py
    metrics = calculate_metrics(all_targets, all_preds)

    print(f'Mean Squared Error: {metrics["mse"]:.4f}')
    print(f'Root Mean Squared Error: {metrics["rmse"]:.4f}')
    print(f'Mean Absolute Error: {metrics["mae"]:.4f}')
    print(f'R2 Score: {metrics["r2"]:.4f}')
    print(f'Pearson Correlation Coefficient: {metrics["pearson_corr"]:.4f}')
    print(f'Spearman Correlation Coefficient: {metrics["spearman_corr"]:.4f}')

    # Return metrics for potential further use
    return metrics

if __name__ == '__main__':
    print("Starting evaluation...")
    metrics = evaluate('checkpoints/model_epoch_20.pth', 'data/', 'data/transformed_physicochemical_properties.csv', 'config.yaml')
    print("Evaluation complete.")
    print("Summary of metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")