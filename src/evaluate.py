import yaml
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import ProteinInteractionModel
from scipy.stats import pearsonr, spearmanr
from data_loader import ProteinDataset, visualize_batch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from plots_evaluate import create_evaluation_plots

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
    test_dataset = ProteinDataset(test_data_dir, phys_prop_file)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'],
                             shuffle=False, num_workers=cfg['data']['num_workers'],
                             collate_fn=ProteinDataset.collate_fn)

    print("Visualizing a batch from the test set...")
    for batch in test_loader:
        visualize_batch(batch)
        break

    all_labels = []
    all_preds = []

    print("Running evaluation...")
    with torch.no_grad():
        for sequences, rsas, secondary_structures, phys_props, chains, labels in tqdm(test_loader, desc='Evaluating'):  # Unpack chains
            outputs = model(sequences, rsas, secondary_structures, phys_props, chains)  # Pass chains to the model
            all_labels.extend(labels.numpy().flatten())
            all_preds.extend(outputs.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Remove any rows with NaN or inf values
    valid_indices = np.isfinite(all_labels) & np.isfinite(all_preds)
    all_labels = all_labels[valid_indices]
    all_preds = all_preds[valid_indices]

    print(f"Shape of all_labels: {all_labels.shape}")
    print(f"Shape of all_preds: {all_preds.shape}")
    print(f"Number of valid samples: {np.sum(valid_indices)}")
    print(f"Sample of all_labels: {all_labels[:5]}")
    print(f"Sample of all_preds: {all_preds[:5]}")

    # Calculate metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    try:
        pearson_corr, _ = pearsonr(all_labels, all_preds)
    except:
        print("Warning: Could not calculate Pearson correlation. Setting to NaN.")
        pearson_corr = np.nan

    try:
        spearman_corr, _ = spearmanr(all_labels, all_preds)
    except:
        print("Warning: Could not calculate Spearman correlation. Setting to NaN.")
        spearman_corr = np.nan

    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'R2 Score: {r2:.4f}')
    print(f'Pearson Correlation Coefficient: {pearson_corr:.4f}')
    print(f'Spearman Correlation Coefficient: {spearman_corr:.4f}')

    create_evaluation_plots(all_labels, all_preds)

    # Return metrics for potential further use
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr
    }

if __name__ == '__main__':
    print("Starting evaluation...")
    metrics = evaluate('checkpoints/model_epoch_20.pth', 'data/', 'data/transformed_physicochemical_properties.csv', 'config.yaml')
    print("Evaluation complete.")
    print("Summary of metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
