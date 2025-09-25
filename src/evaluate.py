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
from scipy.stats import pearsonr, spearmanr

def evaluate(model_path, test_data_dir, phys_prop_file, config, max_batches=None):
    print("Loading configuration...")
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    print("Initializing model...")
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['phys_prop_size'],
                                    cfg['model']['num_chains'], cfg['model']['motif_feature_size'])

    # Load the state dict
    state_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    print(model)

    print("Loading test data...")
    test_dataset = ProteinDataset(test_data_dir, phys_prop_file, normalize_distance=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'],
                             shuffle=False, num_workers=cfg['data']['num_workers'],
                             collate_fn=ProteinDataset.collate_fn)
    
    # Limit the number of batches if specified
    if max_batches:
        print(f"Limited to first {max_batches} batches for quick evaluation")
        test_loader = list(test_loader)[:max_batches]

    print("Running evaluation...")
    
    # Calculate metrics batch by batch to avoid memory issues
    total_squared_error = 0
    total_absolute_error = 0
    total_samples = 0
    
    all_preds_list = []
    all_targets_list = []
    
    with torch.no_grad():
        # Handle both DataLoader and list of batches
        batch_iter = test_loader if isinstance(test_loader, list) else tqdm(test_loader, desc='Evaluating')
        for batch in batch_iter:
            # Use all features during evaluation
            outputs = model(
                batch['sequence'], batch['rsa'], batch['ss'],
                batch['phys_props'], batch['chain'],
                batch['motif_binary'], batch['motif_index'],
                batch['motif_position'], batch['motif_overlap'],
                use_all_features=True
            )
            
            # Calculate batch metrics
            batch_outputs = outputs.cpu().numpy()
            batch_targets = batch['distance_mat'].cpu().numpy()
            
            # Calculate accumulated metrics for overall average
            flat_outputs = batch_outputs.ravel()
            flat_targets = batch_targets.ravel()
            
            n_samples = len(flat_outputs)
            total_squared_error += np.sum((flat_outputs - flat_targets) ** 2)
            total_absolute_error += np.sum(np.abs(flat_outputs - flat_targets))
            total_samples += n_samples
            
            # For correlation metrics, only store a sample to save memory
            if len(all_preds_list) < 5:  # Only store first few batches for correlation
                all_preds_list.append(flat_outputs)
                all_targets_list.append(flat_targets)

    # Calculate overall metrics
    mse = total_squared_error / total_samples
    mae = total_absolute_error / total_samples
    rmse = np.sqrt(mse)
    
    # Concatenate only the sampled data for correlation calculations
    if all_preds_list:
        all_preds = np.concatenate(all_preds_list)
        all_targets = np.concatenate(all_targets_list)
    else:
        all_preds = np.array([])
        all_targets = np.array([])
    
    # Calculate correlation coefficients using sampled data
    if len(all_preds) > 0 and len(all_preds) == len(all_targets):
        from scipy.stats import pearsonr, spearmanr
        
        # Calculate Pearson correlation
        if len(all_preds) > 1:  # Need at least 2 points for correlation
            pearson_corr, _ = pearsonr(all_preds, all_targets)
            spearman_corr, _ = spearmanr(all_preds, all_targets)
        else:
            pearson_corr = 0.0
            spearman_corr = 0.0
    else:
        pearson_corr = 0.0
        spearman_corr = 0.0
    
    # Calculate R2 using sampled data
    if len(all_preds) > 1:
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    else:
        r2 = 0.0
    
    # Create metrics dictionary similar to calculate_metrics function
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr
    }

    print(f'Mean Squared Error: {metrics["mse"]:.4f}')
    print(f'Root Mean Squared Error: {metrics["rmse"]:.4f}')
    print(f'Mean Absolute Error: {metrics["mae"]:.4f}')
    print(f'R2 Score: {metrics["r2"]:.4f}')
    print(f'Pearson Correlation Coefficient: {metrics["pearson_corr"]:.4f}')
    print(f'Spearman Correlation Coefficient: {metrics["spearman_corr"]:.4f}')

    # Generate evaluation plots using sampled data
    if len(all_targets) > 0:
        create_evaluation_plots(all_targets, all_preds)
    else:
        print("Warning: No data available for plots (all samples were skipped)")

    # Return metrics for potential further use
    return metrics

if __name__ == '__main__':
    print("Starting evaluation...")
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the trained model')
    parser.add_argument('--model', default='checkpoints/best_model.pth', help='Path to trained model')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--max-batches', type=int, help='Limit evaluation to first N batches for quick testing')
    args = parser.parse_args()
    
    # Fallback to default model if specified model doesn't exist
    import os
    if args.model and not os.path.exists(args.model):
        args.model = 'checkpoints/best_model.pth' if os.path.exists('checkpoints/best_model.pth') else 'checkpoints/model_epoch_20.pth'
    
    metrics = evaluate(args.model, 'data/', 'data/physicochemical/transformed_physicochemical_properties.csv', 'config.yaml', max_batches=args.max_batches)
    print("Evaluation complete.")
    print("Summary of metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
