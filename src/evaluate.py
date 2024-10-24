# src/evaluate.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from model import ProteinInteractionModel
from data_loader import ProteinDataset, visualize_batch
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model_path, test_data_dir, phys_prop_file, config):
    print("Loading configuration...")
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    print("Initializing model...")
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'],
                                    cfg['model']['phys_prop_size'])

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
        for sequences, rsas, secondary_structures, phys_props, labels in tqdm(test_loader, desc='Evaluating'):
            outputs = model(sequences, rsas, secondary_structures, phys_props)
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

    # Plotting
    plt.figure(figsize=(20, 20))

    # 1. Predicted vs Actual Scatter Plot
    plt.subplot(2, 2, 1)
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', lw=2)
    plt.xlabel('Actual Interaction Score')
    plt.ylabel('Predicted Interaction Score')
    plt.title('Predicted vs Actual Interaction Scores')

    # 2. Residuals Distribution
    plt.subplot(2, 2, 2)
    residuals = all_preds - all_labels
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Distribution of Residuals')

    # 3. Q-Q Plot
    plt.subplot(2, 2, 3)
    from scipy.stats import probplot
    probplot(residuals, plot=plt)
    plt.title('Q-Q Plot of Residuals')

    # 4. Residuals vs Predicted Values
    plt.subplot(2, 2, 4)
    plt.scatter(all_preds, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.axhline(y=0, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig('evaluation_plots.png')
    plt.close()

    # 5. Actual vs Predicted Line Plot
    plt.figure(figsize=(15, 5))
    sorted_indices = np.argsort(all_labels)
    plt.plot(all_labels[sorted_indices], label='Actual', alpha=0.7)
    plt.plot(all_preds[sorted_indices], label='Predicted', alpha=0.7)
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Interaction Score')
    plt.title('Actual vs Predicted Interaction Scores (Sorted)')
    plt.legend()
    plt.savefig('actual_vs_predicted_line.png')
    plt.close()

    # 6. Error Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(all_labels, shade=True, label='Actual')
    sns.kdeplot(all_preds, shade=True, label='Predicted')
    plt.xlabel('Interaction Score')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Scores')
    plt.legend()
    plt.savefig('score_distribution.png')
    plt.close()

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
