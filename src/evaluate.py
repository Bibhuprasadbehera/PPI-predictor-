# src/evaluate.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from model import ProteinInteractionModel
from data_loader import ProteinDataset, visualize_batch
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model_path, test_data_dir, config):
    print("Loading configuration...")
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
   
    print("Initializing model...")
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(model)
   
    print("Loading test data...")
    test_dataset = ProteinDataset(test_data_dir)
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
        for sequences, rsas, labels in tqdm(test_loader, desc='Evaluating'):
            outputs = model(sequences, rsas)
            all_labels.extend(labels.numpy())
            all_preds.extend(outputs.cpu().numpy())
   
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
   
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
   
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')
    
    print("Plotting predicted vs actual values...")
    plt.figure(figsize=(10, 10))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', lw=2)
    plt.xlabel('Actual Interaction Score')
    plt.ylabel('Predicted Interaction Score')
    plt.title('Predicted vs Actual Interaction Scores')
    plt.savefig('predicted_vs_actual.png')
    plt.close()
    
    print("Plotting residuals distribution...")
    residuals = all_preds - all_labels
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Distribution of Residuals')
    plt.savefig('residuals_distribution.png')
    plt.close()

if __name__ == '__main__':
    print("Starting evaluation...")
    evaluate('checkpoints/model_epoch_10.pth', 'data/', 'config.yaml')
    print("Evaluation complete.")