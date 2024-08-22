#src/evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from model import ProteinInteractionModel
from data_loader import ProteinDataset
import yaml
import numpy as np

def evaluate(model_path, test_data_dir, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_dataset = ProteinDataset(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'],
                             shuffle=False, num_workers=cfg['data']['num_workers'],
                             collate_fn=ProteinDataset.collate_fn)
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for sequences, rsas, labels in test_loader:
            outputs = model(sequences, rsas)
            all_labels.extend(labels.numpy())
            all_preds.extend(outputs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')

if __name__ == '__main__':
    evaluate('checkpoints/model_epoch_10.pth', 'data/', 'config.yaml')