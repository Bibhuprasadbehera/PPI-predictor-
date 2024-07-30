#src/evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import ProteinInteractionModel
from data_loader import ProteinDataset
import yaml
import pandas as pd
import numpy as np

class SingleFileDataset(ProteinDataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        print(f"Columns in the dataset: {self.data.columns.tolist()}")
        
        aa_column = self._find_column(['aa', 'amino_acid', 'sequence'])
        rsa_column = self._find_column(['rsa', 'relative_solvent_accessibility'])
        
        if aa_column is None:
            raise KeyError(f"Could not find a column representing amino acid sequences. Available columns: {self.data.columns.tolist()}")
        if rsa_column is None:
            raise KeyError(f"Could not find a column representing RSA values. Available columns: {self.data.columns.tolist()}")
        
        print(f"Using '{aa_column}' for amino acid sequences and '{rsa_column}' for RSA values.")
        
        self.sequences = self.data[aa_column].apply(self.one_hot_encode).values
        self.rsa = self.data[rsa_column].values.astype(float)
        self.labels = self.calculate_labels()

    def _find_column(self, possible_names):
        for name in possible_names:
            matching_columns = [col for col in self.data.columns if name.lower() in col.lower()]
            if matching_columns:
                return matching_columns[0]
        return None

    def calculate_labels(self):
        rsa_threshold = np.percentile(self.rsa, 90)
        return (self.rsa > rsa_threshold).astype(int)

    def one_hot_encode(self, sequence):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        encoding = np.zeros((len(sequence), len(amino_acids)))
        for i, aa in enumerate(sequence):
            if aa in amino_acids:
                encoding[i, amino_acids.index(aa)] = 1
        return encoding.flatten()

def evaluate(model_path, test_data_path, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    print(f"Loading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data columns: {test_data.columns.tolist()}")
    
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'], cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    try:
        test_dataset = SingleFileDataset(test_data_path)
    except KeyError as e:
        print(f"Error loading test dataset: {e}")
        print("Please ensure that your test data file contains columns for amino acid sequences and RSA values.")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

if __name__ == '__main__':
    evaluate('checkpoints/model_epoch_10.pth', 'data/test.csv', 'config.yaml')