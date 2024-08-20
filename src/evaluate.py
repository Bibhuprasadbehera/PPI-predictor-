#src/evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import ProteinInteractionModel
import yaml
import pandas as pd
import numpy as np
import os

class DSSPDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data = self.load_dssp_files(data_dir)
        self.sequences = self.data.groupby('Protein_id')['aa'].apply(list).values
        self.rsa = self.data.groupby('Protein_id')['rsa'].apply(list).values
        self.labels = self.calculate_labels()

    def load_dssp_files(self, data_dir):
        all_data = []
        for file in os.listdir(data_dir):
            if file.endswith('_dssp.csv'):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                all_data.append(df)
        return pd.concat(all_data, ignore_index=True)

    def calculate_labels(self):
        all_rsa = np.concatenate(self.rsa)
        rsa_threshold = np.nanpercentile(all_rsa, 90)
        return [np.array(seq_rsa) > rsa_threshold for seq_rsa in self.rsa]

    def one_hot_encode(self, aa):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        encoding = np.zeros(20)
        if aa in amino_acids:
            encoding[amino_acids.index(aa)] = 1
        return encoding

    def __len__(self):
        return sum(len(seq) for seq in self.sequences)

    def __getitem__(self, idx):
        for seq_idx, seq_len in enumerate(map(len, self.sequences)):
            if idx < seq_len:
                break
            idx -= seq_len

        sequence = torch.tensor(self.one_hot_encode(self.sequences[seq_idx][idx]), dtype=torch.float32)
        rsa = torch.tensor([self.rsa[seq_idx][idx]], dtype=torch.float32)
        feature = torch.cat([sequence, rsa])
        label = torch.tensor(self.labels[seq_idx][idx], dtype=torch.float32)
        return feature, label

def evaluate(model_path, test_data_dir, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    print(f"Loading test data from: {test_data_dir}")
    
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'], cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_dataset = DSSPDataset(test_data_dir)
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
    evaluate('checkpoints/model_epoch_10.pth', 'data', 'config.yaml')