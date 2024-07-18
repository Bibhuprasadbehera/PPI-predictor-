# src/evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import ProteinInteractionModel
from data_loader import ProteinDataset

def evaluate(model_path, test_data_path, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'], cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataset = ProteinDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to('cuda')
            outputs = model(sequences)
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

if __name__ == '__main__':
    evaluate('checkpoints/model_epoch_10.pth', 'data/test.csv', 'config.yaml')
