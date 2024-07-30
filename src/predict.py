#src/predict.py
import torch
import pandas as pd
from model import ProteinInteractionModel
from data_loader import ProteinDataset
import yaml

def predict(model_path, sequence, rsa, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'], cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset = ProteinDataset('')  # Empty string as we don't need to load data here
    one_hot_sequence = dataset.one_hot_encode(sequence)
    feature = torch.cat([torch.tensor(one_hot_sequence, dtype=torch.float32), torch.tensor([rsa], dtype=torch.float32)])
    feature = feature.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(feature)
        prediction = (output > 0.5).float().item()
        return prediction

if __name__ == '__main__':
    sequence = "A"  # Example amino acid
    rsa = 0.95  # Example RSA value
    prediction = predict('checkpoints/model_epoch_10.pth', sequence, rsa, 'config.yaml')
    print(f'Sequence: {sequence}, RSA: {rsa}, Prediction: {prediction}')