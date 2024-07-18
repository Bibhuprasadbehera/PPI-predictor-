# src/predict.py
import torch
import pandas as pd
from model import ProteinInteractionModel
from data_loader import ProteinDataset

def predict(model_path, sequence, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'], cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    one_hot_sequence = ProteinDataset('').one_hot_encode(sequence)
    sequence_tensor = torch.tensor(one_hot_sequence, dtype=torch.float32).unsqueeze(0).to('cuda')

    with torch.no_grad():
        output = model(sequence_tensor)
        prediction = (output > 0.5).float().cpu().numpy()
        return prediction

if __name__ == '__main__':
    sequence = "MVLSPADKTNVKAAW"
    prediction = predict('checkpoints/model_epoch_10.pth', sequence, 'config.yaml')
    print(f'Sequence: {sequence}, Prediction: {prediction}')
