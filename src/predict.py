#src/predict.py
import torch
from model import ProteinInteractionModel
from data_loader import ProteinDataset
import yaml

def predict(model_path, sequence, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'], 
                                    cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset = ProteinDataset('')  # Empty string as we don't need to load data here
    sequence_tensor = torch.tensor([dataset.aa_to_index.get(aa, 20) for aa in sequence], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        prediction = model.predict(sequence_tensor)
        return prediction.item()

if __name__ == '__main__':
    sequence = "ACDEFGHIKLMNPQRSTVWY"  # Example amino acid sequence
    prediction = predict('checkpoints/model_epoch_10.pth', sequence, 'config.yaml')
    print(f'Sequence: {sequence}, Predicted Interaction Score: {prediction:.4f}')