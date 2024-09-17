# src/predict.py

import torch
import yaml
from model import ProteinInteractionModel
import warnings

# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def predict(model_path, sequence, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'])
    
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

    aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    aa_to_index[''] = 20  # Padding token
    
    padded_sequence = '' * 3 + sequence + '' * 3
    predictions = []

    for i in range(3, len(padded_sequence) - 3):
        window = padded_sequence[i-3:i+4]
        sequence_tensor = torch.tensor([aa_to_index.get(aa, 20) for aa in window], dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(sequence_tensor)
        predictions.append(prediction.item())

    return predictions[:len(sequence)]  # Only return predictions for the original sequence length

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Predict protein interaction scores")
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--sequence', required=True, help='Amino acid sequence to predict')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    predictions = predict(args.model, args.sequence, args.config)
    print(f'Sequence: {args.sequence}')
    print('Predicted Interaction Scores:')
    for i, score in enumerate(predictions):
        print(f'Position {i+1} ({args.sequence[i]}): {score:.4f}')