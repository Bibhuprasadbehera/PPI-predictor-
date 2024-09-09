# src/predict.py

import torch
import yaml
from model import ProteinInteractionModel

def predict(model_path, sequence, secondary_structure, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    aa_to_index[''] = 20  # Padding token
    ss_to_index = {'C': 0, 'H': 1, 'E': 2, 'P': 3}

    padded_sequence = '' * 3 + sequence + '' * 3
    padded_ss = 'C' * 3 + secondary_structure + 'C' * 3

    predictions = []
    for i in range(3, len(padded_sequence) - 3):
        window = padded_sequence[i-3:i+4]
        ss_window = padded_ss[i-3:i+4]
        sequence_tensor = torch.tensor([aa_to_index.get(aa, 20) for aa in window], dtype=torch.long).unsqueeze(0)
        ss_tensor = torch.tensor([ss_to_index.get(ss, 0) for ss in ss_window], dtype=torch.long).unsqueeze(0)
        rsa_tensor = torch.tensor([0.5], dtype=torch.float32)  # Using a default RSA value

        with torch.no_grad():
            prediction = model(sequence_tensor, rsa_tensor, ss_tensor)
            predictions.append(prediction.item())

    return predictions[:len(sequence)]  # Only return predictions for the original sequence length

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Predict protein interaction scores")
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--sequence', required=True, help='Amino acid sequence to predict')
    parser.add_argument('--secondary_structure', required=True, help='Secondary structure of the sequence')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    predictions = predict(args.model, args.sequence, args.secondary_structure, args.config)
    print(f'Sequence: {args.sequence}')
    print(f'Secondary Structure: {args.secondary_structure}')
    print('Predicted Interaction Scores:')
    for i, score in enumerate(predictions):
        print(f'Position {i+1} ({args.sequence[i]}, {args.secondary_structure[i]}): {score:.4f}')