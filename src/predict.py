import torch
import yaml
from model import ProteinInteractionModel

def predict(model_path, sequence, config):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a dictionary mapping amino acids to indices
    aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    aa_to_index['<PAD>'] = 20  # Padding token

    # Convert the input sequence to a tensor
    sequence_tensor = torch.tensor([aa_to_index.get(aa, 20) for aa in sequence], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        prediction = model.predict(sequence_tensor)
        return prediction.item()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Predict protein interaction score")
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--sequence', required=True, help='Amino acid sequence to predict')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    prediction = predict(args.model, args.sequence, args.config)
    print(f'Sequence: {args.sequence}')
    print(f'Predicted Interaction Score: {prediction:.4f}')