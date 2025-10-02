# src/main.py 
import argparse
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import train
from evaluate import evaluate
from predict import SequencePredictor

def main():
    parser = argparse.ArgumentParser(description="Protein Interaction Prediction")
    parser.add_argument('action', choices=['train', 'evaluate', 'predict'], help='Action to perform')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--model', help='Path to model file (for evaluate and predict)')
    parser.add_argument('--sequence', help='Amino acid sequence (for predict)')
    parser.add_argument('--sequence2', help='Second amino acid sequence for PPI prediction (for predict)')
    args = parser.parse_args()

    if args.action == 'train':
        train(args.config)
    elif args.action == 'evaluate':
        if not args.model:
            print("Please provide a model path for evaluation")
            return
        evaluate(args.model, args.config)
    elif args.action == 'predict':
        if not args.model or not args.sequence:
            print("Please provide a model path and sequence for prediction")
            return
        # Use SequencePredictor for prediction
        predictor = SequencePredictor(args.model, args.config)
        if args.sequence2:
            # Predict interaction between two sequences
            interaction_matrix, interaction_prob = predictor.predict(args.sequence, args.sequence2)
        else:
            # Predict intra-protein interactions for single sequence
            interaction_matrix, interaction_prob = predictor.predict(args.sequence)
        print(f"Prediction: Interaction matrix shape: {interaction_matrix.shape}")
        print(f"Interaction probability range: {interaction_prob.min():.3f} - {interaction_prob.max():.3f}")

if __name__ == "__main__":
    main()
