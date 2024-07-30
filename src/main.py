# Main script 
import argparse
from train import train
from evaluate import evaluate
from predict import predict

def main():
    parser = argparse.ArgumentParser(description="Protein Interaction Prediction")
    parser.add_argument('action', choices=['train', 'evaluate', 'predict'], help='Action to perform')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--model', help='Path to model file (for evaluate and predict)')
    parser.add_argument('--sequence', help='Amino acid sequence (for predict)')
    parser.add_argument('--rsa', type=float, help='RSA value (for predict)')
    args = parser.parse_args()

    if args.action == 'train':
        train(args.config)
    elif args.action == 'evaluate':
        if not args.model:
            print("Please provide a model path for evaluation")
            return
        evaluate(args.model, 'data/test', args.config)
    elif args.action == 'predict':
        if not all([args.model, args.sequence, args.rsa]):
            print("Please provide a model path, sequence, and RSA value for prediction")
            return
        prediction = predict(args.model, args.sequence, args.rsa, args.config)
        print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()