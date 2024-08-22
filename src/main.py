# Main script 
import argparse
from src.train import train
from src.evaluate import evaluate
from src.predict import predict

def main():
    parser = argparse.ArgumentParser(description="Protein Interaction Prediction")
    parser.add_argument('action', choices=['train', 'evaluate', 'predict'], help='Action to perform')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--model', help='Path to model file (for evaluate and predict)')
    parser.add_argument('--sequence', help='Amino acid sequence (for predict)')
    args = parser.parse_args()

    if args.action == 'train':
        train(args.config)
    elif args.action == 'evaluate':
        if not args.model:
            print("Please provide a model path for evaluation")
            return
        evaluate(args.model, 'data/test', args.config)
    elif args.action == 'predict':
        if not all([args.model, args.sequence]):
            print("Please provide a model path and sequence for prediction")
            return
        prediction = predict(args.model, args.sequence, args.config)
        print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()