# Protein Interaction Prediction

This project aims to predict intra-protein interactions using amino acid sequences. The model uses an LSTM neural network implemented in PyTorch.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Bibhuprasadbehera/PPI-predictor-.git
   cd protein_interaction_prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Check if CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

4. Prepare your data in the `data/` directory.

5. Adjust the configuration file `config.yaml` as needed.

## Protein-Protein Interaction (PPI) Prediction Workflow

1. **Prepare Data**
   - Ensure all required CSV files are in the `data/` directory
   - Verify `config.yaml` has correct paths and parameters

2. **Train Model**
   ```bash
   python src/train.py
   ```
   - Monitor training progress
   - Note the epoch with best validation performance

3. **Evaluate Model**
   ```bash
   python src/evaluate.py
   ```
   - Record MSE and R2 scores
   - Compare with baseline or previous versions

4. **Make Predictions**
   ```bash
   python src/predict.py --model checkpoints/model_epoch_20.pth --sequence NKVQMHRSEMRPKFFSEHIISILNPHCVV --config config.yaml
   ```
   - Use for individual sequences or batch processing

5. **Analyze Results**
   - Compare predictions with known interactions
   - Assess model generalization on new data

6. **Iterate and Improve (if needed)**
   - Adjust hyperparameters in `config.yaml`
   - Modify model architecture in `src/model.py`
   - Collect additional training data

7. **Deploy Model**
   - Integrate into larger bioinformatics pipeline
   - Create user interface for easy access

## Testing

Run unit tests:
```bash
python -m unittest discover tests
```

For more detailed information on each step, refer to the documentation in the `docs/` directory.