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
   python src/predict.py --model checkpoints/model_epoch_20.pth --sequence1 NKVQMHRSEMRPKFFSEHIISILNPHCVV --sequence2 NKVQMHRSEMRPKFFSEHIISILNPHCVV --config config.yaml
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




# if facing environment related issue please use this 
1. 
/mnt/myssd/anaconda3/envs/ml/bin/python /home/bibhuprasad/Documents/PPI\ prediction\ model/PPI-predictor-/src/predict.py --model checkpoints/model_epoch_20.pth --sequence NKVQMHRSEMRPKFFSEHIISILNPHCVV --config config.yaml




first run the rsa_ss_merger


then do the modify 1 by adding a test interaction score 


Major Issues with Training:

  1. Unstable Training Loss
  The logs show wildly inconsistent training behavior:
   - Some runs have extremely low loss (0.0000) which suggests the model may be predicting constants or there's an
     issue with loss calculation
   - Other runs show extremely high losses (3900+) which indicates numerical instability or exploding gradients
   - The validation loss often remains constant at 3908.5627, suggesting the model isn't learning at all in these cases

  2. Potential Data/Target Issues
  Looking at the model architecture and training setup:
   - The model is designed to predict distance matrices (output shape: B, L, L)
   - However, the loss calculation criterion(output, distance_matrices) might have dimension mismatches
   - The target distance matrices may not be properly normalized or scaled

  3. Architecture Problems
  In model.py, there are several issues:
   - The forward method has inconsistent tensor dimensions
   - The distance matrix processing has potential shape mismatch issues
   - The embedding concatenation may not align properly

  4. Data Loading Issues
  In data_loader.py:
   - The distance matrix loading may not be working correctly
   - Padding of distance matrices to (max_len, max_len) might not preserve the actual data structure

  5. Loss Function Mismatch
  The model uses MSELoss for distance matrix prediction, but:
   - Distance matrices typically have specific statistical properties
   - May need normalization or specialized loss functions for distance prediction

  Recommendations to Fix These Issues:

  1. Fix the Model Architecture

   1 # In model.py, ensure proper tensor shapes throughout the forward pass
   2 # The output should match the shape of the target distance matrices

  2. Normalize Distance Matrices

   1 # In data_loader.py, normalize distance matrices to a reasonable range
   2 # This will prevent numerical instability

  3. Add Gradient Clipping

   1 # In train.py, add gradient clipping to prevent exploding gradients:
   2 # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

  4. Check Data Shapes

   1 # Add debugging prints to verify tensor shapes at each step
   2 # Especially important for the distance matrices

  5. Implement Proper Loss Function

   1 # Consider using a more appropriate loss for distance matrix prediction
   2 # Possibly a weighted loss that emphasizes certain distance ranges




Relationship Between DSSP and Distance Matrices

  1. File Pairing
   - DSSP files (e.g., 1AVO_C_D_dssp.tsv) contain sequential information for protein chains
   - Distance matrix files (e.g., 1AVO_C_D_ca.tsv) contain spatial distance information between all pairs of Cα atoms
   - Files are paired by their naming convention: protein_chain1_chain2_dssp.tsv ↔ protein_chain1_chain2_ca.tsv

  2. Data Integration in the Pipeline
  In data_loader.py, the system:
   1. Reads DSSP files to get sequential features (amino acid, RSA, secondary structure, chain)
   2. For each DSSP entry, loads the corresponding distance matrix file
   3. Associates sequential features with spatial distance information

  3. Feature Mapping
  The DSSP file contains:
   - Amino acid sequence (aa column)
   - Relative solvent accessibility (rsa column)
   - Secondary structure (three_hot_ss column)
   - Chain identifier (chain column)

  The distance matrix contains:
   - Pairwise distances between all amino acids in the sequence
   - Indexed by amino acid positions (e.g., A-103-D, V-104-D)

  4. Matching Issues
  There appears to be a mismatch in how the data is being used:

   1. Indexing mismatch: The DSSP file has sequential indices (203, 204, 205...) while the distance matrix has position
      labels (A-103-D, V-104-D...). These don't directly correspond.

   2. Length mismatch: The DSSP file shows 202 rows while the distance matrix shows 62 rows, suggesting they may not be
      covering the same sequence regions.

   3. Chain handling: The DSSP file has a 'chain' column, but it's not clear how this maps to the distance matrix
      indexing.

  5. Model Training Issues
  The current implementation has several problems:
   - The model is trying to predict entire distance matrices, but the input features are sequential
   - There's no clear mechanism to ensure that the sequential features from DSSP match the correct positions in the
     distance matrix
   - The padding approach for distance matrices may not preserve the spatial relationships correctly

  This mismatch is likely contributing to the unstable training behavior we observed in the logs, where losses are
  either extremely high or extremely low, indicating the model isn't properly learning the relationship between
  sequential features and spatial distances.