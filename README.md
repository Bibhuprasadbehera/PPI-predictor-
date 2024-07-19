# Protein Interaction Prediction

This project aims to predict intra-protein interactions using amino acid sequences. The model uses an LSTM neural network implemented in PyTorch.


perl


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

3. Prepare your data in the `data/` directory.

4. Adjust the configuration file `config.yaml` as needed.

## Training

Run the training script:
```bash
python src/train.py
```   

```bash
import torch
print(torch.cuda.is_available())
