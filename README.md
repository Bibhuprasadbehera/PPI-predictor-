# Protein Interaction Prediction

This project aims to predict intra-protein interactions using amino acid sequences. The model uses an LSTM neural network implemented in PyTorch.

## Project Structure

protein_interaction_prediction/
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── validation.csv
├── checkpoints/
│ └── model_epoch_10.pth
├── logs/
│ └── training.log
├── outputs/
│ └── predictions.csv
├── notebooks/
│ └── eda.ipynb
├── src/
│ ├── main.py
│ ├── data_loader.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ └── utils.py
├── tests/
│ └── test_data_loader.py
├── config.yaml
├── requirements.txt
└── README.md

perl


## Setup

1. Clone the repository:
    ```bash
    git clone <repository-url>
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
