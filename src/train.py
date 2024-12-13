import os
import yaml
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ProteinInteractionModel
from data_loader import get_data_loader
from utils import setup_logger

def train(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Set up logger
    logger = setup_logger(cfg['training']['log_dir'] + 'training.log')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    print("Loading data...")
    train_loader = get_data_loader(cfg['data']['csv_file'], cfg['data']['complex_dssp_dir'], 
                                   cfg['data']['distance_matrix_dir'], cfg['data']['phys_prop_file'],
                                   cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])

    print("Initializing model...")
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'],
                                    cfg['model']['phys_prop_size']).to(device)
    print(model)

    criterion = nn.MSELoss()  # You might need to change this depending on the output format
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    num_epochs = cfg['training']['num_epochs']
    train_losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (dssp_data_1, dssp_data_2, enhanced_distance_matrix, interaction_matrix) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            dssp_data_1 = tuple(d.to(device) for d in dssp_data_1)
            dssp_data_2 = tuple(d.to(device) for d in dssp_data_2)
            enhanced_distance_matrix = enhanced_distance_matrix.to(device)
            interaction_matrix = interaction_matrix.to(device)

            optimizer.zero_grad()
            output = model(dssp_data_1, dssp_data_2, enhanced_distance_matrix, interaction_matrix)
            
            # Assuming the model now outputs a matrix of the same size as interaction_matrix
            loss = criterion(output, interaction_matrix.float()) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Log epoch training loss
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

        checkpoint_path = os.path.join(cfg['training']['checkpoint_dir'], f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

    print("Training complete. Plotting loss...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

if __name__ == '__main__':
    train('config.yaml')
