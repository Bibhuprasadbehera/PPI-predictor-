import os
import yaml
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ProteinInteractionModel
from torch.utils.data import random_split
from data_loader import get_data_loader, ProteinDataset
from utils import setup_logger  # Import setup_logger from utils.py

def train(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Set up logger
    logger = setup_logger(cfg['training']['log_dir'] + 'training.log')

    print("Loading data...")
    full_dataset = ProteinDataset(cfg['data']['train_path'], cfg['data']['phys_prop_file'])
    
    # Split the dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['data']['batch_size'],
                                               shuffle=True, num_workers=cfg['data']['num_workers'],
                                               collate_fn=ProteinDataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['data']['batch_size'],
                                             shuffle=False, num_workers=cfg['data']['num_workers'],
                                             collate_fn=ProteinDataset.collate_fn)

    print("Initializing model...")
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'],
                                    cfg['model']['num_layers'], cfg['model']['output_size'],
                                    cfg['model']['phys_prop_size'], cfg['model']['num_chains'])  # Include num_chains
    print(model)

    criterion = nn.MSELoss()  # Use MSE loss for interaction matrix
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    num_epochs = cfg['training']['num_epochs']
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (sequences, rsas, secondary_structures, phys_props, chains, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):  # Unpack chains
            optimizer.zero_grad()

            # Separate sequences and features based on chain IDs
            chain_a_indices = (chains == 0).nonzero(as_tuple=True)[0]
            chain_b_indices = (chains == 1).nonzero(as_tuple=True)[0]

            sequences_a = sequences[chain_a_indices]
            rsas_a = rsas[chain_a_indices]
            secondary_structures_a = secondary_structures[chain_a_indices]
            phys_props_a = phys_props[chain_a_indices]
            chains_a = chains[chain_a_indices]

            sequences_b = sequences[chain_b_indices]
            rsas_b = rsas[chain_b_indices]
            secondary_structures_b = secondary_structures[chain_b_indices]
            phys_props_b = phys_props[chain_b_indices]
            chains_b = chains[chain_b_indices]

            # Pass the separated sequences and features to the model
            _, _, output = model(sequences_a, rsas_a, secondary_structures_a, phys_props_a, chains_a,
                                sequences_b, rsas_b, secondary_structures_b, phys_props_b, chains_b)

            # Flatten the target interaction matrix
            targets = targets.view(-1)

            # Calculate the loss
            loss = criterion(output.view(-1), targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Log epoch training loss
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')
        
        print("Running validation...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, rsas, secondary_structures, phys_props, chains, targets in tqdm(val_loader, desc='Validation'):  # Unpack chains

                # Separate sequences and features based on chain IDs
                chain_a_indices = (chains == 0).nonzero(as_tuple=True)[0]
                chain_b_indices = (chains == 1).nonzero(as_tuple=True)[0]

                sequences_a = sequences[chain_a_indices]
                rsas_a = rsas[chain_a_indices]
                secondary_structures_a = secondary_structures[chain_a_indices]
                phys_props_a = phys_props[chain_a_indices]
                chains_a = chains[chain_a_indices]

                sequences_b = sequences[chain_b_indices]
                rsas_b = rsas[chain_b_indices]
                secondary_structures_b = secondary_structures[chain_b_indices]
                phys_props_b = phys_props[chain_b_indices]
                chains_b = chains[chain_b_indices]

                # Pass the separated sequences and features to the model
                _, _, output = model(sequences_a, rsas_a, secondary_structures_a, phys_props_a, chains_a,
                                    sequences_b, rsas_b, secondary_structures_b, phys_props_b, chains_b)

                # Flatten the target interaction matrix
                targets = targets.view(-1)

                # Calculate the loss
                val_loss += criterion(output.view(-1), targets).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Log epoch validation loss
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')
        
        checkpoint_path = os.path.join(cfg['training']['checkpoint_dir'], f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')
    
    print("Training complete. Plotting loss...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

if __name__ == '__main__':
    train('config.yaml')
