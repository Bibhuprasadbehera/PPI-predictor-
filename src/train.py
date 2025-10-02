# src/train.py
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
from utils import setup_logger

def train(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Set up logger
    logger = setup_logger(cfg['training']['log_dir'] + 'training.log')

    print("Loading data...")
    full_dataset = ProteinDataset(cfg['data']['train_path'], cfg['data']['phys_prop_file'], normalize_distance=True)
    
    # Split the dataset into training and validation by protein IDs to avoid data leakage
    # Get unique protein IDs
    protein_ids = full_dataset.data['protein_id'].unique()
    import numpy as np
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(protein_ids)
    
    train_protein_count = int(0.8 * len(protein_ids))
    train_protein_ids = set(protein_ids[:train_protein_count])
    
    # Create indices for train and validation based on protein IDs
    train_indices = [i for i, row in full_dataset.data.iterrows() if row['protein_id'] in train_protein_ids]
    val_indices = [i for i, row in full_dataset.data.iterrows() if row['protein_id'] not in train_protein_ids]
    
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg['data']['batch_size'],
        shuffle=True, 
        num_workers=cfg['data']['num_workers'],
        collate_fn=ProteinDataset.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg['data']['batch_size'],
        shuffle=False, 
        num_workers=cfg['data']['num_workers'],
        collate_fn=ProteinDataset.collate_fn
    )

    print("Initializing model...")
    model = ProteinInteractionModel(
        cfg['model']['input_size'], 
        cfg['model']['hidden_size'],
        cfg['model']['num_layers'], 
        cfg['model']['phys_prop_size'],
        cfg['model']['num_chains'], 
        cfg['model']['motif_feature_size']
    )
    print(model)

    # Use MSELoss for interaction probability prediction
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    num_epochs = cfg['training']['num_epochs']
    train_losses = []
    val_losses = []

    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            optimizer.zero_grad()
            
            # Use all features during training
            output = model(
                batch['sequence'], 
                batch['rsa'], 
                batch['ss'],
                batch['phys_props'], 
                batch['chain'],
                batch['motif_binary'], 
                batch['motif_index'],
                batch['motif_position'], 
                batch['motif_overlap'],
                use_all_features=True
            )
            
            target = batch['distance_mat']
            
            # Resize target to match output if needed
            if target.shape[1:] != output.shape[1:]:
                bs, out_h, out_w = output.shape
                tgt_bs, tgt_h, tgt_w = target.shape
                new_target = torch.zeros_like(output)
                new_target[:, :min(out_h, tgt_h), :min(out_w, tgt_w)] = target[:, :min(out_h, tgt_h), :min(out_w, tgt_w)]
                target = new_target
            
            loss = criterion(output, target)
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Log epoch training loss
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}')
        
        print("Running validation...")
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                output = model(
                    batch['sequence'], 
                    batch['rsa'], 
                    batch['ss'],
                    batch['phys_props'], 
                    batch['chain'],
                    batch['motif_binary'], 
                    batch['motif_index'],
                    batch['motif_position'], 
                    batch['motif_overlap'],
                    use_all_features=True
                )
                
                target = batch['distance_mat']
                
                # Resize target to match output if needed
                if target.shape[1:] != output.shape[1:]:
                    bs, out_h, out_w = output.shape
                    tgt_bs, tgt_h, tgt_w = target.shape
                    new_target = torch.zeros_like(output)
                    new_target[:, :min(out_h, tgt_h), :min(out_w, tgt_w)] = target[:, :min(out_h, tgt_h), :min(out_w, tgt_w)]
                    target = new_target
                
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}')

        # Log epoch validation loss
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.6f}')
        
        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(cfg['training']['checkpoint_dir'], 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Best model saved to {checkpoint_path}')
        
        # Save checkpoint for every epoch
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
    plt.savefig('plots/loss_plot.png')
    plt.close()

if __name__ == '__main__':
    train('config.yaml')
