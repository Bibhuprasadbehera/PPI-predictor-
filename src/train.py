# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import ProteinInteractionModel
from data_loader.py import get_data_loader
import yaml
import os

def train(config):
    # Load configuration
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)

    # Load data
    train_loader = get_data_loader(cfg['data']['train_path'], cfg['data']['batch_size'], cfg['data']['num_workers'])
    val_loader = get_data_loader(cfg['data']['val_path'], cfg['data']['batch_size'], cfg['data']['num_workers'])

    # Model, loss, optimizer
    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'], cfg['model']['num_layers'], cfg['model']['output_size'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    # Training loop
    writer = SummaryWriter(cfg['training']['log_dir'])
    model.train()

    for epoch in range(cfg['training']['num_epochs']):
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to('cuda'), labels.to('cuda')
            outputs = model(sequences)
            loss = criterion(outputs, labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{cfg['training']['num_epochs']}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('training_loss', loss.item(), epoch * len(train_loader) + i)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for sequences, labels in val_loader:
                sequences, labels = sequences.to('cuda'), labels.to('cuda')
                outputs = model(sequences)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
            val_loss /= len(val_loader)
            writer.add_scalar('validation_loss', val_loss, epoch)
            print(f'Epoch [{epoch + 1}/{cfg['training']['num_epochs']}], Validation Loss: {val_loss:.4f}')

        # Save checkpoint
        checkpoint_path = os.path.join(cfg['training']['checkpoint_dir'], f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    writer.close()

if __name__ == '__main__':
    train('config.yaml')
