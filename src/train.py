# src/train.py
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loader
from model import ProteinInteractionModel
import os
from sklearn.model_selection import train_test_split

def train(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load all data
    full_dataset = get_data_loader(cfg['data']['train_path'], cfg['data']['batch_size'], cfg['data']['num_workers'])

    # Split the dataset
    train_size = int(0.8 * len(full_dataset.dataset))
    val_size = len(full_dataset.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset.dataset, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    model = ProteinInteractionModel(cfg['model']['input_size'], cfg['model']['hidden_size'], cfg['model']['num_layers'], cfg['model']['output_size'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    num_epochs = cfg['training']['num_epochs']
   
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = (output > 0.5).float()
                correct += pred.eq(target).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_dataset)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save checkpoint
        checkpoint_path = os.path.join(cfg['training']['checkpoint_dir'], f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train('config.yaml')