# src/plots_dataloader.py
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_rsa_vs_ss_plot(data):
    """Plot RSA values against secondary structure."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='three_hot_ss', y='rsa', data=data)
    plt.title('RSA Distribution by Secondary Structure')
    plt.xlabel('Secondary Structure')
    plt.ylabel('RSA')
    plt.savefig('plots/rsa_vs_ss.png')
    plt.close()

def create_chain_distribution_plot(data):
    """Plot the distribution of chain IDs."""
    plt.figure(figsize=(12, 6))
    chain_counts = data['chain'].value_counts()
    sns.barplot(x=chain_counts.index, y=chain_counts.values)
    plt.title('Distribution of Chain IDs')
    plt.xlabel('Chain ID')
    plt.ylabel('Count')
    plt.savefig('plots/chain_distribution.png')
    plt.close()

def create_physicochemical_properties_correlation_plot(phys_props):
    """Plot a correlation heatmap for physicochemical properties."""
    plt.figure(figsize=(10, 8))
    corr = phys_props.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Physicochemical Properties')
    plt.savefig('plots/physicochemical_properties_correlation.png')
    plt.close()

def create_rsa_distribution_plot(data):
    plt.figure(figsize=(12, 6))
    sns.histplot(data['rsa'], kde=True)
    plt.title('Distribution of RSA Values')
    plt.xlabel('RSA')
    plt.ylabel('Count')
    plt.savefig('plots/rsa_distribution.png')
    plt.close()

def create_secondary_structure_distribution_plot(data):
    plt.figure(figsize=(12, 6))
    ss_counts = data['three_hot_ss'].apply(lambda x: ''.join(set(x))).value_counts()
    sns.barplot(x=ss_counts.index, y=ss_counts.values)
    plt.title('Distribution of Secondary Structures')
    plt.xlabel('Secondary Structure')
    plt.ylabel('Count')
    plt.savefig('plots/secondary_structure_distribution.png')
    plt.close()

def create_amino_acid_frequency_plot(data):
    plt.figure(figsize=(12, 6))
    amino_acid_counts = data['aa'].value_counts()
    sns.barplot(x=amino_acid_counts.index, y=amino_acid_counts.values)
    plt.title('Amino Acid Frequency')
    plt.xlabel('Amino Acid')
    plt.ylabel('Count')
    plt.savefig('plots/amino_acid_frequency.png')
    plt.close()

def create_sequence_length_distribution_plot(data):
    plt.figure(figsize=(12, 6))
    sequence_lengths = data['aa'].str.len()
    sns.histplot(sequence_lengths, kde=True)
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig('plots/sequence_length_distribution.png')
    plt.close()

def create_physicochemical_properties_distribution_plots(phys_props):

    # Create directory for physicochemical properties plots if it doesn't exist
    os.makedirs("plots/physicochemical_properties_distribution", exist_ok=True)

    # Create distribution plots for each property
    for property in phys_props.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(phys_props[property], kde=True)
        plt.title(f'Distribution of {property}')
        plt.xlabel(property)
        plt.ylabel('Frequency')
        plt.savefig(f'plots/physicochemical_properties_distribution/{property}_distribution.png')
        plt.close()

    print("Distribution plots for physicochemical properties generated successfully.")

def create_batch_visualization(batch, num_samples=5):
    sequences, rsas, secondary_structures, phys_props, chains, labels = batch  # Unpack chains
    fig, axs = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))  # Add a column for chains
    for i in range(num_samples):
        seq = sequences[i].numpy()
        rsa = rsas[i].numpy()
        ss = secondary_structures[i].numpy()
        chain = chains[i].numpy()

        axs[i, 0].imshow(np.eye(20)[seq], aspect='auto', cmap='viridis')
        axs[i, 0].set_title(f'Sample {i+1} - Sequence (One-hot encoded)')
        axs[i, 0].set_ylabel('AA Index')
        axs[i, 0].set_xlabel('Position')

        axs[i, 1].plot(rsa.repeat(len(seq)))
        axs[i, 1].set_title(f'Sample {i+1} - RSA Value')
        axs[i, 1].set_ylabel('RSA')
        axs[i, 1].set_xlabel('Position')

        axs[i, 2].imshow(np.eye(4)[ss], aspect='auto', cmap='viridis')
        axs[i, 2].set_title(f'Sample {i+1} - Secondary Structure')
        axs[i, 2].set_ylabel('SS Index')
        axs[i, 2].set_xlabel('Position')

        axs[i, 3].plot(chain.repeat(len(seq)))  # Visualize chain IDs
        axs[i, 3].set_title(f'Sample {i+1} - Chain ID')
        axs[i, 3].set_ylabel('Chain ID')
        axs[i, 3].set_xlabel('Position')

    plt.tight_layout()
    plt.savefig('plots/batch_visualization.png')
    plt.close()
    print(f"Batch shape - Sequences: {sequences.shape}, RSAs: {rsas.shape}, Secondary Structures: {secondary_structures.shape}, Physicochemical Properties: {phys_props.shape}, Chains: {chains.shape}, Labels: {labels.shape}")  # Include chains
    print(f"Label values: {labels[:num_samples]}")
