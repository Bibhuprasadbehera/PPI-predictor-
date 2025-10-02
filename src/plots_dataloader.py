# src/plots_dataloader.py
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_rsa_vs_ss_plot(data):
    """Plot RSA values against secondary structure."""
    plt.figure(figsize=(12, 6))
    
    # Create a list of paired RSA and SS values
    plot_data = []
    for _, row in data.iterrows():
        ss_chars = list(str(row['three_hot_ss']))
        rsa_values = row['rsa'] if isinstance(row['rsa'], np.ndarray) else np.array([float(row['rsa'])])
        for ss, rsa in zip(ss_chars, rsa_values):
            plot_data.append({'three_hot_ss': ss, 'rsa': rsa})
    
    plot_df = pd.DataFrame(plot_data)
    sns.boxplot(x='three_hot_ss', y='rsa', data=plot_df)
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
    plt.figure(figsize=(10, 6))
    
    # Handle both array and scalar RSA values
    rsa_values = []
    for rsa in data['rsa']:
        if isinstance(rsa, np.ndarray):
            rsa_values.extend(rsa)
        else:
            rsa_values.append(float(rsa))
            
    rsa_values = np.array(rsa_values, dtype=float)
    
    sns.histplot(data=rsa_values, kde=True)
    plt.title('Distribution of Relative Solvent Accessibility (RSA)')
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
    """Create visualization for a batch of protein samples."""
    sequences = batch['sequences']
    rsas = batch['rsas']
    secondary_structures = batch['secondary_structures']
    phys_props = batch['phys_props']
    chains = batch['chains']
    labels = batch['labels']
    
    fig, axs = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    for i in range(num_samples):
        seq = sequences[i].numpy()
        rsa = rsas[i].numpy()
        ss = secondary_structures[i].numpy()
        chain = chains[i].numpy()

        # Sequence visualization (one-hot encoded) - adjusted to match actual amino acid count in model
        axs[i, 0].imshow(np.eye(22)[seq], aspect='auto', cmap='viridis')
        axs[i, 0].set_title(f'Sample {i+1} - Sequence')
        axs[i, 0].set_ylabel('AA Index')
        axs[i, 0].set_xlabel('Position')

        # RSA visualization
        axs[i, 1].plot(rsa)  # No need to repeat since RSA is per-residue now
        axs[i, 1].set_title(f'Sample {i+1} - RSA')
        axs[i, 1].set_ylabel('RSA')
        axs[i, 1].set_xlabel('Position')

        # Secondary structure visualization
        axs[i, 2].imshow(np.eye(4)[ss], aspect='auto', cmap='viridis')
        axs[i, 2].set_title(f'Sample {i+1} - Secondary Structure')
        axs[i, 2].set_ylabel('SS Index')
        axs[i, 2].set_xlabel('Position')

        # Chain visualization
        axs[i, 3].plot(chain)  # No need to repeat since chain is per-residue now
        axs[i, 3].set_title(f'Sample {i+1} - Chain')
        axs[i, 3].set_ylabel('Chain ID')
        axs[i, 3].set_xlabel('Position')

    plt.tight_layout()
    plt.savefig('plots/batch_visualization.png')
    plt.close()

    # Log batch shapes (removed logging since logger is not defined)
    # print statements used as alternative
    print(
        f"Batch shapes - Sequences: {sequences.shape}, RSAs: {rsas.shape}, "
        f"Secondary Structures: {secondary_structures.shape}, "
        f"Physicochemical Properties: {phys_props.shape}, "
        f"Chains: {chains.shape}, Labels: {labels.shape}"
    )
    print(f"First {num_samples} label values: {labels[:num_samples]}")
