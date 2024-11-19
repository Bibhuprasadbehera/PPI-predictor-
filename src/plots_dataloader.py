import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_interaction_score_distribution_plot(data):
    plt.figure(figsize=(12, 6))
    sns.histplot(data['test_interaction_score'], kde=True)
    plt.title('Distribution of Interaction Scores')
    plt.xlabel('Interaction Score')
    plt.ylabel('Count')
    plt.savefig('interaction_score_distribution.png')
    plt.close()

def create_rsa_distribution_plot(data):
    plt.figure(figsize=(12, 6))
    sns.histplot(data['rsa'], kde=True)
    plt.title('Distribution of RSA Values')
    plt.xlabel('RSA')
    plt.ylabel('Count')
    plt.savefig('rsa_distribution.png')
    plt.close()

def create_secondary_structure_distribution_plot(data):
    plt.figure(figsize=(12, 6))
    ss_counts = data['three_hot_ss'].apply(lambda x: ''.join(set(x))).value_counts()
    sns.barplot(x=ss_counts.index, y=ss_counts.values)
    plt.title('Distribution of Secondary Structures')
    plt.xlabel('Secondary Structure')
    plt.ylabel('Count')
    plt.savefig('secondary_structure_distribution.png')
    plt.close()

def create_amino_acid_frequency_plot(data):
    plt.figure(figsize=(12, 6))
    amino_acid_counts = data['aa'].value_counts()
    sns.barplot(x=amino_acid_counts.index, y=amino_acid_counts.values)
    plt.title('Amino Acid Frequency')
    plt.xlabel('Amino Acid')
    plt.ylabel('Count')
    plt.savefig('amino_acid_frequency.png')
    plt.close()

def create_sequence_length_distribution_plot(data):
    plt.figure(figsize=(12, 6))
    sequence_lengths = data['aa'].str.len()
    sns.histplot(sequence_lengths, kde=True)
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig('sequence_length_distribution.png')
    plt.close()

def create_physicochemical_properties_distribution_plots(phys_prop_file):
    # Load the physicochemical properties data
    data = pd.read_csv(phys_prop_file)

    # Create distribution plots for each property
    for property in data.columns[1:]:  # Skip the first column (amino acid)
        plt.figure(figsize=(8, 6))
        sns.histplot(data[property], kde=True)
        plt.title(f'Distribution of {property}')
        plt.xlabel(property)
        plt.ylabel('Frequency')
        plt.savefig(f'physicochemical_properties_distribution/{property}_distribution.png')
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
    plt.savefig('batch_visualization.png')
    plt.close()
    print(f"Batch shape - Sequences: {sequences.shape}, RSAs: {rsas.shape}, Secondary Structures: {secondary_structures.shape}, Physicochemical Properties: {phys_props.shape}, Chains: {chains.shape}, Labels: {labels.shape}")  # Include chains
    print(f"Label values: {labels[:num_samples]}")
