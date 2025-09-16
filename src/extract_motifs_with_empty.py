#!/usr/bin/env python3
"""
Script to extract specific motif information from MAST XML files.
This script focuses on extracting the motif sequences with their start and end positions
for each protein sequence, with an option to filter for self-matches only.
It also includes sequences with no motifs found, marking them as "empty".
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import re

def extract_motif_info(xml_file_path):
    """
    Extract motif information from a single MAST XML file.
    
    Args:
        xml_file_path (str): Path to the mast.xml file
        
    Returns:
        list: List of dictionaries containing motif information
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        return []
    except FileNotFoundError:
        print(f"XML file not found: {xml_file_path}")
        return []
    
    motifs_data = []
    
    # Extract the motif database name from the folder name
    folder_name = Path(xml_file_path).parent.name
    motif_protein_id = re.match(r'^([A-Za-z0-9]+)_', folder_name).group(1) if re.match(r'^([A-Za-z0-9]+)_', folder_name) else folder_name
    
    # Get motif sequences (the actual motif patterns)
    motifs = root.findall('.//motif')
    motif_patterns = {}
    for i, motif in enumerate(motifs):
        motif_id = motif.get('id')  # This is the actual motif sequence pattern
        motif_patterns[i] = motif_id
    
    # Find all sequences
    sequences = root.findall('.//sequence')
    
    for sequence in sequences:
        seq_name = sequence.get('name')
        if not seq_name:
            continue
            
        # Extract the protein ID from the sequence name
        # Handle different formats:
        # 1. Direct format: "1AGR_D8.4e-143" -> "1AGR"
        # 2. SwissProt format: "sp|P10824|GNAI1_RAT1.1e-142" -> "GNAI1"
        if re.match(r'^([A-Za-z0-9]+)_[A-Za-z]', seq_name):
            sequence_protein_id = re.match(r'^([A-Za-z0-9]+)_[A-Za-z]', seq_name).group(1)
        elif re.match(r'^sp\\|[A-Za-z0-9]+\\|([A-Za-z0-9]+)_[A-Za-z]+', seq_name):
            sequence_protein_id = re.match(r'^sp\\|[A-Za-z0-9]+\\|([A-Za-z0-9]+)_[A-Za-z]+', seq_name).group(1)
        elif re.match(r'^tr\\|[A-Za-z0-9]+\\|([A-Za-z0-9]+)_[A-Za-z]+', seq_name):
            sequence_protein_id = re.match(r'^tr\\|[A-Za-z0-9]+\\|([A-Za-z0-9]+)_[A-Za-z]+', seq_name).group(1)
        else:
            sequence_protein_id = seq_name
        
        # Find all motif hits in this sequence
        hits = sequence.findall('.//hit')
        
        if hits:
            # Process motif hits
            for hit in hits:
                motif_idx = int(hit.get('idx', 0))
                pvalue = hit.get('pvalue')
                position = int(hit.get('pos', 0))
                
                # Get the actual motif sequence pattern
                motif_seq = motif_patterns.get(motif_idx, f"Motif_{motif_idx}")
                motif_length = len(motif_seq)
                end_pos = position + motif_length - 1
                
                motifs_data.append({
                    'motif_based_on': motif_protein_id,  # The protein this motif is based on
                    'sequence_name': seq_name,
                    'sequence_protein_id': sequence_protein_id,  # Extracted protein ID
                    'motif_index': motif_idx,
                    'motif_sequence': motif_seq,  # The actual amino acid sequence pattern
                    'start_position': position,
                    'end_position': end_pos,
                    'motif_length': motif_length,
                    'pvalue': pvalue,
                    'motif_found': 'yes'
                })
        else:
            # No motifs found for this sequence
            motifs_data.append({
                'motif_based_on': motif_protein_id,
                'sequence_name': seq_name,
                'sequence_protein_id': sequence_protein_id,
                'motif_index': '',
                'motif_sequence': 'empty',
                'start_position': '',
                'end_position': '',
                'motif_length': '',
                'pvalue': '',
                'motif_found': 'no'
            })
    
    return motifs_data

def process_all_motif_files(motif_base_path):
    """
    Process all motif XML files in the given base path.
    
    Args:
        motif_base_path (str): Path to the motif directory (data/motif_1 or data/motif_2)
        
    Returns:
        list: Combined list of all motif data
    """
    all_motifs_data = []
    
    if not os.path.exists(motif_base_path):
        print(f"Base path {motif_base_path} does not exist.")
        return all_motifs_data
    
    # Iterate through all subdirectories
    for dir_name in sorted(os.listdir(motif_base_path)):
        dir_path = os.path.join(motif_base_path, dir_name)
        if os.path.isdir(dir_path):
            # Look for mast.xml file
            xml_file_path = os.path.join(dir_path, 'mast.xml')
            if os.path.exists(xml_file_path):
                print(f"Processing {xml_file_path}")
                motifs_data = extract_motif_info(xml_file_path)
                all_motifs_data.extend(motifs_data)
    
    return all_motifs_data

def filter_self_matches(data):
    """
    Filter data to only include entries where motif_based_on matches sequence_protein_id.
    
    Args:
        data (list): List of motif data dictionaries
        
    Returns:
        list: Filtered list containing only self-matches
    """
    return [entry for entry in data if entry['motif_based_on'] == entry['sequence_protein_id']]

def save_to_csv(data, output_file):
    """
    Save motif data to a CSV file.
    
    Args:
        data (list): List of motif data dictionaries
        output_file (str): Path to output CSV file
    """
    if not data:
        print("No data to save.")
        return
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(data)} entries to {output_file}")

def main():
    """
    Main function to process motif directories and extract information.
    """
    # Define paths
    project_root = Path(__file__).parent.parent
    motif_1_path = project_root / 'data' / 'motif_1'
    motif_2_path = project_root / 'data' / 'motif_2'
    
    output_dir = project_root / 'data' / 'parsed_motifs'
    output_dir.mkdir(exist_ok=True)
    
    # Process both motif directories
    print("Processing motif_1 directory...")
    motif_1_data = process_all_motif_files(str(motif_1_path))
    
    print("\nProcessing motif_2 directory...")
    motif_2_data = process_all_motif_files(str(motif_2_path))
    
    # Combine datasets
    all_data = motif_1_data + motif_2_data
    
    if all_data:
        # Filter for self-matches only
        self_matches = filter_self_matches(all_data)
        save_to_csv(self_matches, output_dir / 'self_match_motif_data_with_empty.csv')
        
        # Show summary
        df_all = pd.DataFrame(all_data)
        df_self = pd.DataFrame(self_matches)
        
        total_with_motifs = len(df_all[df_all['motif_found'] == 'yes'])
        total_without_motifs = len(df_all[df_all['motif_found'] == 'no'])
        
        self_with_motifs = len(df_self[df_self['motif_found'] == 'yes'])
        self_without_motifs = len(df_self[df_self['motif_found'] == 'no'])
        
        print(f"\nSummary:")
        print(f"  Total sequences processed: {len(all_data)}")
        print(f"  Total with motifs: {total_with_motifs}")
        print(f"  Total without motifs (empty): {total_without_motifs}")
        print(f"  Self-match with motifs: {self_with_motifs}")
        print(f"  Self-match without motifs (empty): {self_without_motifs}")
        print(f"  Motif 1 occurrences: {len(motif_1_data)}")
        print(f"  Motif 2 occurrences: {len(motif_2_data)}")
        
        # Show some examples of self-matches
        if not df_self.empty:
            print(f"\nSample self-match entries:")
            print(df_self[['motif_based_on', 'sequence_protein_id', 'motif_sequence', 
                          'start_position', 'end_position', 'motif_found']].head(10))
            
            # Show specific examples you mentioned
            print(f"\nSpecific self-match examples:")
            # Look for 1A0O self-matches
            a0o_entries = df_self[df_self['motif_based_on'] == '1A0O']
            if not a0o_entries.empty:
                print("1A0O self-matches:")
                print(a0o_entries[['motif_based_on', 'sequence_protein_id', 'motif_sequence', 
                                 'start_position', 'end_position', 'motif_found']].head())
            
            # Look for 1AGR self-matches
            agr_entries = df_self[df_self['motif_based_on'] == '1AGR']
            if not agr_entries.empty:
                print("\n1AGR self-matches:")
                print(agr_entries[['motif_based_on', 'sequence_protein_id', 'motif_sequence', 
                                 'start_position', 'end_position', 'motif_found']].head())
                
            # Look for 1B6C self-matches
            b6c_entries = df_self[df_self['motif_based_on'] == '1B6C']
            if not b6c_entries.empty:
                print("\n1B6C self-matches:")
                print(b6c_entries[['motif_based_on', 'sequence_protein_id', 'motif_sequence', 
                                 'start_position', 'end_position', 'motif_found']].head())
                
            # Show some empty examples
            empty_entries = df_self[df_self['motif_found'] == 'no']
            if not empty_entries.empty:
                print(f"\nSample empty (no motif found) entries:")
                print(empty_entries[['motif_based_on', 'sequence_protein_id', 'motif_sequence', 
                                   'motif_found']].head(5))
    else:
        print("No motif data found.")

if __name__ == "__main__":
    main()
