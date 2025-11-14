#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data splitting script for large single-cell datasets.
Creates train/test splits at both cell and donor levels.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from collections import Counter

def create_cell_level_split(h5ad_path: str, output_path: str, test_size: float = 0.2, 
                          adnc_col: str = "ADNC", random_state: int = 42):
    """
    Create cell-level train/test split for large datasets.
    
    Args:
        h5ad_path: Path to h5ad file
        output_path: Path to save split JSON
        test_size: Fraction of cells for testing
        adnc_col: Column name for ADNC labels
        random_state: Random seed
    """
    print(f"Loading data from {h5ad_path}")
    
    # Read h5ad file
    adata = sc.read_h5ad(h5ad_path, backed='r')
    obs = adata.obs.copy()
    
    # Get ADNC labels
    if adnc_col not in obs.columns:
        raise KeyError(f"Column '{adnc_col}' not found in obs. Available columns: {list(obs.columns)}")
    
    adnc_labels = obs[adnc_col].astype(str).values
    
    # Get valid indices (non-null ADNC labels)
    valid_mask = ~pd.isna(adnc_labels) & (adnc_labels != 'nan') & (adnc_labels != '')
    valid_indices = np.where(valid_mask)[0]
    
    print(f"Total cells: {len(obs)}")
    print(f"Valid cells with ADNC labels: {len(valid_indices)}")
    
    # Show ADNC distribution
    adnc_counts = Counter(adnc_labels[valid_indices])
    print("ADNC distribution:")
    for label, count in adnc_counts.items():
        print(f"  {label}: {count}")
    
    # Create stratified split
    train_idx, test_idx = train_test_split(
        valid_indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=adnc_labels[valid_indices]
    )
    
    print(f"Train cells: {len(train_idx)}")
    print(f"Test cells: {len(test_idx)}")
    
    # Save split
    split_data = {
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "adnc_col": adnc_col,
        "total_cells": len(obs),
        "valid_cells": len(valid_indices),
        "train_cells": len(train_idx),
        "test_cells": len(test_idx),
        "test_size": test_size,
        "random_state": random_state
    }
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"Split saved to {output_path}")
    
    # Close file
    try:
        adata.file.close()
    except:
        pass

def create_donor_level_split(h5ad_path: str, output_path: str, test_size: float = 0.2,
                           donor_col: str = "Donor ID", adnc_col: str = "ADNC", 
                           random_state: int = 42):
    """
    Create donor-level train/test split.
    
    Args:
        h5ad_path: Path to h5ad file
        output_path: Path to save split JSON
        test_size: Fraction of donors for testing
        donor_col: Column name for donor IDs
        adnc_col: Column name for ADNC labels
        random_state: Random seed
    """
    print(f"Loading data from {h5ad_path}")
    
    # Read h5ad file
    adata = sc.read_h5ad(h5ad_path, backed='r')
    obs = adata.obs.copy()
    
    # Get donor and ADNC information
    if donor_col not in obs.columns:
        raise KeyError(f"Column '{donor_col}' not found in obs. Available columns: {list(obs.columns)}")
    if adnc_col not in obs.columns:
        raise KeyError(f"Column '{adnc_col}' not found in obs. Available columns: {list(obs.columns)}")
    
    donors = obs[donor_col].astype(str).values
    adnc_labels = obs[adnc_col].astype(str).values
    
    # Get unique donors with valid ADNC labels
    donor_adnc_pairs = []
    for i, (donor, adnc) in enumerate(zip(donors, adnc_labels)):
        if not pd.isna(adnc) and adnc != 'nan' and adnc != '':
            donor_adnc_pairs.append((donor, adnc))
    
    # Create donor-level dataframe
    donor_df = pd.DataFrame(donor_adnc_pairs, columns=['donor', 'adnc'])
    donor_df = donor_df.drop_duplicates()
    
    print(f"Total unique donor-ADNC pairs: {len(donor_df)}")
    
    # Show ADNC distribution at donor level
    donor_adnc_counts = donor_df['adnc'].value_counts()
    print("Donor-level ADNC distribution:")
    for label, count in donor_adnc_counts.items():
        print(f"  {label}: {count}")
    
    # Handle small datasets (like test case with 4 donors)
    unique_donors = donor_df['donor'].unique()
    n_donors = len(unique_donors)
    
    if n_donors <= 4:
        print(f"Small dataset detected ({n_donors} donors) - using half-half split")
        # For small datasets, use half-half split
        np.random.seed(random_state)
        shuffled_donors = np.random.permutation(unique_donors)
        
        n_test = n_donors // 2
        n_train = n_donors - n_test
        
        train_donors = shuffled_donors[:n_train]
        test_donors = shuffled_donors[n_train:]
        
        print(f"Half-half split: {n_train} train, {n_test} test donors")
    else:
        # For larger datasets, use stratified split
        # Ensure all classes are represented in training set
        adnc_per_donor = donor_df.set_index('donor')['adnc']
        
        # Check if all classes are present
        unique_adnc = adnc_per_donor.unique()
        print(f"Available ADNC classes: {unique_adnc}")
        
        # Use stratified split to ensure all classes in training
        try:
            train_donors, test_donors = train_test_split(
                unique_donors,
                test_size=test_size,
                random_state=random_state,
                stratify=adnc_per_donor
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            print("Falling back to random split...")
            train_donors, test_donors = train_test_split(
                unique_donors,
                test_size=test_size,
                random_state=random_state
            )
        
        # Verify all classes are in training set
        train_adnc = adnc_per_donor[train_donors].unique()
        test_adnc = adnc_per_donor[test_donors].unique()
        
        print(f"Training set ADNC classes: {train_adnc}")
        print(f"Test set ADNC classes: {test_adnc}")
        
        # If any class is missing from training, move one donor from test to train
        missing_classes = set(unique_adnc) - set(train_adnc)
        if missing_classes:
            print(f"Warning: Missing classes in training set: {missing_classes}")
            for missing_class in missing_classes:
                # Find a donor with this class in test set
                test_donors_with_class = [d for d in test_donors if adnc_per_donor[d] == missing_class]
                if test_donors_with_class:
                    # Move one donor from test to train
                    donor_to_move = test_donors_with_class[0]
                    train_donors = np.append(train_donors, donor_to_move)
                    test_donors = np.delete(test_donors, np.where(test_donors == donor_to_move)[0])
                    print(f"Moved donor {donor_to_move} (class {missing_class}) from test to train")
        
        # Final verification for both train and test sets
        final_train_adnc = adnc_per_donor[train_donors].unique()
        final_test_adnc = adnc_per_donor[test_donors].unique()
        print(f"Final training set ADNC classes: {final_train_adnc}")
        print(f"Final test set ADNC classes: {final_test_adnc}")
        print(f"All classes in training: {set(unique_adnc).issubset(set(final_train_adnc))}")
        print(f"All classes in test: {set(unique_adnc).issubset(set(final_test_adnc))}")
        
        # If test set is missing classes, try to balance by moving donors
        missing_test_classes = set(unique_adnc) - set(final_test_adnc)
        if missing_test_classes:
            print(f"Warning: Missing classes in test set: {missing_test_classes}")
            # Try to move one donor of each missing class from train to test
            for missing_class in missing_test_classes:
                train_donors_with_class = [d for d in train_donors if adnc_per_donor[d] == missing_class]
                if len(train_donors_with_class) > 1:  # Only move if we have more than one donor of this class in train
                    donor_to_move = train_donors_with_class[0]
                    test_donors = np.append(test_donors, donor_to_move)
                    train_donors = np.delete(train_donors, np.where(train_donors == donor_to_move)[0])
                    print(f"Moved donor {donor_to_move} (class {missing_class}) from train to test")
        
        # Final verification after balancing
        final_train_adnc = adnc_per_donor[train_donors].unique()
        final_test_adnc = adnc_per_donor[test_donors].unique()
        print(f"Balanced training set ADNC classes: {final_train_adnc}")
        print(f"Balanced test set ADNC classes: {final_test_adnc}")
        print(f"All classes in training: {set(unique_adnc).issubset(set(final_train_adnc))}")
        print(f"All classes in test: {set(unique_adnc).issubset(set(final_test_adnc))}")
    
    print(f"Train donors: {len(train_donors)}")
    print(f"Test donors: {len(test_donors)}")
    
    # Print donor IDs with their associated labels for verification
    print("Train donors with their ADNC labels:")
    for donor in sorted(train_donors.tolist()):
        donor_label = adnc_per_donor[donor]
        print(f"  {donor}: {donor_label}")
    
    print("Test donors with their ADNC labels:")
    for donor in sorted(test_donors.tolist()):
        donor_label = adnc_per_donor[donor]
        print(f"  {donor}: {donor_label}")
    
    # Check for overlap between train and test donors
    train_set = set(train_donors)
    test_set = set(test_donors)
    overlap = train_set.intersection(test_set)
    if overlap:
        print(f"❌ ERROR: Overlap found between train and test donors: {sorted(overlap)}")
        raise ValueError(f"Train and test donors have overlap: {sorted(overlap)}")
    else:
        print("✅ No overlap between train and test donors")
    
    # Get cell indices for each split
    train_mask = np.isin(donors, train_donors)
    test_mask = np.isin(donors, test_donors)
    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    print(f"Train cells: {len(train_idx)}")
    print(f"Test cells: {len(test_idx)}")
    
    # Save split
    split_data = {
        "train_donors": train_donors.tolist(),
        "test_donors": test_donors.tolist(),
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "donor_col": donor_col,
        "adnc_col": adnc_col,
        "total_cells": len(obs),
        "train_cells": len(train_idx),
        "test_cells": len(test_idx),
        "test_size": test_size,
        "random_state": random_state
    }
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"Split saved to {output_path}")
    
    # Close file
    try:
        adata.file.close()
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description="Create data splits for large single-cell datasets")
    parser.add_argument("--h5ad_path", required=True, help="Path to h5ad file")
    parser.add_argument("--output_path", required=True, help="Path to save split JSON")
    parser.add_argument("--split_type", choices=["cell", "donor"], default="cell", 
                       help="Type of split: cell-level or donor-level")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for testing")
    parser.add_argument("--adnc_col", default="ADNC", help="ADNC column name")
    parser.add_argument("--donor_col", default="Donor ID", help="Donor column name")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.split_type == "cell":
        create_cell_level_split(
            args.h5ad_path, args.output_path, args.test_size, 
            args.adnc_col, args.random_state
        )
    else:
        create_donor_level_split(
            args.h5ad_path, args.output_path, args.test_size,
            args.donor_col, args.adnc_col, args.random_state
        )

if __name__ == "__main__":
    main()
