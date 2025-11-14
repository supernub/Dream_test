#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Extract cell-level embeddings from CellFM and aggregate to donor level.
This script extracts embeddings from the trained CellFM model and saves them
in a format compatible with the existing donor_classifier.py pipeline.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

# Add cellfm to path
sys.path.insert(0, '/home/ubuntu/LLM-inference/xinze-project/cellfm')
from model import Finetune_Cell_FM
from layers.utils import SCrna, Prepare, build_dataset, Config_80M

def load_cellfm_model(model_path, h5ad_path, device='cuda:0'):
    """Load trained CellFM model."""
    print(f"Loading model from {model_path}...")
    
    # Read data to get gene count
    adata = sc.read_h5ad(h5ad_path)
    n_genes = adata.n_vars
    
    cfg = Config_80M()
    cfg.num_cls = 4
    cfg.ckpt_path = None
    cfg.device = device
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    
    # Change to cellfm directory for loading gene info
    original_dir = os.getcwd()
    os.chdir('/home/ubuntu/LLM-inference/xinze-project/cellfm')
    
    try:
        model = Finetune_Cell_FM(cfg).to(device)
        
        # Re-initialize gene embedding
        with torch.no_grad():
            emb_dim = model.extractor.net.gene_emb.shape[1]
            pad_to = ((n_genes - 1) // 8 + 1) * 8
            model.extractor.net.gene_emb = nn.Parameter(torch.empty(pad_to, emb_dim))
            nn.init.xavier_normal_(model.extractor.net.gene_emb)
            model.extractor.net.gene_emb.data[0, :] = 0
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()
        # Double check all parameters are on the correct device
        for name, param in model.named_parameters():
            if param.device != torch.device(device):
                print(f"Warning: Parameter {name} is on {param.device}, moving to {device}")
                param.data = param.data.to(device)
    finally:
        os.chdir(original_dir)
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())}")
    return model

def extract_and_aggregate_embeddings(model, h5ad_path, split_json, device='cuda:0', batch_size=32):
    """Extract cell-level embeddings and aggregate to donor level."""
    print(f"\nExtracting embeddings from {h5ad_path}...")
    
    original_dir = os.getcwd()
    os.chdir('/home/ubuntu/LLM-inference/xinze-project/cellfm')
    
    try:
        # Load data
        adata = sc.read_h5ad(h5ad_path)
        print(f"Data shape: {adata.shape}")
        
        # Ensure required columns
        if 'ADNC' in adata.obs.columns:
            adnc_series = adata.obs['ADNC']
            if pd.api.types.is_numeric_dtype(adnc_series):
                num_to_name = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}
                adnc_norm = adnc_series.map(num_to_name).astype(str)
            else:
                s = adnc_series.astype(str)
                map_num = {'0': 'Not AD', '1': 'Low', '2': 'Intermediate', '3': 'High'}
                adnc_norm = s.replace(map_num)
            ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
            adata.obs['feat'] = [ADNC_MAP.get(lbl, 0) for lbl in adnc_norm.values]
        
        if 'celltype' not in adata.obs.columns:
            if 'Supertype' in adata.obs.columns:
                adata.obs['celltype'] = adata.obs['Supertype']
            else:
                adata.obs['celltype'] = 'Unknown'
        
        if 'train' not in adata.obs.columns:
            adata.obs['train'] = 2  # test mode
        
        # Get donor column
        if 'Donor_space_ID' in adata.obs.columns:
            adata.obs['donor_id'] = adata.obs['Donor_space_ID']
        elif 'Donor ID' in adata.obs.columns:
            adata.obs['donor_id'] = adata.obs['Donor ID']
        else:
            raise KeyError("No Donor ID column found")
        
        # Add required str_batch column for SCrna
        if 'str_batch' not in adata.obs.columns:
            adata.obs['str_batch'] = adata.obs['donor_id'].astype(str)
        
        # Create dataset and loader
        cfg = Config_80M()
        dataset = SCrna(adata, mode='eval')
        prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=0.5)
        loader = build_dataset(dataset, prep=prep, batch_size=batch_size, pad_zero=True, drop=False, shuffle=False)
        
        # Extract embeddings
        model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting"):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                
                _, _, cls_token = model(
                    raw_nzdata=batch['raw_nzdata'],
                    dw_nzdata=batch['dw_nzdata'],
                    ST_feat=batch['ST_feat'],
                    nonz_gene=batch['nonz_gene'],
                    mask_gene=batch['mask_gene'],
                    zero_idx=batch['zero_idx']
                )
                
                all_embeddings.append(cls_token.cpu().numpy())
                all_labels.append(batch['feat'].cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
        print(f"Extracted {len(embeddings)} cell embeddings")
        print(f"Embedding dim: {embeddings.shape[1]}")
        print(f"Label distribution: {np.bincount(labels.astype(int))}")
        
        # Now we need to get donor and celltype from adata
        # Since the loader doesn't preserve this, we need to track it
        # Let's get the indices from the dataset
        donor_ids = []
        celltype_ids = []
        feat_labels = []
        
        for idx in range(len(adata)):
            # Check if this cell is in the filtered dataset
            if adata.obs.iloc[idx]['train'] == 2:
                donor_ids.append(str(adata.obs.iloc[idx]['donor_id']))
                celltype_ids.append(str(adata.obs.iloc[idx].get('celltype', 'Unknown')))
                feat_labels.append(adata.obs.iloc[idx].get('feat', 0))
        
        # Trim to match extracted count
        max_len = len(embeddings)
        donor_ids = donor_ids[:max_len]
        celltype_ids = celltype_ids[:max_len]
        
        # Aggregate by donor and celltype
        print("\nAggregating to donor level...")
        
        # Group by donor
        donor_groups = defaultdict(lambda: defaultdict(list))
        
        for i, (emb, label, donor, celltype) in enumerate(zip(embeddings, feat_labels, donor_ids, celltype_ids)):
            donor_groups[donor][celltype].append({
                'embedding': emb,
                'label': label
            })
        
        print(f"Found {len(donor_groups)} unique donors")
        
        # Create donor embeddings by averaging over celltypes
        donor_emb_dict = {}
        donor_label_dict = {}
        
        for donor, celltype_data in donor_groups.items():
            # Get all cell type embeddings for this donor
            all_cell_embeddings = []
            for celltype, cells in celltype_data.items():
                for cell in cells:
                    all_cell_embeddings.append(cell['embedding'])
            
            # Simple mean pooling
            donor_emb = np.mean(all_cell_embeddings, axis=0)
            donor_emb_dict[donor] = donor_emb
            
            # Get label (should be same for all cells from same donor)
            donor_label_dict[donor] = int(label)  # Use the last cell's label
        
        print(f"Aggregated to {len(donor_emb_dict)} donor embeddings")
        
    finally:
        os.chdir(original_dir)
    
    return donor_emb_dict, donor_label_dict, len(donor_groups)

def main():
    # Paths
    model_path = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/best_model.pth'
    train_h5ad = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/train_subset.h5ad'
    test_h5ad = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/test_subset.h5ad'
    split_json = '/home/ubuntu/LLM-inference/xinze-project/outputs/real_data_pipeline_mtg/donor_split.json'
    output_dir = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/embeddings'
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda:0'
    
    # Load model (using test h5ad for gene count)
    model = load_cellfm_model(model_path, test_h5ad, device)
    
    # Extract embeddings from train data
    train_donor_emb, train_donor_labels, train_num_donors = extract_and_aggregate_embeddings(
        model, train_h5ad, split_json, device, batch_size=32
    )
    
    # Extract embeddings from test data
    test_donor_emb, test_donor_labels, test_num_donors = extract_and_aggregate_embeddings(
        model, test_h5ad, split_json, device, batch_size=32
    )
    
    # Combine all donor embeddings
    all_donor_emb = {}
    all_donor_labels = {}
    all_donor_emb.update(train_donor_emb)
    all_donor_emb.update(test_donor_emb)
    all_donor_labels.update(train_donor_labels)
    all_donor_labels.update(test_donor_labels)
    
    print(f"\nTotal donors: {len(all_donor_emb)}")
    print(f"Train donors: {train_num_donors}")
    print(f"Test donors: {test_num_donors}")
    
    # Save embeddings in the format expected by donor_classifier.py
    np.save(os.path.join(output_dir, 'donor_embeddings.npy'), all_donor_emb)
    np.save(os.path.join(output_dir, 'donor_predictions.npy'), all_donor_labels)
    np.save(os.path.join(output_dir, 'donor_labels.npy'), all_donor_labels)
    
    # Save metadata
    metadata = {
        'donor_split_info': {
            'train_donors': list(train_donor_emb.keys()),
            'test_donors': list(test_donor_emb.keys())
        },
        'num_repetitions': 1,
        'embedding_dim': list(all_donor_emb.values())[0].shape[0],
        'num_donors': len(all_donor_emb),
        'model_path': model_path
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nEmbeddings saved to {output_dir}")
    print(f"Metadata saved to {output_dir}/metadata.json")

if __name__ == '__main__':
    main()

