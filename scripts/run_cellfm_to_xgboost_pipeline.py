#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete pipeline: CellFM feature extraction → XGBoost → QWK evaluation
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
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from collections import defaultdict
from tqdm import tqdm
import xgboost as xgb

# Add cellfm to path
sys.path.insert(0, '/home/ubuntu/LLM-inference/xinze-project/cellfm')
from model import Finetune_Cell_FM
from layers.utils import SCrna, Prepare, build_dataset, Config_80M

class EmbeddingExtractor:
    """Extract embeddings from CellFM model."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.embeddings = []
        self.labels = []
        self.donors = []
        self.celltypes = []
    
    def extract_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            _, _, cls_token = self.model(
                raw_nzdata=batch['raw_nzdata'],
                dw_nzdata=batch['dw_nzdata'],
                ST_feat=batch['ST_feat'],
                nonz_gene=batch['nonz_gene'],
                mask_gene=batch['mask_gene'],
                zero_idx=batch['zero_idx']
            )
            
            self.embeddings.append(cls_token.cpu().numpy())
            self.labels.extend(batch['label'].cpu().numpy())
            self.donors.extend(batch.get('donor', [f'donor_{i}' for i in range(len(batch['label']))]))
            self.celltypes.extend(batch.get('celltype', ['Unknown'] * len(batch['label'])))
    
    def get_results(self):
        embeddings = np.vstack(self.embeddings)
        labels = np.array(self.labels)
        return embeddings, labels, self.donors, self.celltypes

def load_cellfm_model(model_path, device='cuda:0'):
    """Load trained CellFM model."""
    print(f"Loading model from {model_path}...")
    
    # Read data to get gene count
    test_h5ad = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/test_subset.h5ad'
    adata = sc.read_h5ad(test_h5ad)
    n_genes = adata.n_vars
    
    cfg = Config_80M()
    cfg.num_cls = 4
    cfg.ckpt_path = None
    cfg.device = device
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    
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
    model.eval()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())}")
    return model

def extract_cell_embeddings(model, h5ad_path, device='cuda:0', batch_size=32):
    """Extract cell-level embeddings from h5ad using CellFM."""
    print(f"\nExtracting embeddings from {h5ad_path}...")
    
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
    
    # Create dataset and loader
    dataset = SCrna(adata, mode='eval')
    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=0.5)
    loader = build_dataset(dataset, prep=prep, batch_size=batch_size, pad_zero=True, drop=False, shuffle=False)
    
    # Extract embeddings
    extractor = EmbeddingExtractor(model, device)
    for batch in tqdm(loader, desc="Extracting"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        extractor.extract_batch(batch)
    
    embeddings, labels, donors, celltypes = extractor.get_results()
    
    print(f"Extracted {len(embeddings)} cell embeddings")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return embeddings, labels, donors, celltypes, adata

def aggregate_donor_embeddings(embeddings, labels, donors, celltypes, adata):
    """Aggregate cell embeddings to donor level by cell type."""
    print("\nAggregating cell embeddings to donor level...")
    
    donor_emb_dict = defaultdict(list)
    donor_label_dict = {}
    
    # Group by donor
    for i, donor in enumerate(donors):
        donor_emb_dict[donor].append(embeddings[i])
        
        # Store label (should be same for all cells from same donor)
        if donor not in donor_label_dict:
            donor_label_dict[donor] = labels[i]
    
    # Aggregate by donor and cell type
    final_donor_emb = {}
    final_donor_labels = {}
    
    for donor, emb_list in donor_emb_dict.items():
        # Simple mean pooling
        emb_mean = np.mean(emb_list, axis=0)
        final_donor_emb[donor] = emb_mean
        final_donor_labels[donor] = donor_label_dict[donor]
    
    print(f"Aggregated {len(final_donor_emb)} donor embeddings")
    
    return final_donor_emb, final_donor_labels

def train_xgboost_with_qwk(train_emb, train_labels, val_emb, val_labels):
    """Train XGBoost and evaluate with QWK."""
    print("\nTraining XGBoost...")
    
    # Convert to arrays
    train_emb_array = np.array([train_emb[k] for k in sorted(train_emb.keys())])
    train_labels_array = np.array([train_labels[k] for k in sorted(train_labels.keys())])
    val_emb_array = np.array([val_emb[k] for k in sorted(val_emb.keys())])
    val_labels_array = np.array([val_labels[k] for k in sorted(val_labels.keys())])
    
    # XGBoost training
    dtrain = xgb.DMatrix(train_emb_array, label=train_labels_array)
    dval = xgb.DMatrix(val_emb_array, label=val_labels_array)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'mlogloss',
    }
    
    def qwk_eval(y_pred, y_true):
        y_true = y_true.get_label()
        y_pred_class = np.argmax(y_pred.reshape(-1, 4), axis=1)
        qwk = cohen_kappa_score(y_true, y_pred_class, weights='quadratic')
        return 'qwk', qwk
    
    model = xgb.train(
        params,
        dtrain,
        evals=[(dtrain, 'train'), (dval, 'val')],
        feval=qwk_eval,
        maximize=True,
        early_stopping_rounds=20,
        num_boost_round=200,
        verbose_eval=10
    )
    
    # Evaluate
    val_pred = model.predict(dval)
    val_pred_class = np.argmax(val_pred, axis=1)
    
    val_acc = accuracy_score(val_labels_array, val_pred_class)
    val_f1 = f1_score(val_labels_array, val_pred_class, average='weighted')
    val_qwk = cohen_kappa_score(val_labels_array, val_pred_class, weights='quadratic')
    
    print(f"\nXGBoost Results:")
    print(f"Test Accuracy: {val_acc:.4f}")
    print(f"Test F1 Score: {val_f1:.4f}")
    print(f"Test QWK: {val_qwk:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_labels_array, val_pred_class, 
                                target_names=['Not AD', 'Low', 'Intermediate', 'High']))
    
    return val_qwk, val_acc, val_f1

def main():
    # Paths
    model_path = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/best_model.pth'
    train_h5ad = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/train_subset.h5ad'
    test_h5ad = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/test_subset.h5ad'
    split_json = '/home/ubuntu/LLM-inference/xinze-project/outputs/real_data_pipeline_mtg/donor_split.json'
    output_dir = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/xgboost_results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda:0'
    
    # Load model
    model = load_cellfm_model(model_path, device)
    
    # Extract embeddings
    train_emb, train_labels, train_donors, train_celltypes, train_adata = extract_cell_embeddings(
        model, train_h5ad, device, batch_size=32
    )
    
    test_emb, test_labels, test_donors, test_celltypes, test_adata = extract_cell_embeddings(
        model, test_h5ad, device, batch_size=32
    )
    
    # Load donor split
    with open(split_json, 'r') as f:
        split_data = json.load(f)
    train_donor_list = split_data['train_donors']
    test_donor_list = split_data['test_donors']
    
    # Aggregate embeddings
    all_donor_emb, all_donor_labels = aggregate_donor_embeddings(
        np.vstack([train_emb, test_emb]),
        np.concatenate([train_labels, test_labels]),
        train_donors + test_donors,
        train_celltypes + test_celltypes,
        None
    )
    
    # Split by donor
    train_donor_emb = {k: all_donor_emb[k] for k in train_donor_list if k in all_donor_emb}
    train_donor_labels = {k: all_donor_labels[k] for k in train_donor_list if k in all_donor_labels}
    
    test_donor_emb = {k: all_donor_emb[k] for k in test_donor_list if k in all_donor_emb}
    test_donor_labels = {k: all_donor_labels[k] for k in test_donor_list if k in all_donor_labels}
    
    print(f"\nTrain donors: {len(train_donor_emb)}")
    print(f"Test donors: {len(test_donor_emb)}")
    
    # Train XGBoost and evaluate
    qwk, acc, f1 = train_xgboost_with_qwk(
        train_donor_emb, train_donor_labels,
        test_donor_emb, test_donor_labels
    )
    
    # Save results
    results = {
        'test_qwk': float(qwk),
        'test_accuracy': float(acc),
        'test_f1': float(f1)
    }
    
    with open(os.path.join(output_dir, 'xgboost_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/xgboost_results.json")
    print(f"Final Test QWK: {qwk:.4f}")

if __name__ == '__main__':
    # Change to cellfm directory
    original_dir = os.getcwd()
    os.chdir('/home/ubuntu/LLM-inference/xinze-project/cellfm')
    
    try:
        main()
    finally:
        os.chdir(original_dir)



