#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellFM training with balanced sampling and weighted loss
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

# Add cellfm to path
sys.path.insert(0, '/home/ubuntu/LLM-inference/xinze-project/cellfm')
from model import Finetune_Cell_FM
from layers.utils import SCrna, Prepare, build_dataset, Config_80M

ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}


def normalize_adnc(series: pd.Series) -> pd.Series:
    """Normalize ADNC labels to consistent format."""
    if pd.api.types.is_numeric_dtype(series):
        num_to_name = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}
        return series.map(num_to_name).astype(str)
    s = series.astype(str)
    map_num = {'0': 'Not AD', '1': 'Low', '2': 'Intermediate', '3': 'High'}
    return s.replace(map_num)


def prepare_adata_for_cellfm(h5ad_path, split_json, train_split):
    """Prepare AnnData with proper fields for CellFM."""
    adata = sc.read_h5ad(h5ad_path)
    
    # Normalize ADNC labels
    if 'ADNC' in adata.obs.columns:
        adnc_series = adata.obs['ADNC']
        adnc_norm = normalize_adnc(adnc_series)
        adata.obs['feat'] = [ADNC_MAP.get(lbl, 0) for lbl in adnc_norm.values]
    
    # Ensure celltype
    if 'celltype' not in adata.obs.columns:
        if 'Supertype' in adata.obs.columns:
            adata.obs['celltype'] = adata.obs['Supertype']
        else:
            adata.obs['celltype'] = 'Unknown'
    
    # Load split
    with open(split_json, 'r') as f:
        split = json.load(f)
    
    if train_split:
        split_indices = split.get('train_indices', [])
        adata.obs['train'] = 0
    else:
        split_indices = split.get('test_indices', [])
        adata.obs['train'] = 2
    
    # Set split info
    adata.obs['batch_id'] = 0
    
    return adata


def create_balanced_sampler(dataset, batch_size):
    """Create a weighted sampler for balanced sampling."""
    # Get all labels from dataset
    labels = []
    for i in range(len(dataset)):
        _, _, _, _, _, label = dataset[i]
        labels.append(label)
    
    labels = np.array(labels)
    class_counts = np.bincount(labels.astype(int))
    
    # Calculate weights: inverse of class frequency
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]
    
    # Normalize
    sample_weights = sample_weights / sample_weights.sum()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    print("\nBalanced sampling setup:")
    print(f"  Class distribution: {dict(zip(['Not AD', 'Low', 'Intermediate', 'High'], class_counts))}")
    print(f"  Sample weights (inverse freq): {class_weights}")
    
    return sampler


def get_class_weights(labels):
    """Calculate class weights for weighted loss."""
    class_counts = np.bincount(labels.astype(int))
    total = len(labels)
    n_classes = len(class_counts)
    
    class_weights = total / (n_classes * class_counts.astype(float))
    class_weights = class_weights / class_weights.sum() * n_classes  # Normalize
    
    print(f"\nClass weights for loss: {dict(zip(['Not AD', 'Low', 'Intermediate', 'High'], class_weights))}")
    
    return torch.FloatTensor(class_weights)


def create_balanced_loader(dataset, prep, batch_size, pad_zero=True, device='cuda:0'):
    """Create a DataLoader with balanced sampling."""
    
    def collate_fn(samples):
        raw_nzdata_batch = []
        dw_nzdata_batch = []
        ST_feat_batch = []
        nonz_gene_batch = []
        mask_gene_batch = []
        zero_idx_batch = []
        celltype_label_batch = []
        batch_id_batch = []
        feat_batch = []

        for data, gene, T, celltype_label, batch_id, feat in samples:
            raw_data, nonz, zero = prep.seperate(data)
            data, nonz, cuted, z_sample, seq_len = prep.sample(raw_data, nonz, zero)
            raw_data, raw_nzdata, nonz = prep.compress(raw_data, nonz)
            gene, nonz_gene, _ = prep.compress(gene, nonz)
            raw_nzdata, dw_nzdata, S, T = prep.bayes(raw_nzdata, T)
            dw_nzdata, S = prep.normalize(dw_nzdata, S)
            raw_nzdata, T = prep.normalize(raw_nzdata, T)
            ST_feat = prep.cat_st(S, T)

            if pad_zero:
                zero_idx = prep.attn_mask(seq_len)
                dw_nzdata, mask_gene = prep.mask(dw_nzdata)
                raw_nzdata = prep.pad_zero(raw_nzdata)
                dw_nzdata = prep.pad_zero(dw_nzdata)
                nonz_gene = prep.pad_zero(nonz_gene)
                mask_gene = prep.pad_zero(mask_gene)
            else:
                dw_nzdata, zero_idx = prep.zero_idx(dw_nzdata)
                dw_nzdata, mask_gene = prep.mask(dw_nzdata)
                zero_pad, zero_mask = prep.zero_mask(seq_len)
                gene, z_gene, z_sample = prep.compress(gene, z_sample)
                nonz_gene = prep.pad_gene(nonz_gene, z_gene)
                raw_nzdata = prep.pad_zero(raw_nzdata)
                dw_nzdata = prep.pad_gene(dw_nzdata, zero_pad)
                mask_gene = prep.pad_gene(mask_gene, zero_mask)

            raw_nzdata_batch.append(torch.tensor(raw_nzdata, dtype=torch.float32))
            dw_nzdata_batch.append(torch.tensor(dw_nzdata, dtype=torch.float32))
            ST_feat_batch.append(torch.tensor(ST_feat, dtype=torch.float32))
            nonz_gene_batch.append(torch.tensor(nonz_gene, dtype=torch.int32))
            mask_gene_batch.append(torch.tensor(mask_gene, dtype=torch.float32))
            zero_idx_batch.append(torch.tensor(zero_idx, dtype=torch.float32))
            celltype_label_batch.append(torch.tensor(celltype_label, dtype=torch.long))
            batch_id_batch.append(torch.tensor(batch_id, dtype=torch.long))
            feat_batch.append(torch.tensor(feat, dtype=torch.float32))

        return {
            'raw_nzdata': torch.stack(raw_nzdata_batch),
            'dw_nzdata': torch.stack(dw_nzdata_batch),
            'ST_feat': torch.stack(ST_feat_batch),
            'nonz_gene': torch.stack(nonz_gene_batch),
            'mask_gene': torch.stack(mask_gene_batch),
            'zero_idx': torch.stack(zero_idx_batch),
            'celltype_label': torch.stack(celltype_label_batch),
            'batch_id': torch.stack(batch_id_batch),
            'feat': torch.stack(feat_batch),
        }
    
    sampler = create_balanced_sampler(dataset, batch_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_fn
    )


def train_balanced(h5ad_path, split_json, output_dir, ckpt_path, device='cuda:0',
                  batch_size=16, epochs=3, lr=1e-4, num_cls=4):
    """Train CellFM with balanced sampling and weighted loss."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CellFM Balanced Training for ADNC Prediction")
    print("="*80)
    
    # Prepare data
    print("\n1. Preparing data...")
    train_adata = prepare_adata_for_cellfm(h5ad_path, split_json, train_split=True)
    test_adata = prepare_adata_for_cellfm(h5ad_path, split_json, train_split=False)
    
    train_labels = train_adata.obs['feat'].values
    print(f"Train labels distribution: {np.bincount(train_labels.astype(int))}")
    
    # Create datasets
    train_dataset = SCrna(train_adata, mode="train")
    test_dataset = SCrna(test_adata, mode="test")
    
    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
    
    # Create balanced loader
    train_loader = create_balanced_loader(train_dataset, prep, batch_size, pad_zero=True, device=device)
    test_loader = build_dataset(test_dataset, prep=prep, batch_size=batch_size, 
                               pad_zero=True, drop=False, shuffle=False)
    
    # Calculate class weights
    class_weights = get_class_weights(train_labels)
    class_weights = class_weights.to(device)
    
    # Initialize model
    print("\n2. Initializing model...")
    net = Finetune_Cell_FM(cfg)
    net = net.to(device)
    
    # Setup optimizer and weighted loss
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler()
    
    print("\n3. Starting balanced training...")
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                cls, mask_loss, _ = net(**{k: batch[k] for k in batch if k in ['raw_nzdata', 'dw_nzdata', 'ST_feat', 'nonz_gene', 'mask_gene', 'zero_idx']})
                cls_loss = criterion(cls, batch['feat'].long())
                loss = mask_loss + cls_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            acc = (cls.argmax(1) == batch['feat']).float().mean().item()
            running_loss += loss.item()
            running_acc += acc
        
        # Evaluate
        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                cls, _, _ = net(**{k: batch[k] for k in batch if k in ['raw_nzdata', 'dw_nzdata', 'ST_feat', 'nonz_gene', 'mask_gene', 'zero_idx']})
                all_preds.extend(cls.argmax(1).cpu().numpy())
                all_labels.extend(batch['feat'].cpu().numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        test_qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        print(f"Epoch {epoch+1}: Test Acc={test_acc:.4f}, QWK={test_qwk:.4f}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), f"{output_dir}/best_model_balanced.pth")
    
    print(f"\nTraining complete! Best test acc: {best_test_acc:.4f}")
    return best_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad_path", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    global cfg
    cfg = Config_80M()
    cfg.num_cls = 4
    cfg.ckpt_path = args.ckpt_path
    cfg.device = args.device
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.nonz_len = 2048
    cfg.mask_ratio = 0.5
    cfg.use_bs = args.batch_size
    cfg.epoch = args.epochs
    
    # Change to cellfm directory
    original_dir = os.getcwd()
    os.chdir('/home/ubuntu/LLM-inference/xinze-project/cellfm')
    
    try:
        train_balanced(**vars(args))
    finally:
        os.chdir(original_dir)


if __name__ == '__main__':
    main()



