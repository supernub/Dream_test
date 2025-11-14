#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellFM Fine-tune for ADNC prediction on MTG dataset.
Adapts CellFM-torch for ADNC (Alzheimer's Disease Neuropathologic Change) classification.

Usage:
    python train_cellfm_adnc.py --h5ad_path path/to/data.h5ad --output_dir outputs/cellfm_model
    
This script:
1. Loads pre-trained CellFM 80M weights
2. Fine-tunes on ADNC classification task
3. Evaluates on test set and reports accuracy
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Add CellFM to path
sys.path.append('/home/ubuntu/LLM-inference/xinze-project/cellfm')
from model import Finetune_Cell_FM
from layers.utils import Config_80M, read_h5ad, map_gene_list, SCrna, build_dataset, Prepare

# ADNC mapping
ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
ADNC_ALLOWED = set(ADNC_MAP.keys())


def prepare_adata_for_cellfm(adata_path, split_json, output_h5ad=None):
    """
    Prepare h5ad file for CellFM training format.
    
    Args:
        adata_path: Path to original h5ad file
        split_json: Path to split JSON file
        output_h5ad: Optional path to save prepared h5ad
    """
    print(f"Loading data from {adata_path}")
    adata = sc.read_h5ad(adata_path, backed='r')
    
    # Load split information
    with open(split_json, 'r') as f:
        split_info = json.load(f)
    
    # Mark train/test split
    adata.obs['train'] = 2  # default to test
    if 'train_indices' in split_info:
        train_idx = split_info['train_indices']
        adata.obs.loc[adata.obs.index[train_idx], 'train'] = 0  # train
    if 'test_indices' in split_info:
        test_idx = split_info['test_indices']
        adata.obs.loc[adata.obs.index[test_idx], 'train'] = 2  # test
    
    # Ensure ADNC column exists
    adnc_col = split_info.get('adnc_col', 'ADNC')
    if adnc_col not in adata.obs.columns:
        raise KeyError(f"Column '{adnc_col}' not found in obs. Available columns: {list(adata.obs.columns)}")
    
    # Map ADNC labels
    adnc_labels = adata.obs[adnc_col].astype(str).values
    valid_mask = ~pd.isna(adnc_labels) & (adnc_labels != 'nan') & (adnc_labels != '')
    valid_mask = valid_mask & np.isin(adnc_labels, ADNC_ALLOWED)
    
    print(f"Valid cells with ADNC labels: {valid_mask.sum()}/{len(valid_mask)}")
    
    # Filter valid cells
    adata = adata[valid_mask].copy()
    
    # Create 'celltype' column (required by CellFM)
    if 'Supertype' in adata.obs.columns:
        adata.obs['celltype'] = adata.obs['Supertype']
    elif 'cell_type' in adata.obs.columns:
        adata.obs['celltype'] = adata.obs['cell_type']
    else:
        adata.obs['celltype'] = 'Unknown'
    
    # Create 'feat' column for classification target
    adata.obs['feat'] = [ADNC_MAP[label] for label in adata.obs[adnc_col].values]
    
    # Set batch_id (using donor ID if available)
    donor_col = split_info.get('donor_col', 'Donor ID')
    if donor_col in adata.obs.columns:
        adata.obs['batch_id'] = adata.obs[donor_col].astype('category').cat.codes
    else:
        adata.obs['batch_id'] = 0
    
    if output_h5ad:
        print(f"Saving prepared data to {output_h5ad}")
        os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
        adata.write(output_h5ad)
    
    return adata


def train_cellfm(h5ad_path, split_json, output_dir, ckpt_path, device='cuda:0', 
                 batch_size=16, epochs=5, lr=1e-4, num_cls=4):
    """Train CellFM on ADNC classification task."""
    
    print("="*80)
    print("CellFM Fine-tune for ADNC Prediction")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print("\n1. Preparing data...")
    adata = prepare_adata_for_cellfm(h5ad_path, split_json)
    
    # Split into train/test
    train_adata = adata[adata.obs['train'] == 0].copy()
    test_adata = adata[adata.obs['train'] == 2].copy()
    
    print(f"Train cells: {len(train_adata)}, Test cells: {len(test_adata)}")
    
    # Prepare for CellFM data loading
    # Save temporary h5ad files
    train_h5ad = os.path.join(output_dir, 'train.h5ad')
    test_h5ad = os.path.join(output_dir, 'test.h5ad')
    train_adata.write(train_h5ad)
    test_adata.write(test_h5ad)
    
    # Configure CellFM
    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.use_bs = batch_size
    cfg.mask_ratio = 0.5
    cfg.num_cls = num_cls
    cfg.dataset = "MTG_ADNC"
    cfg.feature_col = "feat"
    cfg.ckpt_path = ckpt_path
    cfg.device = device
    cfg.epoch = epochs
    
    # Change to CellFM directory for relative paths
    original_dir = os.getcwd()
    cellfm_dir = '/home/ubuntu/LLM-inference/xinze-project/cellfm'
    os.chdir(cellfm_dir)
    
    try:
        # Load data loaders
        print("\n2. Creating data loaders...")
        train_adata = read_h5ad(train_h5ad)
        train_adata.obs['train'] = 0
        train_adata.obs['celltype'] = train_adata.obs['celltype']
        train_adata.obs['feat'] = train_adata.obs['feat'].astype('category').cat.codes
        
        test_adata = read_h5ad(test_h5ad)
        test_adata.obs['train'] = 2
        test_adata.obs['celltype'] = test_adata.obs['celltype']
        test_adata.obs['feat'] = test_adata.obs['feat'].astype('category').cat.codes
        
        train_dataset = SCrna(train_adata, mode="train")
        test_dataset = SCrna(test_adata, mode="test")
        
        prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
        train_loader = build_dataset(train_dataset, prep=prep, batch_size=cfg.use_bs, 
                                     pad_zero=cfg.pad_zero, drop=False, shuffle=True)
        test_loader = build_dataset(test_dataset, prep=prep, batch_size=cfg.use_bs, 
                                   pad_zero=cfg.pad_zero, drop=False, shuffle=False)
        
        # Initialize model
        print("\n3. Initializing CellFM model...")
        net = Finetune_Cell_FM(cfg)
        
        # Set trainable parameters
        for name, param in net.named_parameters():
            param.requires_grad = "cls." in name or "encoder" in name
        
        print("Trainable parameters:")
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(f"  {name}")
        
        net = net.to(cfg.device)
        net.extractor.load_model(weight=True, moment=False)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 
                                     lr=lr, weight_decay=1e-5)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        print("\n4. Starting training...")
        best_test_acc = 0.0
        train_acc_history = []
        test_acc_history = []
        
        for epoch in range(cfg.epoch):
            # Training
            net.train()
            running_loss = 0.0
            running_acc = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epoch}")
            
            for step, batch in enumerate(progress):
                # Move to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(cfg.device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    cls, mask_loss, cls_token = net(
                        raw_nzdata=batch['raw_nzdata'],
                        dw_nzdata=batch['dw_nzdata'],
                        ST_feat=batch['ST_feat'],
                        nonz_gene=batch['nonz_gene'],
                        mask_gene=batch['mask_gene'],
                        zero_idx=batch['zero_idx']
                    )
                    
                    cls_loss = criterion(cls, batch['feat'].long())
                    loss = mask_loss + cls_loss
                
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                accuracy = (cls.argmax(1) == batch['feat']).float().mean().item()
                running_loss += loss.item()
                running_acc += accuracy
                
                avg_loss = running_loss / (step + 1)
                avg_acc = running_acc / (step + 1)
                progress.set_postfix(loss=avg_loss, acc=avg_acc)
            
            scheduler.step()
            train_acc_history.append(avg_acc)
            print(f"Epoch {epoch+1} completed, avg_loss: {avg_loss:.6f}, avg_acc: {avg_acc:.6f}")
            
            # Save checkpoint
            torch.save(net.state_dict(), f"{output_dir}/checkpoint_epoch_{epoch+1}.pth")
            
            # Evaluation
            net.eval()
            print("Evaluating...")
            running_acc = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing"):
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].to(cfg.device)
                    
                    with torch.cuda.amp.autocast():
                        cls, mask_loss, cls_token = net(
                            raw_nzdata=batch['raw_nzdata'],
                            dw_nzdata=batch['dw_nzdata'],
                            ST_feat=batch['ST_feat'],
                            nonz_gene=batch['nonz_gene'],
                            mask_gene=batch['mask_gene'],
                            zero_idx=batch['zero_idx']
                        )
                    
                    pred = cls.argmax(1)
                    accuracy = (pred == batch['feat']).float().mean().item()
                    running_acc += accuracy
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(batch['feat'].cpu().numpy())
            
            test_acc = running_acc / len(test_loader)
            test_acc_history.append(test_acc)
            
            # Calculate F1 score
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            print(f"Test accuracy: {test_acc:.6f}, Test F1: {f1:.6f}")
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(net.state_dict(), f"{output_dir}/best_model.pth")
                print(f"✓ New best model saved with test accuracy: {best_test_acc:.6f}")
        
        # Save results
        results = {
            'train_acc_history': train_acc_history,
            'test_acc_history': test_acc_history,
            'best_test_acc': best_test_acc,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'num_classes': num_cls
        }
        
        with open(f"{output_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print(f"Training completed! Best test accuracy: {best_test_acc:.6f}")
        print(f"Results saved to: {output_dir}")
        print("="*80)
        
    finally:
        os.chdir(original_dir)
    
    return best_test_acc


def main():
    parser = argparse.ArgumentParser(description="CellFM Fine-tune for ADNC Prediction")
    parser.add_argument("--h5ad_path", required=True, help="Path to h5ad file")
    parser.add_argument("--split_json", required=True, help="Path to split JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for model and results")
    parser.add_argument("--ckpt_path", default=None, 
                       help="Path to CellFM pre-trained checkpoint (will download if not provided)")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_cls", type=int, default=4, help="Number of classes (ADNC levels)")
    
    args = parser.parse_args()
    
    # Download checkpoint if needed
    if args.ckpt_path is None:
        print("No checkpoint provided. Please download CellFM weights from:")
        print("https://huggingface.co/ShangguanNingyuan/CellFM/tree/main")
        return
    
    train_cellfm(
        h5ad_path=args.h5ad_path,
        split_json=args.split_json,
        output_dir=args.output_dir,
        ckpt_path=args.ckpt_path,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_cls=args.num_cls
    )


if __name__ == "__main__":
    main()




