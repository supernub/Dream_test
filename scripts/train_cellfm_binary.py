#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellFM Fine-tune for Binary Classification (Not AD vs High)
This version supports:
- Binary classification (num_cls=2)
- Using binary_label instead of ADNC
- Donor-level split (based on train_donors/test_donors)
- Donor metadata features added to obs (for reference, CellFM may not directly use them)

Key modification:
- Gene embedding: Re-initialized with our data's vocab
- All other layers: Load pre-trained CellFM weights
- Binary labels: 0 (Not AD) vs 1 (High)
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
try:
    from mindspore.train.serialization import load_checkpoint  # type: ignore
    _HAS_MINDSPORE = True
except Exception:
    _HAS_MINDSPORE = False

# Add CellFM to path
sys.path.append('/home/ubuntu/LLM-inference/xinze-project/cellfm')
from model import Finetune_Cell_FM
from layers.utils import Config_80M, read_h5ad, map_gene_list, SCrna, build_dataset, Prepare

# Binary label mapping
BINARY_LABEL_MAP = {0: 'Not AD', 1: 'High'}


def prepare_adata_for_cellfm_binary(adata_path, split_json, label_column='binary_label', 
                                    concat_donor_metadata=False, metadata_csv=None, output_h5ad=None):
    """
    Prepare h5ad file for CellFM binary classification training format.
    
    Args:
        adata_path: Path to h5ad file
        split_json: Path to split JSON file with train_donors/test_donors
        label_column: Column name for binary labels (default: 'binary_label')
        concat_donor_metadata: Whether to add donor metadata features to obs
        metadata_csv: Optional path to CSV file with binary_label (if not in h5ad)
        output_h5ad: Optional path to save prepared h5ad
    """
    print(f"Loading data from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    # Load split information
    with open(split_json, 'r') as f:
        split_info = json.load(f)
    
    # Find donor column
    donor_col = None
    for col_candidate in ["Donor ID", "Donor_space_ID", "donor_id", "donor"]:
        if col_candidate in adata.obs.columns:
            donor_col = col_candidate
            break
    
    if donor_col is None:
        raise KeyError("No donor ID column found in obs. Expected: 'Donor ID', 'Donor_space_ID', 'donor_id', or 'donor'")
    
    print(f"Using donor column: {donor_col}")
    
    # Load binary_label from metadata CSV if not in h5ad
    if label_column not in adata.obs.columns:
        if metadata_csv is None:
            # Try to find metadata CSV in the same directory
            metadata_csv = os.path.join(os.path.dirname(adata_path), 'cells_metadata.csv')
        
        if os.path.exists(metadata_csv):
            print(f"Loading {label_column} from {metadata_csv}")
            metadata_df = pd.read_csv(metadata_csv)
            
            # Find cell barcode column
            barcode_col = None
            for col in ['cell_barcode', 'barcode', 'index']:
                if col in metadata_df.columns:
                    barcode_col = col
                    break
            
            if barcode_col is None:
                raise KeyError(f"Could not find cell barcode column in {metadata_csv}")
            
            # Merge binary_label into adata.obs
            if label_column in metadata_df.columns:
                metadata_subset = metadata_df[[barcode_col, label_column]].set_index(barcode_col)
                # Match by index (cell barcode)
                adata.obs[label_column] = metadata_subset.loc[adata.obs.index, label_column].values
                print(f"Added {label_column} to obs from metadata CSV")
            else:
                raise KeyError(f"Column '{label_column}' not found in {metadata_csv}")
        else:
            raise KeyError(f"Column '{label_column}' not found in obs and metadata CSV not found at {metadata_csv}")
    
    # Mark train/test split based on donors
    adata.obs['train'] = 2  # default to test
    
    # Use train_donors/test_donors for split
    if 'train_donors' in split_info:
        train_donors = split_info['train_donors']
        train_mask = adata.obs[donor_col].astype(str).isin(train_donors)
        adata.obs.loc[train_mask, 'train'] = 0  # train
        print(f"Train donors: {len(train_donors)} donors, {train_mask.sum()} cells")
    
    if 'test_donors' in split_info:
        test_donors = split_info['test_donors']
        test_mask = adata.obs[donor_col].astype(str).isin(test_donors)
        adata.obs.loc[test_mask, 'train'] = 2  # test
        print(f"Test donors: {len(test_donors)} donors, {test_mask.sum()} cells")
    
    # Ensure binary_label column exists
    if label_column not in adata.obs.columns:
        raise KeyError(f"Column '{label_column}' not found in obs. Available columns: {list(adata.obs.columns)}")
    
    # Normalize binary labels: ensure they are 0 or 1
    binary_series = adata.obs[label_column]
    if pd.api.types.is_numeric_dtype(binary_series):
        # Convert to int and ensure 0/1
        binary_norm = binary_series.astype(int)
        valid_mask = binary_norm.isin([0, 1])
    else:
        # Try to convert string labels
        s = binary_series.astype(str).str.lower()
        binary_norm = s.replace({'not ad': 0, 'high': 1, '0': 0, '1': 1}).astype(int)
        valid_mask = binary_norm.isin([0, 1])
    
    print(f"Valid cells with binary labels: {int(valid_mask.sum())}/{len(valid_mask)}")
    print(f"Label distribution: {binary_norm[valid_mask].value_counts().to_dict()}")
    
    # Filter valid cells
    if valid_mask.sum() == 0:
        raise ValueError("No valid binary labels found. Please check the dataset.")
    adata = adata[valid_mask].copy()
    
    # Create 'celltype' column (required by CellFM)
    if 'Subclass' in adata.obs.columns:
        adata.obs['celltype'] = adata.obs['Subclass']
    elif 'Supertype' in adata.obs.columns:
        adata.obs['celltype'] = adata.obs['Supertype']
    elif 'cell_type' in adata.obs.columns:
        adata.obs['celltype'] = adata.obs['cell_type']
    else:
        adata.obs['celltype'] = 'Unknown'
    
    # Create 'feat' column for classification target (binary: 0 or 1)
    adata.obs['feat'] = binary_norm[valid_mask].values.astype(int)
    
    # Set batch_id from donor
    adata.obs['batch_id'] = adata.obs[donor_col].astype('category').cat.codes
    
    # Add donor metadata features if requested
    if concat_donor_metadata:
        print("\nAdding donor metadata features to obs...")
        train_mask = adata.obs['train'] == 0
        donor_metadata = _build_donor_metadata_features(
            adata.obs, label_column=label_column, train_mask=train_mask.values
        )
        
        # Add each donor feature as a column in obs
        for i, feat_name in enumerate(donor_metadata['feature_names']):
            adata.obs[f'donor_meta_{feat_name}'] = donor_metadata['features'][:, i]
        
        print(f"Added {len(donor_metadata['feature_names'])} donor metadata features to obs")
        print(f"Feature names: {donor_metadata['feature_names']}")
    
    if output_h5ad:
        print(f"Saving prepared data to {output_h5ad}")
        os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
        adata.write(output_h5ad)
    
    return adata


def _build_donor_metadata_features(obs_df, label_column='binary_label', train_mask=None):
    """
    Build donor-level statistical features and broadcast to each cell.
    
    Returns:
        dict with 'features' (numpy array) and 'feature_names' (list)
    """
    # Find donor column
    donor_col = None
    for col_candidate in ["Donor ID", "Donor_space_ID", "donor_id", "donor"]:
        if col_candidate in obs_df.columns:
            donor_col = col_candidate
            break
    
    if donor_col is None:
        return {'features': np.array([]), 'feature_names': []}
    
    # Use only training set donors to compute statistics (avoid data leakage)
    if train_mask is not None:
        train_obs = obs_df.loc[train_mask].copy()
    else:
        train_obs = obs_df.copy()
    
    donor_stats = []
    
    # Cell count (log scale)
    cell_counts = train_obs.groupby(donor_col, observed=True).size()
    donor_feat_df = pd.DataFrame({
        'log_cell_count': np.log1p(cell_counts)
    }, index=cell_counts.index)
    
    # ADNC distribution if available
    if 'ADNC' in train_obs.columns:
        adnc_dummies = pd.get_dummies(train_obs[[donor_col, 'ADNC']], columns=['ADNC'], prefix='adnc')
        adnc_ratios = adnc_dummies.groupby(donor_col, observed=True).mean()
        donor_feat_df = pd.concat([donor_feat_df, adnc_ratios], axis=1)
    
    # Fill missing values
    donor_feat_df = donor_feat_df.fillna(0.0)
    
    # For test set donors not in training set, use mean values
    all_donors = obs_df[donor_col].unique()
    default_values = donor_feat_df.mean().to_dict()
    for donor in all_donors:
        if donor not in donor_feat_df.index:
            donor_feat_df.loc[donor] = default_values
    
    # Broadcast to each cell
    cell_donor_features = donor_feat_df.loc[obs_df[donor_col].values].values
    
    return {
        'features': cell_donor_features,
        'feature_names': donor_feat_df.columns.tolist()
    }


def load_pretrained_weights_without_gene_emb(model, ckpt_path, exclude_gene_emb=True):
    """Load pre-trained CellFM weights while excluding gene embedding layer."""
    if not _HAS_MINDSPORE:
        print("[WARN] MindSpore 未安装，跳过预训练权重加载（将从随机初始化开始训练除 gene_emb 外的层）。")
        return [], []

    print(f"Loading checkpoint from: {ckpt_path}")
    ms_ckpt = load_checkpoint(ckpt_path)
    
    def map_ms_to_pt(ms_key):
        name = ms_key
        name = name.replace("layer_norm.gamma", "weight")
        name = name.replace("layer_norm.beta", "bias")
        name = name.replace("post_norm1.gamma", "post_norm1.weight")
        name = name.replace("post_norm1.beta", "post_norm1.bias")
        name = name.replace("post_norm2.gamma", "post_norm2.weight")
        name = name.replace("post_norm2.beta", "post_norm2.bias")
        return name
    
    torch_state_dict = {}
    
    for ms_key, ms_param in ms_ckpt.items():
        pt_key = map_ms_to_pt(ms_key)
        
        # Skip gene embedding if requested
        if exclude_gene_emb and 'gene_emb' in pt_key:
            print(f"Skipping gene embedding weight: {pt_key}")
            continue
        
        pt_tensor = torch.tensor(ms_param.asnumpy())
        torch_state_dict[pt_key] = pt_tensor
    
    # Load only non-gene-emb weights
    missing_keys, unexpected_keys = model.extractor.net.load_state_dict(
        torch_state_dict, strict=False
    )
    
    print(f"[Load Report]")
    print(f"Successfully loaded {len(torch_state_dict)} parameters")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"First 10 missing keys: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"First 10 unexpected keys: {unexpected_keys[:10]}")
    
    return missing_keys, unexpected_keys


def train_cellfm_binary(h5ad_path, split_json, output_dir, ckpt_path, device=None, 
                        batch_size=16, epochs=5, lr=1e-4, num_cls=2,
                        label_column='binary_label', concat_donor_metadata=False,
                        metadata_csv=None, eval_interval_steps=200, patience_steps=3):
    """Train CellFM on binary classification task."""
    
    print("="*80)
    print("CellFM Fine-tune for Binary Classification (Not AD vs High)")
    print("="*80)
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f"CUDA available, using device: {device}")
        else:
            device = 'cpu'
            print(f"CUDA not available, using device: {device}")
            print("WARNING: Training on CPU will be very slow!")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print("\n1. Preparing data...")
    adata = prepare_adata_for_cellfm_binary(
        h5ad_path, split_json, 
        label_column=label_column,
        concat_donor_metadata=concat_donor_metadata,
        metadata_csv=metadata_csv
    )
    
    # Split into train/test
    train_adata = adata[adata.obs['train'] == 0].copy()
    test_adata = adata[adata.obs['train'] == 2].copy()
    print(f"Train cells: {len(train_adata)}, Test cells: {len(test_adata)}")
    
    # Check label distribution
    print(f"\nTrain label distribution:")
    print(train_adata.obs['feat'].value_counts().sort_index())
    print(f"\nTest label distribution:")
    print(test_adata.obs['feat'].value_counts().sort_index())
    
    # Save temporary h5ad files for utils.read_h5ad pipeline
    train_h5ad = os.path.join(output_dir, 'train.h5ad')
    test_h5ad = os.path.join(output_dir, 'test.h5ad')
    train_adata.write(train_h5ad)
    test_adata.write(test_h5ad)
    
    # Reload via utils (CellFM's read_h5ad does normalization)
    train_adata = read_h5ad(train_h5ad)
    test_adata = read_h5ad(test_h5ad)
    
    # Carry over obs fields
    train_raw = sc.read_h5ad(train_h5ad)
    test_raw = sc.read_h5ad(test_h5ad)
    for col in ['feat', 'celltype', 'batch_id']:
        train_adata.obs[col] = train_raw.obs[col].values
        test_adata.obs[col] = test_raw.obs[col].values
    
    n_genes = train_adata.n_vars
    print(f"Number of genes: {n_genes}")
    
    # Configure CellFM
    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.use_bs = batch_size
    cfg.mask_ratio = 0.5
    cfg.num_cls = num_cls  # Binary: 2 classes
    cfg.dataset = "Binary_Classification"
    cfg.feature_col = "feat"
    cfg.ckpt_path = ckpt_path
    cfg.device = device
    cfg.epoch = epochs
    
    # Change to CellFM directory for relative paths
    original_dir = os.getcwd()
    cellfm_dir = '/home/ubuntu/LLM-inference/xinze-project/cellfm'
    os.chdir(cellfm_dir)
    
    try:
        # Load data for scRNA dataset initialization
        print("\n2. Creating data loaders...")
        train_adata_load = train_adata
        train_adata_load.obs['train'] = 0
        train_adata_load.obs['celltype'] = train_adata_load.obs['celltype']
        train_adata_load.obs['feat'] = train_adata_load.obs['feat']

        test_adata_load = test_adata
        test_adata_load.obs['train'] = 2
        test_adata_load.obs['celltype'] = test_adata_load.obs['celltype']
        test_adata_load.obs['feat'] = test_adata_load.obs['feat']
        
        train_dataset = SCrna(train_adata_load, mode="train")
        test_dataset = SCrna(test_adata_load, mode="test")
        
        prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
        train_loader = build_dataset(train_dataset, prep=prep, batch_size=cfg.use_bs, 
                                     pad_zero=cfg.pad_zero, drop=True, shuffle=True)
        test_loader = build_dataset(test_dataset, prep=prep, batch_size=cfg.use_bs, 
                                   pad_zero=cfg.pad_zero, drop=False, shuffle=False)
        
        # Initialize model with our gene count
        print("\n3. Initializing CellFM model with custom gene embedding...")
        print(f"Using {n_genes} genes (will pad to 2048)")
        
        # Create model
        net = Finetune_Cell_FM(cfg)
        
        # Initialize gene embedding for our vocabulary
        print("Re-initializing gene embedding...")
        with torch.no_grad():
            emb_dim = net.extractor.net.gene_emb.shape[1]
            pad_to = ((n_genes - 1) // 8 + 1) * 8
            net.extractor.net.gene_emb = nn.Parameter(
                torch.empty(pad_to, emb_dim)
            )
            nn.init.xavier_normal_(net.extractor.net.gene_emb)
            net.extractor.net.gene_emb.data[0, :] = 0  # Zero pad
            print(f"Initialized new gene embedding: {net.extractor.net.gene_emb.shape}")
        
        # Set trainable parameters
        for name, param in net.named_parameters():
            param.requires_grad = "cls." in name or "encoder" in name
        
        print("\nTrainable parameters:")
        trainable_count = 0
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")
                trainable_count += 1
        print(f"Total trainable parameters: {trainable_count}")
        
        net = net.to(cfg.device)
        
        # Load pre-trained weights (excluding gene embedding)
        print("\n4. Loading pre-trained weights (excluding gene embedding)...")
        missing_keys, unexpected_keys = load_pretrained_weights_without_gene_emb(
            net, cfg.ckpt_path, exclude_gene_emb=True
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 
                                     lr=lr, weight_decay=1e-5)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        
        # Calculate class weights for balanced training (binary)
        print("\nCalculating class weights for balanced training...")
        train_labels_array = train_adata_load.obs['feat'].values.astype(int)
        class_counts = np.bincount(train_labels_array, minlength=2)
        class_weights = 1.0 / (class_counts.astype(float) + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_cls
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        print(f"Class counts: {dict(zip(['Not AD', 'High'], class_counts))}")
        print(f"Class weights: {dict(zip(['Not AD', 'High'], class_weights))}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Training loop
        print("\n5. Starting training...")
        best_test_acc = 0.0
        best_test_f1 = 0.0
        train_acc_history = []
        test_acc_history = []
        
        global_step = 0
        no_improve_steps = 0

        for epoch in range(cfg.epoch):
            # Training
            net.train()
            running_loss = 0.0
            running_acc = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epoch}")
            
            for step, batch in enumerate(progress):
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
                
                cls_loss = criterion(cls.float(), batch['feat'].long())
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

                # Early stopping by steps
                global_step += 1
                if eval_interval_steps > 0 and global_step % eval_interval_steps == 0:
                    net.eval()
                    eval_all_preds = []
                    eval_all_labels = []
                    eval_all_probs = []
                    with torch.no_grad():
                        for tbatch in test_loader:
                            for tk in tbatch:
                                if isinstance(tbatch[tk], torch.Tensor):
                                    tbatch[tk] = tbatch[tk].to(cfg.device)
                            with torch.cuda.amp.autocast():
                                tcls, tmask_loss, tcls_token = net(
                                    raw_nzdata=tbatch['raw_nzdata'],
                                    dw_nzdata=tbatch['dw_nzdata'],
                                    ST_feat=tbatch['ST_feat'],
                                    nonz_gene=tbatch['nonz_gene'],
                                    mask_gene=tbatch['mask_gene'],
                                    zero_idx=tbatch['zero_idx']
                                )
                            tpred = tcls.argmax(1)
                            tprob = F.softmax(tcls, dim=1)[:, 1]  # Probability of class 1
                            eval_all_preds.extend(tpred.cpu().numpy())
                            eval_all_labels.extend(tbatch['feat'].cpu().numpy())
                            eval_all_probs.extend(tprob.cpu().numpy())
                    
                    step_test_acc = accuracy_score(eval_all_labels, eval_all_preds)
                    step_test_f1 = f1_score(eval_all_labels, eval_all_preds)
                    
                    # Improvement check
                    if step_test_acc > best_test_acc:
                        best_test_acc = step_test_acc
                        best_test_f1 = step_test_f1
                        torch.save(net.state_dict(), f"{output_dir}/best_model.pth")
                        no_improve_steps = 0
                        print(f"✓ Step {global_step}: new best acc {best_test_acc:.6f}, F1 {best_test_f1:.6f}, checkpoint saved")
                    else:
                        no_improve_steps += 1
                        print(f"✗ Step {global_step}: no improvement ({step_test_acc:.6f} ≤ {best_test_acc:.6f}), no_improve_steps={no_improve_steps}/{patience_steps}")
                        if no_improve_steps >= patience_steps:
                            print("Early stopping triggered by steps (no improvement).")
                            raise StopIteration
                    net.train()
            
            scheduler.step()
            train_acc_history.append(avg_acc)
            print(f"Epoch {epoch+1} completed, avg_loss: {avg_loss:.6f}, avg_acc: {avg_acc:.6f}")
            
            # Save checkpoint
            torch.save(net.state_dict(), f"{output_dir}/checkpoint_epoch_{epoch+1}.pth")
        
        # Final evaluation
        net.eval()
        print("\n6. Final evaluation...")
        all_preds = []
        all_labels = []
        all_probs = []
        
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
                prob = F.softmax(cls, dim=1)[:, 1]  # Probability of class 1
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch['feat'].cpu().numpy())
                all_probs.extend(prob.cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds, zero_division=0)
        test_recall = recall_score(all_labels, all_preds, zero_division=0)
        try:
            test_roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            test_roc_auc = float('nan')
        
        test_acc_history.append(test_acc)
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {test_acc:.6f}")
        print(f"  F1 Score: {test_f1:.6f}")
        print(f"  Precision: {test_precision:.6f}")
        print(f"  Recall: {test_recall:.6f}")
        print(f"  ROC-AUC: {test_roc_auc:.6f}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
            torch.save(net.state_dict(), f"{output_dir}/best_model.pth")
            print(f"✓ New best model saved with test accuracy: {best_test_acc:.6f}")
        
        # Save results
        results = {
            'train_acc_history': train_acc_history,
            'test_acc_history': test_acc_history,
            'best_test_acc': float(best_test_acc),
            'best_test_f1': float(best_test_f1),
            'test_metrics': {
                'accuracy': float(test_acc),
                'f1': float(test_f1),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'roc_auc': float(test_roc_auc)
            },
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'num_classes': num_cls,
            'n_genes': int(n_genes),
            'gene_embedding': 'reinitialized_for_our_vocab',
            'concat_donor_metadata': concat_donor_metadata
        }
        
        with open(f"{output_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print(f"Training completed! Best test accuracy: {best_test_acc:.6f}, F1: {best_test_f1:.6f}")
        print(f"Results saved to: {output_dir}")
        print("="*80)
    
    finally:
        os.chdir(original_dir)
    
    return best_test_acc


def main():
    parser = argparse.ArgumentParser(description="CellFM Fine-tune for Binary Classification")
    parser.add_argument("--h5ad_path", required=True, help="Path to h5ad file")
    parser.add_argument("--split_json", required=True, help="Path to split JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for model and results")
    parser.add_argument("--ckpt_path", required=True, help="Path to CellFM pre-trained checkpoint")
    parser.add_argument("--device", default=None, help="Device to use (default: auto-detect, cuda:0 if available, else cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_cls", type=int, default=2, help="Number of classes (default: 2 for binary)")
    parser.add_argument("--label-column", type=str, default="binary_label", help="Column name for binary labels")
    parser.add_argument("--concat-donor-metadata", action="store_true", help="Add donor metadata features to obs")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Path to metadata CSV with binary_label (if not in h5ad)")
    parser.add_argument("--eval_interval_steps", type=int, default=200, help="Evaluate on test set every N steps")
    parser.add_argument("--patience_steps", type=int, default=3, help="Early stopping patience")
    
    args = parser.parse_args()
    
    try:
        train_cellfm_binary(
            h5ad_path=args.h5ad_path,
            split_json=args.split_json,
            output_dir=args.output_dir,
            ckpt_path=args.ckpt_path,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_cls=args.num_cls,
            label_column=args.label_column,
            concat_donor_metadata=args.concat_donor_metadata,
            metadata_csv=args.metadata_csv,
            eval_interval_steps=args.eval_interval_steps,
            patience_steps=args.patience_steps
        )
    except StopIteration:
        print("Training stopped early due to no improvement.")


if __name__ == "__main__":
    main()

