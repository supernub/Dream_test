#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellFM Fine-tune for ADNC prediction - Custom Gene Embedding Version
This version re-initializes gene embedding for our custom vocabulary while
loading pre-trained weights for all other layers.

Key modification:
- Gene embedding: Re-initialized with our data's vocab
- All other layers: Load pre-trained CellFM weights
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
try:
    from mindspore.train.serialization import load_checkpoint  # type: ignore
    _HAS_MINDSPORE = True
except Exception:
    _HAS_MINDSPORE = False

# Add CellFM to path
sys.path.append('/home/ubuntu/LLM-inference/xinze-project/cellfm')
from model import Finetune_Cell_FM
from layers.utils import Config_80M, read_h5ad, map_gene_list, SCrna, build_dataset, Prepare

# ADNC mapping
ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
ADNC_ALLOWED = set(ADNC_MAP.keys())


def prepare_adata_for_cellfm(adata_path, split_json, output_h5ad=None):
    """Prepare h5ad file for CellFM training format with gene vocab mapping."""
    print(f"Loading data from {adata_path}")
    # Load into memory to allow filtering and column edits
    adata = sc.read_h5ad(adata_path)
    
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
        raise KeyError(f"Column '{adnc_col}' not found in obs.")
    
    # Normalize ADNC labels: support numeric {0,1,2,3} and string names
    adnc_series = adata.obs[adnc_col]
    if pd.api.types.is_numeric_dtype(adnc_series):
        # Convert 0/1/2/3 to names
        num_to_name = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}
        adnc_norm = adnc_series.map(num_to_name).astype(str)
    else:
        s = adnc_series.astype(str)
        map_num = {'0': 'Not AD', '1': 'Low', '2': 'Intermediate', '3': 'High'}
        adnc_norm = s.replace(map_num)

    # Build valid mask after normalization
    valid_mask = adnc_norm.isin(ADNC_ALLOWED)
    
    print(f"Valid cells with ADNC labels: {int(valid_mask.sum())}/{len(valid_mask)}")
    
    # Filter valid cells
    if valid_mask.sum() == 0:
        print("[WARN] No valid ADNC labels found after normalization. Please check the dataset.")
    adata = adata[valid_mask].copy()
    
    # Create 'celltype' column (required by CellFM)
    if 'Supertype' in adata.obs.columns:
        adata.obs['celltype'] = adata.obs['Supertype']
    elif 'cell_type' in adata.obs.columns:
        adata.obs['celltype'] = adata.obs['cell_type']
    else:
        adata.obs['celltype'] = 'Unknown'
    
    # Create 'feat' column for classification target based on normalized labels
    adata.obs['feat'] = [ADNC_MAP[str(lbl)] for lbl in adnc_norm[valid_mask].values]
    
    # Set batch_id
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


def load_pretrained_weights_without_gene_emb(model, ckpt_path, exclude_gene_emb=True):
    """
    Load pre-trained CellFM weights while excluding gene embedding layer.
    
    Args:
        model: Finetune_Cell_FM model
        ckpt_path: Path to pre-trained checkpoint
        exclude_gene_emb: If True, skip loading gene_emb weights
    
    Returns:
        Dictionary of successfully loaded parameters
    """
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


def train_cellfm(h5ad_path, split_json, output_dir, ckpt_path, device='cuda:0', 
                 batch_size=16, epochs=5, lr=1e-4, num_cls=4,
                 train_h5ad: str | None = None, test_h5ad: str | None = None,
                 eval_interval_steps=200, patience_steps=3):
    """Train CellFM on ADNC classification task with custom gene embedding."""
    
    print("="*80)
    print("CellFM Fine-tune for ADNC Prediction (Custom Gene Vocab)")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print("\n1. Preparing data...")
    if train_h5ad and test_h5ad:
        # Use pre-generated subsets to avoid loading the full dataset into memory
        print(f"Using provided subsets:\n  train: {train_h5ad}\n  test:  {test_h5ad}")
        train_raw = sc.read_h5ad(train_h5ad)
        test_raw = sc.read_h5ad(test_h5ad)

        # Ensure required columns exist
        def ensure_obs_fields(ad):
            # ADNC → feat
            adnc_col = 'ADNC'
            adnc_series = ad.obs[adnc_col] if adnc_col in ad.obs.columns else ad.obs.iloc[:, 0]
            # normalize labels
            if pd.api.types.is_numeric_dtype(adnc_series):
                num_to_name = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}
                adnc_norm = adnc_series.map(num_to_name).astype(str)
            else:
                s = adnc_series.astype(str)
                map_num = {'0': 'Not AD', '1': 'Low', '2': 'Intermediate', '3': 'High'}
                adnc_norm = s.replace(map_num)
            ad.obs['feat'] = [ADNC_MAP.get(lbl, 0) for lbl in adnc_norm.values]
            # celltype
            if 'celltype' not in ad.obs.columns:
                if 'Supertype' in ad.obs.columns:
                    ad.obs['celltype'] = ad.obs['Supertype']
                elif 'cell_type' in ad.obs.columns:
                    ad.obs['celltype'] = ad.obs['cell_type']
                else:
                    ad.obs['celltype'] = 'Unknown'
            # batch_id from donor
            donor_col = 'Donor ID' if 'Donor ID' in ad.obs.columns else ('Donor_space_ID' if 'Donor_space_ID' in ad.obs.columns else None)
            if donor_col:
                ad.obs['batch_id'] = ad.obs[donor_col].astype('category').cat.codes
            else:
                ad.obs['batch_id'] = 0

        ensure_obs_fields(train_raw)
        ensure_obs_fields(test_raw)

        # Load via CellFM utils for normalization, then carry over obs columns
        train_adata = read_h5ad(train_h5ad)
        test_adata = read_h5ad(test_h5ad)
        # carry over obs fields
        for col in ['feat', 'celltype', 'batch_id']:
            train_adata.obs[col] = train_raw.obs[col].values
            test_adata.obs[col] = test_raw.obs[col].values
        n_genes = train_adata.n_vars
        print(f"Train cells: {len(train_adata)}, Test cells: {len(test_adata)}")
    else:
        # Fallback: derive split from full h5ad (may be heavy for MTG)
        adata = prepare_adata_for_cellfm(h5ad_path, split_json)
        # Split into train/test
        train_adata = adata[adata.obs['train'] == 0].copy()
        test_adata = adata[adata.obs['train'] == 2].copy()
        print(f"Train cells: {len(train_adata)}, Test cells: {len(test_adata)}")
        # Save temporary h5ad files for utils.read_h5ad pipeline
        train_h5ad = os.path.join(output_dir, 'train.h5ad')
        test_h5ad = os.path.join(output_dir, 'test.h5ad')
        train_adata.write(train_h5ad)
        test_adata.write(test_h5ad)
        # reload via utils
        train_adata = read_h5ad(train_h5ad)
        test_adata = read_h5ad(test_h5ad)
        n_genes = train_adata.n_vars

    # Gene vocab size
    our_genes = list(train_adata.var_names)
    n_genes = len(our_genes)
    print(f"Our data has {n_genes} genes")
    print(f"CellFM default uses 27855 genes")
    print(f"Will re-initialize gene embedding for our {n_genes} genes")
    
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
        # 注意：训练时开启 drop_last 以避免 BatchNorm 在 batch=1 时报错
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
            # Get original embedding dimension
            emb_dim = net.extractor.net.gene_emb.shape[1]
            
            # Create new embedding for our genes (will be padded to 2048)
            # We pad to nearest multiple of 8 for CellFM requirement
            pad_to = ((n_genes - 1) // 8 + 1) * 8
            net.extractor.net.gene_emb = nn.Parameter(
                torch.empty(pad_to, emb_dim)
            )
            nn.init.xavier_normal_(net.extractor.net.gene_emb)
            
            # Zero pad is always zero
            net.extractor.net.gene_emb.data[0, :] = 0
            
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
        
        # Calculate class weights for balanced training
        print("\nCalculating class weights for balanced training...")
        train_labels_array = train_adata_load.obs['feat'].values.astype(int)
        class_counts = np.bincount(train_labels_array)
        class_weights = 1.0 / class_counts.astype(float)
        class_weights = class_weights / class_weights.sum() * num_cls
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        print(f"Class counts: {dict(zip(['Not AD', 'Low', 'Intermediate', 'High'], class_counts))}")
        print(f"Class weights: {dict(zip(['Not AD', 'Low', 'Intermediate', 'High'], class_weights))}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Training loop
        print("\n5. Starting training...")
        best_test_acc = 0.0
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
                
                # Ensure logits are float32 when computing CE loss to avoid Half vs Float mismatch
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

            # early stopping by steps: periodic evaluation on test set
            global_step += 1
            if eval_interval_steps > 0 and global_step % eval_interval_steps == 0:
                net.eval()
                eval_running_acc = 0.0
                eval_all_preds = []
                eval_all_labels = []
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
                        teval_acc = (tpred == tbatch['feat']).float().mean().item()
                        eval_running_acc += teval_acc
                        eval_all_preds.extend(tpred.cpu().numpy())
                        eval_all_labels.extend(tbatch['feat'].cpu().numpy())
                step_test_acc = eval_running_acc / len(test_loader)
                # improvement check
                if step_test_acc > best_test_acc:
                    best_test_acc = step_test_acc
                    torch.save(net.state_dict(), f"{output_dir}/best_model.pth")
                    no_improve_steps = 0
                    print(f"✓ Step {global_step}: new best acc {best_test_acc:.6f}, checkpoint saved")
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
            'num_classes': num_cls,
            'n_genes': n_genes,
            'gene_embedding': 'reinitialized_for_our_vocab'
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
    parser = argparse.ArgumentParser(description="CellFM Fine-tune for ADNC Prediction (Custom Gene Vocab)")
    parser.add_argument("--h5ad_path", required=True, help="Path to h5ad file")
    parser.add_argument("--split_json", required=True, help="Path to split JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for model and results")
    parser.add_argument("--ckpt_path", required=True, help="Path to CellFM pre-trained checkpoint")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_cls", type=int, default=4, help="Number of classes (ADNC levels)")
    parser.add_argument("--train_h5ad", default=None, help="Optional pre-made train subset h5ad")
    parser.add_argument("--test_h5ad", default=None, help="Optional pre-made test subset h5ad")
    parser.add_argument("--eval_interval_steps", type=int, default=200, help="Evaluate on test set every N steps for early stopping")
    parser.add_argument("--patience_steps", type=int, default=3, help="Early stopping patience (number of evals without improvement)")
    
    args = parser.parse_args()
    
    try:
        train_cellfm(
        h5ad_path=args.h5ad_path,
        split_json=args.split_json,
        output_dir=args.output_dir,
        ckpt_path=args.ckpt_path,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_cls=args.num_cls,
        train_h5ad=args.train_h5ad,
        test_h5ad=args.test_h5ad,
        eval_interval_steps=args.eval_interval_steps,
        patience_steps=args.patience_steps
    )
    except StopIteration:
        print("Training stopped early due to no improvement.")


if __name__ == "__main__":
    main()

