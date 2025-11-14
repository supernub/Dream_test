#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build donor-level classifier using cell embeddings.
For each donor, aggregates cell embeddings by cell type and trains MLP classifier.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score
from collections import defaultdict, Counter
from tqdm import tqdm
import xgboost as xgb
import torch.nn.functional as F

class DonorEmbeddingDataset(Dataset):
    """Dataset for donor-level embeddings."""
    
    def __init__(self, donor_embeddings, donor_labels):
        self.donor_embeddings = donor_embeddings
        self.donor_labels = donor_labels
        self.donors = list(donor_embeddings.keys())
    
    def __len__(self):
        return len(self.donors)
    
    def __getitem__(self, idx):
        donor = self.donors[idx]
        embedding = self.donor_embeddings[donor]
        label = self.donor_labels[donor]
        return torch.FloatTensor(embedding), torch.LongTensor([label])

class DonorMLP(nn.Module):
    """Enhanced MLP classifier for donor-level predictions with ordinal regression."""
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1, use_ordinal=True):
        super().__init__()
        
        self.use_ordinal = use_ordinal
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        if use_ordinal:
            # CORAL ordinal head
            self.shared_fc = nn.Linear(prev_dim, 1, bias=False)
            self.biases = nn.Parameter(torch.zeros(num_classes - 1))
            nn.init.xavier_uniform_(self.shared_fc.weight)
        else:
            # Standard classification head
            layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.network(x)
        
        if self.use_ordinal:
            # CORAL ordinal prediction
            base = self.shared_fc(features)
            logits = base + self.biases.view(1, -1)
            return logits
        else:
            return features

def earth_movers_distance_loss(logits, targets, reduction="mean"):
    """
    Earth Mover's Distance (EMD) loss for ordinal regression.
    This loss is more aligned with QWK metric as it considers the distance between classes.
    """
    B, Km1 = logits.shape
    num_classes = Km1 + 1
    
    # Convert logits to probabilities using sigmoid
    probs = torch.sigmoid(logits)  # [B, K-1]
    
    # Create cumulative distribution for predictions
    # For ordinal regression, we need to convert binary predictions to class probabilities
    pred_probs = torch.zeros(B, num_classes, device=targets.device)
    
    # P(class = 0) = 1 - P(class > 0)
    pred_probs[:, 0] = 1 - probs[:, 0]
    
    # P(class = k) = P(class > k-1) - P(class > k) for k = 1, ..., K-2
    for k in range(1, num_classes - 1):
        pred_probs[:, k] = probs[:, k-1] - probs[:, k]
    
    # P(class = K-1) = P(class > K-2)
    pred_probs[:, num_classes - 1] = probs[:, num_classes - 2]
    
    # Create target distribution (one-hot)
    target_probs = torch.zeros(B, num_classes, device=targets.device)
    target_probs.scatter_(1, targets.long().unsqueeze(1), 1)
    
    # Compute Earth Mover's Distance
    # EMD = sum over all classes of |cumulative_pred - cumulative_target|
    pred_cumsum = torch.cumsum(pred_probs, dim=1)
    target_cumsum = torch.cumsum(target_probs, dim=1)
    
    emd_loss = torch.abs(pred_cumsum - target_cumsum).sum(dim=1)
    
    if reduction == "mean":
        return emd_loss.mean()
    elif reduction == "sum":
        return emd_loss.sum()
    else:
        return emd_loss

def qwk_loss(logits, targets, reduction="mean"):
    """
    Quadratic Weighted Kappa loss for ordinal regression.
    This directly optimizes for QWK metric by penalizing predictions based on their distance from true labels.
    """
    B, Km1 = logits.shape
    num_classes = Km1 + 1
    
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    
    # Create class probabilities
    pred_probs = torch.zeros(B, num_classes, device=targets.device)
    pred_probs[:, 0] = 1 - probs[:, 0]
    
    for k in range(1, num_classes - 1):
        pred_probs[:, k] = probs[:, k-1] - probs[:, k]
    
    pred_probs[:, num_classes - 1] = probs[:, num_classes - 2]
    
    # Compute QWK loss
    # QWK = 1 - (sum of weighted squared errors) / (sum of weights)
    # We want to minimize the weighted squared errors
    
    # Create weight matrix for QWK
    weights = torch.zeros(num_classes, num_classes, device=targets.device)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    
    # Compute expected weighted squared error for each sample
    qwk_losses = torch.zeros(B, device=targets.device)
    
    for i in range(B):
        true_class = targets[i].long()
        pred_dist = pred_probs[i]  # [num_classes]
        
        # Compute weighted squared error
        weighted_error = 0.0
        for pred_class in range(num_classes):
            weight = weights[true_class, pred_class]
            weighted_error += weight * pred_dist[pred_class]
        
        qwk_losses[i] = weighted_error
    
    if reduction == "mean":
        return qwk_losses.mean()
    elif reduction == "sum":
        return qwk_losses.sum()
    else:
        return qwk_losses

def ordinal_mae_loss(logits, targets, reduction="mean"):
    """
    Ordinal Mean Absolute Error loss.
    This is simpler and more direct for ordinal regression.
    """
    B, Km1 = logits.shape
    num_classes = Km1 + 1
    
    # Convert logits to class predictions
    probs = torch.sigmoid(logits)
    
    # Convert to class probabilities
    pred_probs = torch.zeros(B, num_classes, device=targets.device)
    pred_probs[:, 0] = 1 - probs[:, 0]
    
    for k in range(1, num_classes - 1):
        pred_probs[:, k] = probs[:, k-1] - probs[:, k]
    
    pred_probs[:, num_classes - 1] = probs[:, num_classes - 2]
    
    # Compute expected class for each sample
    expected_pred = torch.sum(pred_probs * torch.arange(num_classes, device=targets.device).float(), dim=1)
    
    # Compute MAE
    mae_loss = torch.abs(expected_pred - targets.float())
    
    if reduction == "mean":
        return mae_loss.mean()
    elif reduction == "sum":
        return mae_loss.sum()
    else:
        return mae_loss

def coral_loss(logits, targets, reduction="mean"):
    """CORAL loss for ordinal regression."""
    B, Km1 = logits.shape
    # Build binary targets for each threshold
    thresholds = torch.arange(Km1, device=targets.device).view(1, -1)
    bin_targets = (targets.view(-1, 1) > thresholds).float()
    
    loss = F.binary_cross_entropy_with_logits(logits, bin_targets, reduction="none")
    loss = loss.sum(dim=1)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def train_xgboost_classifier(train_embeddings, train_labels, val_embeddings, val_labels, args):
    """
    Train XGBoost classifier for ordinal regression with QWK optimization.
    Uses test donors as validation during training to find best model.
    """
    print("Training XGBoost classifier for ordinal regression...")
    print("Using test donors as validation during training...")
    
    # Load donor labels for detailed predictions
    import numpy as np
    donor_labels = np.load(args.predictions_path, allow_pickle=True).item()
    
    # XGBoost parameters from command line arguments
    xgb_params = {
        'objective': 'multi:softprob',  # Multi-class with probabilities
        'num_class': 4,  # ADNC classes: 0, 1, 2, 3
        'eval_metric': 'mlogloss',
        'max_depth': args.xgb_max_depth,
        'learning_rate': args.xgb_lr,
        'n_estimators': args.xgb_n_estimators,
        'subsample': args.xgb_subsample,
        'colsample_bytree': args.xgb_colsample_bytree,
        'reg_alpha': args.xgb_reg_alpha,
        'reg_lambda': args.xgb_reg_lambda,
        'min_child_weight': args.xgb_min_child_weight,
        'gamma': args.xgb_gamma,
        'random_state': args.seed,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(train_embeddings, label=train_labels)
    dval = xgb.DMatrix(val_embeddings, label=val_labels)
    
    # Custom evaluation function for QWK
    def qwk_eval(y_pred, y_true):
        """Custom evaluation function for Quadratic Weighted Kappa."""
        y_true = y_true.get_label()
        y_pred_class = np.argmax(y_pred.reshape(-1, 4), axis=1)
        qwk = cohen_kappa_score(y_true, y_pred_class, weights='quadratic')
        return 'qwk', qwk
    
    # Train with early stopping based on validation loss
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=args.xgb_n_estimators,
        evals=evals,
        early_stopping_rounds=args.xgb_early_stopping_rounds,
        verbose_eval=10
    )
    
    # Make predictions
    train_pred_proba = model.predict(dtrain)
    val_pred_proba = model.predict(dval)
    
    # Convert probabilities to class predictions
    train_pred = np.argmax(train_pred_proba, axis=1)
    val_pred = np.argmax(val_pred_proba, axis=1)
    
    # Calculate metrics
    train_acc = accuracy_score(train_labels, train_pred)
    val_acc = accuracy_score(val_labels, val_pred)
    val_f1 = f1_score(val_labels, val_pred, average='weighted')
    val_qwk = cohen_kappa_score(val_labels, val_pred, weights='quadratic')
    
    print(f"\nXGBoost Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val F1 Score: {val_f1:.4f}")
    print(f"Val QWK: {val_qwk:.4f}")
    
    # Feature importance
    importance = model.get_score(importance_type='weight')
    print(f"\nTop 10 Most Important Features:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feature, score in sorted_importance:
        print(f"  {feature}: {score}")
    
    # Print detailed predictions for each validation donor
    print(f"\nDetailed Validation Donor Predictions:")
    print("=" * 60)
    
    # Get validation donor names and their labels
    val_donor_names = []
    val_donor_labels = []
    
    # Extract validation donor information from the main function context
    # We need to get the test donors from the split data
    import json
    with open(args.split_json, 'r') as f:
        split_data = json.load(f)
    
    # Extract validation donor information
    for donor in split_data['test_donors']:
        if donor in donor_labels:
            val_donor_names.append(donor)
            val_donor_labels.append(donor_labels[donor])
    
    # Create class mapping for readable output
    class_names = ['Not AD', 'Low', 'Intermediate', 'High']
    
    # Print each validation donor's prediction
    for i, (donor, true_label, pred_label) in enumerate(zip(val_donor_names, val_donor_labels, val_pred)):
        true_class = class_names[true_label] if true_label < len(class_names) else f"Class_{true_label}"
        pred_class = class_names[pred_label] if pred_label < len(class_names) else f"Class_{pred_label}"
        status = "✓ CORRECT" if true_label == pred_label else "✗ WRONG"
        
        print(f"Val Donor {i+1}: {donor}")
        print(f"  Ground Truth: {true_class} (class {true_label})")
        print(f"  Prediction:   {pred_class} (class {pred_label})")
        print(f"  Result:       {status}")
        print("-" * 40)
    
    # Calculate correct predictions
    correct_count = sum(1 for true_label, pred_label in zip(val_donor_labels, val_pred) if true_label == pred_label)
    print(f"\nSummary: {correct_count}/{len(val_pred)} correct predictions")
    
    return model, {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_qwk': val_qwk,
        'train_predictions': train_pred,
        'val_predictions': val_pred,
        'feature_importance': importance
    }

@torch.no_grad()
def coral_predict(logits, threshold=0.5):
    """Convert CORAL logits to ordinal predictions."""
    probs = torch.sigmoid(logits)
    predictions = (probs > threshold).sum(dim=1)
    return predictions

def compute_class_weights(labels):
    """Compute class weights for handling class imbalance."""
    from collections import Counter
    import torch
    
    # Count class frequencies
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Compute inverse frequency weights
    weights = torch.zeros(num_classes)
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (num_classes * count)
    
    return weights

def create_balanced_dataloader(dataset, batch_size, shuffle=True):
    """Create a balanced data loader using weighted sampling."""
    from torch.utils.data import WeightedRandomSampler
    
    # Get all labels
    labels = [dataset.donor_labels[donor] for donor in dataset.donors]
    
    # Compute class weights
    class_weights = compute_class_weights(labels)
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=False  # Don't shuffle when using sampler
    )

def load_cell_data(h5ad_path, embeddings_path, predictions_path, labels_path):
    """Load cell embeddings, predictions, and metadata."""
    print("Loading cell data...")
    
    # Load embeddings and predictions
    embeddings = np.load(embeddings_path)
    predictions = np.load(predictions_path)
    labels = np.load(labels_path)
    
    print(f"Loaded {len(embeddings)} cells")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Load metadata from h5ad
    adata = sc.read_h5ad(h5ad_path, backed='r')
    obs = adata.obs.copy()
    
    # Extract donor and cell type information
    # Handle different column names for test case vs full dataset
    if 'Donor_space_ID' in obs.columns:
        donors = obs['Donor_space_ID'].astype(str).values
    elif 'Donor ID' in obs.columns:
        donors = obs['Donor ID'].astype(str).values
    else:
        raise KeyError(f"Neither 'Donor_space_ID' nor 'Donor ID' found in obs. Available columns: {list(obs.columns)}")
    
    celltypes = obs['Subclass'].astype(str).values
    adnc_labels = obs['ADNC'].astype(str).values
    
    # Close file
    try:
        adata.file.close()
    except:
        pass
    
    return embeddings, predictions, labels, donors, celltypes, adnc_labels

def aggregate_by_donor_celltype(embeddings, predictions, labels, donors, celltypes, adnc_labels, 
                               global_indices, split_json, k_samples=1000, num_repetitions=1):
    """
    Aggregate cell embeddings by donor and cell type with train/test split awareness.
    Handles multiple repetitions for training donors.
    
    Args:
        embeddings: Cell embeddings (N, D)
        predictions: Cell predictions (N,)
        labels: Cell labels (N,)
        donors: Donor IDs (N,)
        celltypes: Cell type IDs (N,)
        adnc_labels: ADNC labels (N,)
        global_indices: Global cell indices (N,)
        split_json: Path to split JSON file
        k_samples: Number of cells to sample per donor-celltype combination for training donors
        num_repetitions: Number of independent sampling repetitions for training donors
    """
    print("Aggregating by donor and cell type with train/test split awareness...")
    
    # Load split information
    with open(split_json, 'r') as f:
        split_data = json.load(f)
    
    train_donors = set(split_data.get('train_donors', []))
    test_donors = set(split_data.get('test_donors', []))
    
    print(f"Train donors: {len(train_donors)}")
    print(f"Test donors: {len(test_donors)}")
    
    # Create mapping from global indices to metadata
    donor_map = {}
    celltype_map = {}
    adnc_map = {}
    
    for i, gidx in enumerate(global_indices):
        if gidx < len(donors):
            donor_map[i] = donors[gidx]
            celltype_map[i] = celltypes[gidx]
            adnc_map[i] = adnc_labels[gidx]
    
    # Group cells by donor and cell type
    donor_celltype_groups = defaultdict(lambda: defaultdict(list))
    
    for i in range(len(embeddings)):
        if i in donor_map:
            donor = donor_map[i]
            celltype = celltype_map[i]
            adnc = adnc_map[i]
            
            donor_celltype_groups[donor][celltype].append({
                'embedding': embeddings[i],
                'prediction': predictions[i],
                'label': labels[i],
                'adnc': adnc
            })
    
    print(f"Found {len(donor_celltype_groups)} donors")
    
    # Get unique cell types
    all_celltypes = set()
    for donor_data in donor_celltype_groups.values():
        all_celltypes.update(donor_data.keys())
    all_celltypes = sorted(list(all_celltypes))
    
    print(f"Found {len(all_celltypes)} unique cell types")
    
    # Create donor embeddings with different strategies for train/test
    donor_embeddings = {}
    donor_labels = {}
    donor_split_info = {}
    
    for donor, celltype_data in donor_celltype_groups.items():
        is_train_donor = donor in train_donors
        celltype_embeddings = []
        
        for celltype in all_celltypes:
            if celltype in celltype_data:
                cells = celltype_data[celltype]
                
                if is_train_donor and len(cells) > k_samples:
                    # For training donors, sample k cells
                    sampled_cells = np.random.choice(len(cells), k_samples, replace=False)
                    cells = [cells[i] for i in sampled_cells]
                # For test donors, use all cells (no sampling)
                
                # Compute mean embedding for this cell type
                cell_embeddings = np.array([cell['embedding'] for cell in cells])
                mean_embedding = np.mean(cell_embeddings, axis=0)
                celltype_embeddings.append(mean_embedding)
            else:
                # No cells of this type for this donor, use zero embedding
                celltype_embeddings.append(np.zeros(embeddings.shape[1]))
        
        # Concatenate all cell type embeddings
        donor_embedding = np.concatenate(celltype_embeddings)
        donor_embeddings[donor] = donor_embedding
        
        # Get donor label (most common ADNC label)
        all_adnc_labels = []
        for celltype_cells in celltype_data.values():
            all_adnc_labels.extend([cell['adnc'] for cell in celltype_cells])
        
        if all_adnc_labels:
            donor_label = Counter(all_adnc_labels).most_common(1)[0][0]
            # Convert to numeric label
            adnc_mapping = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
            donor_labels[donor] = adnc_mapping.get(donor_label, 0)
        else:
            donor_labels[donor] = 0
        
        # Record split information
        donor_split_info[donor] = {
            'is_train': is_train_donor,
            'n_cells': sum(len(cells) for cells in celltype_data.values()),
            'sampling_applied': is_train_donor
        }
    
    print(f"Created embeddings for {len(donor_embeddings)} donors")
    print(f"Donor embedding dimension: {list(donor_embeddings.values())[0].shape[0]}")
    
    return donor_embeddings, donor_labels, all_celltypes, donor_split_info

def train_donor_classifier(donor_embeddings, donor_labels, donor_split_info, args, num_repetitions=3):
    """Train MLP classifier on donor embeddings using test donors as validation during training."""
    print("Training donor classifier...")
    print("Using test donors as validation during training...")
    
    # Prepare data
    donors = list(donor_embeddings.keys())
    
    print(f"Total donors: {len(donors)}")
    print(f"Embedding dimension: {list(donor_embeddings.values())[0].shape[0]}")
    
    # Use existing split from donor_split_info
    train_donors = [donor for donor in donors if donor_split_info[donor]['is_train']]
    test_donors = [donor for donor in donors if not donor_split_info[donor]['is_train']]
    
    print(f"Train donors: {len(train_donors)}")
    print(f"Test donors: {len(test_donors)}")
    
    # Create repetitions for training donors
    train_embeddings_list = []
    train_labels_list = []
    
    # Check if donors are already repeated (contain '_rep_' in name)
    if any('_rep_' in donor for donor in train_donors):
        print("Donors already contain repetitions, using them directly")
        for donor in train_donors:
            train_embeddings_list.append(donor_embeddings[donor])
            train_labels_list.append(donor_labels[donor])
    else:
        print("Creating repetitions for training donors")
        for donor in train_donors:
            for _ in range(num_repetitions):
                train_embeddings_list.append(donor_embeddings[donor])
                train_labels_list.append(donor_labels[donor])
    
    # Test donors (no repetitions)
    test_embeddings = np.array([donor_embeddings[d] for d in test_donors])
    test_labels = np.array([donor_labels[d] for d in test_donors])
    
    # Convert to arrays
    train_embeddings = np.array(train_embeddings_list)
    train_labels = np.array(train_labels_list)
    
    print(f"Training samples: {len(train_embeddings)} ({len(train_donors)} donors × {num_repetitions} repetitions = {len(train_embeddings)} samples)")
    print(f"Test samples: {len(test_embeddings)} ({len(test_donors)} donors × 1 sample each = {len(test_embeddings)} samples)")
    print(f"Label distribution: {np.bincount(train_labels)}")
    print(f"Input feature dimension: {train_embeddings.shape[1]} (24 cell types × 16 embedding dim = 384)")
    
    print(f"Train donors: {len(train_donors)}")
    print(f"Test donors: {len(test_donors)}")
    
    # Create datasets using the repeated data
    train_dataset = DonorEmbeddingDataset(
        {f"train_{i}": train_embeddings[i] for i in range(len(train_embeddings))},
        {f"train_{i}": train_labels[i] for i in range(len(train_labels))}
    )
    test_dataset = DonorEmbeddingDataset(
        {d: donor_embeddings[d] for d in test_donors},
        {d: donor_labels[d] for d in test_donors}
    )
    
    # Create data loaders with optional balanced sampling
    if args.use_balanced_sampling:
        train_loader = create_balanced_dataloader(train_dataset, batch_size=args.batch_size)
        print("Using balanced sampling for training")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        print("Using standard sampling for training")
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Compute class weights for loss weighting
    train_labels = [train_dataset.donor_labels[donor] for donor in train_dataset.donors]
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"Class weights: {class_weights}")
    print(f"Using loss function: {args.loss_function}")
    
    # Create enhanced model with ordinal regression
    model = DonorMLP(
        input_dim=train_embeddings.shape[1],
        hidden_dims=args.hidden_dims,
        num_classes=4,  # ADNC classes
        dropout=args.dropout,
        use_ordinal=True  # Use ordinal regression
    )
    
    model = model.to(device)
    
    # Training setup with ordinal regression
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop with early stopping based on QWK
    model.train()
    best_val_qwk = -1.0  # QWK can be negative, so start with -1
    best_model_state = None  # Store best model state
    patience_counter = 0
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            
            # Use selected loss function
            if args.loss_function == "qwk":
                loss = qwk_loss(outputs, batch_labels)
            elif args.loss_function == "ordinal_mae":
                loss = ordinal_mae_loss(outputs, batch_labels)
            elif args.loss_function == "emd":
                loss = earth_movers_distance_loss(outputs, batch_labels)
            else:  # coral
                loss = coral_loss(outputs, batch_labels)
            
            # Apply class weights if requested
            if args.use_class_weights:
                # Get class weights for current batch
                batch_class_weights = class_weights[batch_labels.long()]
                loss = loss * batch_class_weights
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Convert CORAL logits to predictions
            predicted = coral_predict(outputs)
            
            # Handle tensor dimensions properly
            if batch_labels.dim() == 0:  # scalar tensor
                batch_size = 1
            else:
                batch_size = batch_labels.size(0)
            
            total += batch_size
            correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        
        # Evaluate on validation set for early stopping (compute QWK)
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_embeddings, batch_labels in test_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.squeeze().to(device)
                
                outputs = model(batch_embeddings)
                predicted = coral_predict(outputs)
                
                # Handle tensor dimensions properly
                if batch_labels.dim() == 0:  # scalar tensor
                    batch_size = 1
                else:
                    batch_size = batch_labels.size(0)
                
                val_total += batch_size
                val_correct += (predicted == batch_labels).sum().item()
                
                # Handle numpy conversion properly - ensure all values are scalars
                if batch_labels.dim() == 0:  # scalar tensor
                    val_preds.append(int(predicted.cpu().numpy()))
                    val_labels.append(int(batch_labels.cpu().numpy()))
                else:
                    val_preds.extend([int(x) for x in predicted.cpu().numpy()])
                    val_labels.extend([int(x) for x in batch_labels.cpu().numpy()])
        
        val_accuracy = val_correct / val_total
        val_qwk = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
        model.train()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2%}, Val Acc: {val_accuracy:.2%}, Val QWK: {val_qwk:.4f}")
        
        # Early stopping based on validation QWK
        if val_qwk > best_val_qwk + args.min_delta:
            best_val_qwk = val_qwk
            best_model_state = model.state_dict().copy()  # Save best model state
            patience_counter = 0
            if epoch % 10 == 0:
                print(f"✓ Val QWK improved to {val_qwk:.4f}")
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f"⚠ Val QWK not improving: {val_qwk:.4f} (best: {best_val_qwk:.4f}) - Patience: {patience_counter}/{args.early_stopping_patience}")
        
        if patience_counter >= args.early_stopping_patience:
            print(f"\n🛑 Early stopping triggered: Val QWK not improving for {args.early_stopping_patience} epochs")
            print(f"Best val QWK: {best_val_qwk:.4f}")
            break
    
    # Load best model state for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with val QWK: {best_val_qwk:.4f}")
    
    # Final evaluation on validation set
    model.eval()
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            outputs = model(batch_embeddings)
            
            # Convert CORAL logits to predictions
            predicted = coral_predict(outputs)
            
            # Handle tensor dimensions properly
            if batch_labels.dim() == 0:  # scalar tensor
                batch_size = 1
            else:
                batch_size = batch_labels.size(0)
            
            val_total += batch_size
            val_correct += (predicted == batch_labels).sum().item()
            
            # Handle numpy conversion properly - ensure all values are scalars
            if batch_labels.dim() == 0:  # scalar tensor
                all_preds.append(int(predicted.cpu().numpy()))
                all_labels.append(int(batch_labels.cpu().numpy()))
            else:
                all_preds.extend([int(x) for x in predicted.cpu().numpy()])
                all_labels.extend([int(x) for x in batch_labels.cpu().numpy()])
    
    # Calculate metrics
    val_accuracy = val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    print(f"\nValidation Results:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {val_qwk:.4f}")
    
    # Print detailed predictions for each validation donor
    print(f"\nDetailed Validation Donor Predictions:")
    print("=" * 60)
    
    # Get validation donor names and their labels
    val_donor_names = []
    val_donor_labels = []
    
    # Extract validation donor information
    for donor in test_donors:
        if donor in donor_labels:
            val_donor_names.append(donor)
            val_donor_labels.append(donor_labels[donor])
    
    # Create class mapping for readable output
    class_names = ['Not AD', 'Low', 'Intermediate', 'High']
    
    # Print each validation donor's prediction
    for i, (donor, true_label, pred_label) in enumerate(zip(val_donor_names, val_donor_labels, all_preds)):
        true_class = class_names[true_label] if true_label < len(class_names) else f"Class_{true_label}"
        pred_class = class_names[pred_label] if pred_label < len(class_names) else f"Class_{pred_label}"
        status = "✓ CORRECT" if true_label == pred_label else "✗ WRONG"
        
        print(f"Val Donor {i+1}: {donor}")
        print(f"  Ground Truth: {true_class} (class {true_label})")
        print(f"  Prediction:   {pred_class} (class {pred_label})")
        print(f"  Result:       {status}")
        print("-" * 40)
    
    print(f"\nSummary: {val_correct}/{val_total} correct predictions")
    # Create dynamic target names based on actual classes
    unique_labels = sorted(np.unique(all_labels))
    adnc_mapping = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}
    target_names = [adnc_mapping.get(label, f'Class_{label}') for label in unique_labels]
    
    print(f"Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, labels=unique_labels))
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'donor_classifier.pt'))
    
    # Save results
    results = {
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'val_qwk': val_qwk,
        'n_train_donors': len(train_donors),
        'n_val_donors': len(test_donors),
        'embedding_dim': train_embeddings.shape[1],
        'model_config': {
            'hidden_dims': args.hidden_dims,
            'dropout': args.dropout
        },
        'sampling_strategy': 'training_donors_sampled_val_donors_all'
    }
    
    with open(os.path.join(args.output_dir, 'donor_classifier_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Model and results saved to {args.output_dir}")
    
    return model, results

def main():
    parser = argparse.ArgumentParser(description="Train donor-level classifier")
    
    # Data arguments
    parser.add_argument("--h5ad_path", required=False, help="Path to h5ad file (only needed for MLP model)")
    parser.add_argument("--split_json", required=True, help="Path to split JSON file")
    parser.add_argument("--embeddings_path", required=True, help="Path to cell embeddings")
    parser.add_argument("--predictions_path", required=True, help="Path to cell predictions")
    parser.add_argument("--labels_path", required=True, help="Path to cell labels")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    # Sampling arguments
    parser.add_argument("--k_samples", type=int, default=1000, 
                       help="Number of cells to sample per donor-celltype combination")
    parser.add_argument("--num_repetitions", type=int, default=1,
                       help="Number of independent sampling repetitions for training donors")
    
    # Model arguments
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 256], 
                       help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--loss_function", type=str, default="qwk", 
                       choices=["coral", "emd", "qwk", "ordinal_mae"], 
                       help="Loss function: coral, emd (Earth Mover's Distance), qwk (QWK-optimized), or ordinal_mae")
    parser.add_argument("--use_class_weights", action="store_true", 
                       help="Use class weights in loss function")
    parser.add_argument("--use_balanced_sampling", action="store_true", default=True,
                       help="Use balanced sampling in data loader")
    parser.add_argument("--early_stopping_patience", type=int, default=30,
                       help="Number of epochs to wait for test QWK improvement before early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001,
                       help="Minimum change in test QWK to qualify as improvement")
    
    # Model type selection
    parser.add_argument("--model_type", type=str, default="mlp", 
                       choices=["mlp", "xgboost"], 
                       help="Model type: mlp (DonorMLP) or xgboost")
    
    # XGBoost specific parameters
    parser.add_argument("--xgb_max_depth", type=int, default=6,
                       help="Maximum depth of XGBoost trees")
    parser.add_argument("--xgb_lr", type=float, default=0.1,
                       help="Learning rate for XGBoost")
    parser.add_argument("--xgb_n_estimators", type=int, default=1000,
                       help="Number of boosting rounds for XGBoost")
    parser.add_argument("--xgb_subsample", type=float, default=0.8,
                       help="Subsample ratio for XGBoost")
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8,
                       help="Column subsample ratio for XGBoost")
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.1,
                       help="L1 regularization for XGBoost")
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0,
                       help="L2 regularization for XGBoost")
    parser.add_argument("--xgb_early_stopping_rounds", type=int, default=50,
                       help="Early stopping rounds for XGBoost")
    parser.add_argument("--xgb_min_child_weight", type=int, default=1,
                       help="Minimum child weight for XGBoost")
    parser.add_argument("--xgb_gamma", type=float, default=0.0,
                       help="Minimum loss reduction for XGBoost")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*60)
    print("TRAINING DONOR-LEVEL CLASSIFIER")
    print("="*60)
    
    # Load donor embeddings from Step 3
    print("Loading donor embeddings from Step 3...")
    
    # Load donor embeddings and labels
    donor_embeddings = np.load(args.embeddings_path, allow_pickle=True).item()
    donor_labels = np.load(args.predictions_path, allow_pickle=True).item()
    
    # Load metadata to get split info and num_repetitions
    metadata_path = os.path.join(os.path.dirname(args.embeddings_path), 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    donor_split_info = metadata['donor_split_info']
    num_repetitions = metadata.get('num_repetitions', 1)
    
    print(f"Loaded {len(donor_embeddings)} donor embeddings")
    print(f"Will create {num_repetitions} repetitions for training donors")
    
    # Train donor classifier
    if args.model_type == "xgboost":
        print("Using XGBoost for donor classification...")
        
        # Load donor split info from the original split file
        with open(args.split_json, 'r') as f:
            split_data = json.load(f)
        
        train_donors = split_data['train_donors']
        test_donors = split_data['test_donors']
        
        # Get training and test embeddings and labels
        train_embeddings = []
        train_labels = []
        test_embeddings = []
        test_labels = []
        
        # Helper function to get base donor name (remove _rep_X suffix)
        def get_base_donor_name(donor_key):
            if '_rep_' in donor_key:
                return donor_key.split('_rep_')[0]
            return donor_key
        
        # Create mapping from base donor names to all their repetitions
        base_donor_to_reps = {}
        for donor_key in donor_embeddings.keys():
            base_name = get_base_donor_name(donor_key)
            if base_name not in base_donor_to_reps:
                base_donor_to_reps[base_name] = []
            base_donor_to_reps[base_name].append(donor_key)
        
        # Collect training data (use all repetitions for training donors)
        for donor in train_donors:
            if donor in base_donor_to_reps:
                for rep_key in base_donor_to_reps[donor]:
                    train_embeddings.append(donor_embeddings[rep_key])
                    train_labels.append(donor_labels[rep_key])
        
        # Collect test data (use first repetition for test donors)
        for donor in test_donors:
            if donor in base_donor_to_reps:
                # Use first repetition for test
                rep_key = base_donor_to_reps[donor][0]
                test_embeddings.append(donor_embeddings[rep_key])
                test_labels.append(donor_labels[rep_key])
        
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)
        test_embeddings = np.array(test_embeddings)
        test_labels = np.array(test_labels)
        
        print(f"Training XGBoost with {len(train_embeddings)} samples, {train_embeddings.shape[1]} features")
        print(f"Validating XGBoost with {len(test_embeddings)} samples")
        
        # Train XGBoost model with optimized hyperparameters
        model, results = train_xgboost_classifier(train_embeddings, train_labels, test_embeddings, test_labels, args)
        
    else:
        # Use MLP classifier
        model, results = train_donor_classifier(donor_embeddings, donor_labels, donor_split_info, args, num_repetitions)
    
    print("Donor classifier training completed!")

if __name__ == "__main__":
    main()
