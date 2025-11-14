# model_ordinal_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple


class GeneTransformerOrdinal(nn.Module):
    """
    Enhanced Encoder: Embedding -> scale by expression -> TransformerEncoder -> attention pooling
    Head: CORAL ordinal head (shared linear + K-1 biases) with class weighting
    """
    def __init__(
        self,
        num_genes: int,
        num_classes: int,
        embedding_dim: int = 128,
        dim_feedforward: Optional[int] = None,
        nhead: int = 8,
        depth: int = 4,
        dropout: float = 0.1,
        pad_idx: int = 0,
        use_attention_pooling: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_classes = num_classes
        self.use_attention_pooling = use_attention_pooling
        self.use_layer_norm = use_layer_norm
        
        # Set dim_feedforward to 4*embedding_dim if not specified
        if dim_feedforward is None:
            dim_feedforward = 4 * embedding_dim

        # Enhanced embedding with uniform initialization
        self.emb = nn.Embedding(num_genes + 1, embedding_dim, padding_idx=pad_idx)
        nn.init.uniform_(self.emb.weight, a=-1.0/num_genes, b=1.0/num_genes)
        with torch.no_grad():
            self.emb.weight.data[pad_idx].fill_(0.0)

        # Enhanced transformer with layer normalization
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        
        # Optional layer normalization after encoder
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)
        else:
            self.layer_norm = None

        # Attention pooling for better representation
        if use_attention_pooling:
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.attention_query = nn.Parameter(torch.randn(1, 1, embedding_dim))
            nn.init.xavier_uniform_(self.attention_query)

        # Enhanced CORAL head with better initialization
        self.shared_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1, bias=False)
        )
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))
        nn.init.xavier_uniform_(self.shared_fc[0].weight)
        nn.init.xavier_uniform_(self.shared_fc[3].weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, vals, attn_mask=None):
        """
        Enhanced forward pass with attention pooling and better regularization
        seq: LongTensor [B, L] with PAD=0, gene ids start from 1
        vals: FloatTensor [B, L] expression values aligned with seq
        attn_mask: BoolTensor [B, L] True for PAD positions to be masked out
        """
        # Enhanced embedding with expression scaling
        x = self.emb(seq)                       # [B, L, D]
        x = x * vals.unsqueeze(-1)              # scale by expression
        
        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=attn_mask)  # attn_mask: True=pad
        
        # Apply layer normalization if enabled
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        # Enhanced pooling: attention-based or mean pooling
        if self.use_attention_pooling and attn_mask is not None:
            # Attention pooling for better representation
            query = self.attention_query.expand(x.size(0), -1, -1)  # [B, 1, D]
            attn_output, _ = self.attention_pool(
                query, x, x, 
                key_padding_mask=attn_mask
            )
            x_pooled = attn_output.squeeze(1)  # [B, D]
        else:
            # Mean pooling over non-pad tokens
            x = x.masked_fill(attn_mask.unsqueeze(-1), 0.0) if attn_mask is not None else x
            if attn_mask is not None:
                valid_count = (~attn_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
                x_pooled = x.sum(dim=1) / valid_count
            else:
                x_pooled = x.mean(dim=1)

        x_pooled = self.dropout(x_pooled)

        # Enhanced CORAL head with better regularization
        base = self.shared_fc(x_pooled)               # [B, 1]
        logits = base + self.biases.view(1, -1)       # [B, K-1]
        return logits  # CORAL logits for P(y > k)

def coral_loss(logits, targets, reduction="mean", class_weights=None):
    """
    Clean CORAL loss for ordinal regression without focal loss complexity
    logits: [B, K-1] (no sigmoid)
    targets: [B] with values in {0,1,...,K-1}
    CORAL loss = sum_k BCEWithLogits(sigmoid(logit_k), 1_{y > k})
    """
    B, Km1 = logits.shape
    # Build binary targets for each threshold: target_k = 1 if y > k else 0
    thresholds = torch.arange(Km1, device=targets.device).view(1, -1)  # [1, K-1]
    bin_targets = (targets.view(-1, 1) > thresholds).float()           # [B, K-1]

    # Compute BCE loss with optional class weighting
    if class_weights is not None:
        # Weight per class index (0..K-1). Map to per-example weights.
        w = class_weights[targets].view(-1, 1)  # [B,1]
        loss = F.binary_cross_entropy_with_logits(logits, bin_targets, weight=w, reduction="none")
    else:
        loss = F.binary_cross_entropy_with_logits(logits, bin_targets, reduction="none")

    # Sum over thresholds
    loss = loss.sum(dim=1)  # [B]
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def compute_class_weights(labels, num_classes, method='balanced'):
    """
    Compute class weights for imbalanced ordinal regression
    """
    from collections import Counter
    import numpy as np
    
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    if method == 'balanced':
        # sklearn-style balanced weights
        weights = np.zeros(num_classes)
        for class_idx in range(num_classes):
            if class_idx in label_counts:
                weights[class_idx] = total_samples / (num_classes * label_counts[class_idx])
            else:
                weights[class_idx] = 1.0
    elif method == 'inverse':
        # Inverse frequency weighting
        weights = np.zeros(num_classes)
        for class_idx in range(num_classes):
            if class_idx in label_counts:
                weights[class_idx] = total_samples / label_counts[class_idx]
            else:
                weights[class_idx] = 1.0
    else:
        weights = np.ones(num_classes)
    
    return torch.FloatTensor(weights)  # [B]

@torch.no_grad()
def coral_predict(logits, threshold=0.5):
    """
    logits: [B, K-1]
    Predict class = count(sigmoid(logit_k) > threshold)
    returns LongTensor [B]
    """
    probs = torch.sigmoid(logits)           # [B, K-1]
    preds = (probs > threshold).sum(dim=1)  # in {0..K-1}
    return preds
