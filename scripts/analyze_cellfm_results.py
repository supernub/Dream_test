#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析CellFM MTG模型的预测结果，生成详细的性能分析报告
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import sys

# Add cellfm to path
sys.path.insert(0, '/home/ubuntu/LLM-inference/xinze-project/cellfm')

from model import Finetune_Cell_FM

ADNC_CLASSES = ['Not AD', 'Low', 'Intermediate', 'High']
ADNC_LABELS = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}


def load_model(model_path, device='cuda:0'):
    """加载训练好的CellFM模型"""
    print(f"Loading model from {model_path}...")
    
    # 读取基因词汇信息
    test_h5ad = "/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/test_subset.h5ad"
    adata = sc.read_h5ad(test_h5ad)
    n_genes = adata.n_vars
    print(f"Dataset has {n_genes} genes")
    
    # 使用与训练时相同的配置
    from layers.utils import Config_80M
    cfg = Config_80M()
    cfg.num_cls = 4
    cfg.ckpt_path = None  # 不需要加载预训练权重
    cfg.device = device
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    
    # 创建模型 - 需要修改extractor使用正确的基因数量
    model = Finetune_Cell_FM(cfg).to(device)
    
    # 重新初始化基因embedding以匹配我们的vocabulary
    print(f"Re-initializing gene embedding for {n_genes} genes...")
    with torch.no_grad():
        emb_dim = model.extractor.net.gene_emb.shape[1]
        pad_to = ((n_genes - 1) // 8 + 1) * 8
        model.extractor.net.gene_emb = nn.Parameter(
            torch.empty(pad_to, emb_dim)
        )
        nn.init.xavier_normal_(model.extractor.net.gene_emb)
        model.extractor.net.gene_emb.data[0, :] = 0
        print(f"Initialized new gene embedding: {model.extractor.net.gene_emb.shape}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters())}")
    return model


def predict_on_data(model, data_path, device='cuda:0', batch_size=16):
    """在数据上进行预测"""
    print(f"\nLoading data from {data_path}...")
    adata = sc.read_h5ad(data_path)
    
    # 使用CellFM的数据加载器
    from layers.utils import SCrna
    
    # 确保必要的列存在
    if 'feat' not in adata.obs.columns:
        print("WARNING: 'feat' column not found. Trying to create it...")
        # 尝试从ADNC列创建feat
        adnc_col = 'ADNC' if 'ADNC' in adata.obs.columns else None
        if adnc_col:
            # 归一化ADNC标签
            adnc_series = adata.obs[adnc_col]
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
    
    # 添加train列
    if 'train' not in adata.obs.columns:
        adata.obs['train'] = 2  # 2表示测试集
    
    # 创建数据集
    dataset = SCrna(adata, mode='eval')
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_preds = []
    all_true = []
    all_probs = []
    
    print("Running predictions...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 100 == 0:
                print(f"  Processing batch {i}/{len(dataloader)}")
            
            x = batch['genes'].to(device)
            true_labels = batch['label'].cpu().numpy()
            
            # 前向传播
            output = model(x)
            preds = output.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(output, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_true.extend(true_labels)
            all_probs.extend(probs)
    
    return np.array(all_preds), np.array(all_true), np.array(all_probs)


def analyze_results(y_true, y_pred, y_probs, output_dir):
    """详细分析预测结果"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 基本统计
    print("\n" + "="*80)
    print("Prediction Results Analysis")
    print("="*80)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    
    # 2. 分类报告
    print("\n" + "="*80)
    print("Detailed Classification Report")
    print("="*80)
    print("\n" + classification_report(y_true, y_pred, 
                                       target_names=ADNC_CLASSES,
                                       labels=[0, 1, 2, 3]))
    
    # 3. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    print("\n" + "="*80)
    print("Confusion Matrix")
    print("="*80)
    print("\nActual -> Predicted")
    print(f"{'':>15}", end='')
    for label in ADNC_CLASSES:
        print(f"{label:>15}", end='')
    print()
    for i, true_label in enumerate(ADNC_CLASSES):
        print(f"{true_label:>15}", end='')
        for j in range(4):
            print(f"{cm[i, j]:>15}", end='')
        print()
    
    # 保存混淆矩阵图片
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ADNC_CLASSES, yticklabels=ADNC_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CellFM MTG Predictions - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"\nConfusion matrix saved to {output_dir}/confusion_matrix.png")
    
    # 4. 按类别分析
    print("\n" + "="*80)
    print("Per-Class Analysis")
    print("="*80)
    
    results_by_class = {}
    for class_idx, class_name in enumerate(ADNC_CLASSES):
        mask = y_true == class_idx
        if mask.sum() > 0:
            class_true = y_true[mask]
            class_pred = y_pred[mask]
            class_acc = accuracy_score(class_true, class_pred)
            
            results_by_class[class_name] = {
                'count': int(mask.sum()),
                'correct': int((class_true == class_pred).sum()),
                'accuracy': class_acc,
                'most_confused_with': Counter(class_pred).most_common(1)[0]
            }
            
            print(f"\n{class_name}:")
            print(f"  Total samples: {mask.sum()}")
            print(f"  Correct predictions: {(class_true == class_pred).sum()}")
            print(f"  Accuracy: {class_acc:.4f}")
            print(f"  Most confused with: {ADNC_CLASSES[Counter(class_pred).most_common(1)[0][0]]}")
    
    # 5. 保存详细结果到CSV
    df_results = pd.DataFrame({
        'predicted': [ADNC_CLASSES[p] for p in y_pred],
        'actual': [ADNC_CLASSES[t] for t in y_true],
        'predicted_idx': y_pred,
        'actual_idx': y_true,
        'correct': (y_pred == y_true),
        'confidence_notad': y_probs[:, 0],
        'confidence_low': y_probs[:, 1],
        'confidence_intermediate': y_probs[:, 2],
        'confidence_high': y_probs[:, 3]
    })
    
    csv_path = os.path.join(output_dir, 'detailed_predictions.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nDetailed predictions saved to {csv_path}")
    
    # 6. 置信度分析
    print("\n" + "="*80)
    print("Confidence Analysis")
    print("="*80)
    
    correct_preds = df_results[df_results['correct'] == True]
    incorrect_preds = df_results[df_results['correct'] == False]
    
    print(f"\nCorrect predictions: {len(correct_preds)}")
    if len(correct_preds) > 0:
        print(f"  Average confidence for correct: {correct_preds[[col for col in df_results.columns if 'confidence' in col]].max(axis=1).mean():.4f}")
    
    print(f"\nIncorrect predictions: {len(incorrect_preds)}")
    if len(incorrect_preds) > 0:
        print(f"  Average confidence for incorrect: {incorrect_preds[[col for col in df_results.columns if 'confidence' in col]].max(axis=1).mean():.4f}")
    
    # 7. 保存JSON结果
    summary = {
        'overall_accuracy': float(accuracy),
        'weighted_f1': float(f1),
        'total_samples': len(y_true),
        'per_class_results': results_by_class,
        'confusion_matrix': cm.tolist(),
        'class_distribution': {
            ADNC_CLASSES[i]: int((y_true == i).sum()) 
            for i in range(4)
        }
    }
    
    json_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {json_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze CellFM prediction results')
    parser.add_argument('--test_h5ad', default='/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/test_subset.h5ad')
    parser.add_argument('--model_path', default='/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/best_model.pth')
    parser.add_argument('--output_dir', default='/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/analysis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    print("="*80)
    print("CellFM MTG Model - Prediction Analysis")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Test data: {args.test_h5ad}")
    print(f"Device: {args.device}")
    
    # 切换到cellfm目录以使用相对路径
    import os
    original_dir = os.getcwd()
    cellfm_dir = '/home/ubuntu/LLM-inference/xinze-project/cellfm'
    os.chdir(cellfm_dir)
    
    try:
        # 加载模型
        model = load_model(args.model_path, device=args.device)
        
        # 进行预测
        y_pred, y_true, y_probs = predict_on_data(model, args.test_h5ad, device=args.device, batch_size=args.batch_size)
    finally:
        os.chdir(original_dir)
    
    # 分析结果
    summary = analyze_results(y_true, y_pred, y_probs, args.output_dir)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

