#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速分析CellFM预测结果
"""

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter

# 读取训练结果文件
results_path = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/training_results.json'
test_h5ad = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/test_subset.h5ad'

print("="*80)
print("CellFM MTG 预测结果分析")
print("="*80)

# 1. 读取总体结果
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    print("\n整体性能指标:")
    print(f"  最佳测试准确率: {results['best_test_acc']:.4f}")
    print(f"  训练轮数: {results['epochs']}")
    print(f"  批次大小: {results['batch_size']}")
    print(f"  学习率: {results['lr']}")
    print(f"  基因数量: {results['n_genes']}")

# 2. 读取数据查看真实标签分布
print("\n" + "="*80)
print("真实标签分布 (测试集)")
print("="*80)

adata = sc.read_h5ad(test_h5ad)
print(f"\n测试集总细胞数: {len(adata)}")

if 'ADNC' in adata.obs.columns:
    # 分析真实ADNC分布
    if pd.api.types.is_numeric_dtype(adata.obs['ADNC']):
        # 数字标签
        adnc_numeric = adata.obs['ADNC']
        num_to_name = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}
        adnc_labels = [num_to_name.get(int(x), 'Unknown') for x in adnc_numeric]
    else:
        # 字符串标签，需要转换
        adnc_labels = adata.obs['ADNC'].astype(str).values
        # 统一格式
        label_map = {
            '0': 'Not AD', '1': 'Low', '2': 'Intermediate', '3': 'High',
            'Not AD': 'Not AD', 'Low': 'Low', 'Intermediate': 'Intermediate', 'High': 'High'
        }
        adnc_labels = [label_map.get(str(x).strip(), 'Unknown') for x in adnc_labels]
    
    true_counts = Counter(adnc_labels)
    print("\n各ADNC级别的细胞数量:")
    for label in ['Not AD', 'Low', 'Intermediate', 'High']:
        count = true_counts.get(label, 0)
        percentage = count / len(adata) * 100 if len(adata) > 0 else 0
        print(f"  {label:15s}: {count:7d} ({percentage:5.2f}%)")

# 3. 读取训练日志中的关键信息
print("\n" + "="*80)
print("训练过程详情")
print("="*80)

if os.path.exists('/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/training.log'):
    with open('/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/training.log', 'r') as f:
        log_content = f.read()
        
    # 提取准确率信息
    import re
    acc_matches = re.findall(r'acc=(\d+\.\d+)', log_content)
    if acc_matches:
        print(f"\n发现的准确率值: {len(acc_matches)} 个")
        print(f"  最后的准确率值: {acc_matches[-1]}")
    
    # 检查是否有测试准确率信息
    test_acc_match = re.search(r'Test accuracy: ([\d.]+)', log_content)
    if test_acc_match:
        test_acc = float(test_acc_match.group(1))
        f1_match = re.search(r'Test F1: ([\d.]+)', log_content)
        test_f1 = float(f1_match.group(1)) if f1_match else 0
        
        print(f"\n最终测试结果:")
        print(f"  测试准确率: {test_acc:.4f}")
        print(f"  测试F1分数: {test_f1:.4f}")

# 4. 模型文件信息
print("\n" + "="*80)
print("模型文件")
print("="*80)

model_path = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/best_model.pth'
if os.path.exists(model_path):
    import os
    size = os.path.getsize(model_path) / (1024*1024)
    print(f"\n最佳模型文件: {model_path}")
    print(f"  文件大小: {size:.2f} MB")
    print(f"  状态: ✓ 存在")

# 5. 保存的文件列表
print("\n" + "="*80)
print("输出文件总结")
print("="*80)

output_dir = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg'
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    print(f"\n输出目录: {output_dir}")
    print(f"总文件数: {len(files)}")
    print("\n主要文件:")
    important_files = ['best_model.pth', 'training_results.json', 'training.log', 
                       'train_subset.h5ad', 'test_subset.h5ad']
    for fname in important_files:
        filepath = os.path.join(output_dir, fname)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_str = f"{size/(1024*1024):.2f}MB" if size > 1024*1024 else f"{size/1024:.2f}KB"
            print(f"  ✓ {fname:30s} ({size_str})")
        else:
            print(f"  ✗ {fname:30s} (不存在)")

print("\n" + "="*80)
print("分析完成")
print("="*80)
print("\n提示: 如需查看每个预测的详细对比，请使用 analyze_cellfm_results.py 脚本")
print("命令: python /home/ubuntu/LLM-inference/xinze-project/dream_test/scripts/analyze_cellfm_results.py")



