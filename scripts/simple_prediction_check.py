#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单检查预测分布 - 查看预测结果中各标签的分布
"""

import pandas as pd
import scanpy as sc
import sys
import os

# 读取训练日志，提取关键信息
log_path = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/training.log'

print("="*80)
print("分析训练日志中的预测分布")
print("="*80)

# 尝试从日志中提取预测信息
with open(log_path, 'r') as f:
    content = f.read()

# 提取测试准确率
import re

# 查找测试结果
test_match = re.search(r'Test accuracy: ([\d.]+), Test F1: ([\d.]+)', content)
if test_match:
    test_acc = float(test_match.group(1))
    test_f1 = float(test_match.group(2))
    print(f"\n测试准确率: {test_acc:.4f}")
    print(f"测试F1分数: {test_f1:.4f}")

# 读取测试数据，查看真实标签分布
test_h5ad = '/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/test_subset.h5ad'
print(f"\n读取测试数据: {test_h5ad}")
adata = sc.read_h5ad(test_h5ad)

print(f"测试集总细胞数: {len(adata)}")

# 分析真实标签
if 'ADNC' in adata.obs.columns:
    adnc_col = adata.obs['ADNC']
    
    # 转换为分类标签
    def convert_to_label(val):
        if pd.isna(val):
            return 'Unknown'
        val_str = str(val).strip()
        label_map = {
            '0': 'Not AD',
            '1': 'Low', 
            '2': 'Intermediate',
            '3': 'High',
            'Not AD': 'Not AD',
            'Low': 'Low',
            'Intermediate': 'Intermediate',
            'High': 'High'
        }
        return label_map.get(val_str, 'Unknown')
    
    labels = [convert_to_label(x) for x in adnc_col]
    true_dist = pd.Series(labels).value_counts()
    
    print("\n真实标签分布:")
    for label in ['Not AD', 'Low', 'Intermediate', 'High', 'Unknown']:
        if label in true_dist.index:
            count = true_dist[label]
            pct = count / len(adata) * 100
            print(f"  {label:15s}: {count:7d} ({pct:5.2f}%)")

# 计算如果全部预测High类会得到什么准确率
if 'High' in true_dist.index:
    high_count = true_dist['High']
    all_high_acc = high_count / len(adata)
    print(f"\n{'='*80}")
    print("关键分析:")
    print(f"{'='*80}")
    print(f"\n如果全部预测为'High'类:")
    print(f"  准确率会 = {all_high_acc:.4f} ({all_high_acc*100:.2f}%)")
    print(f"  当前模型准确率 = {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\n两者比较:")
    if abs(test_acc - all_high_acc) < 0.01:
        print(f"  ⚠️  模型准确率 ({test_acc:.4f}) 几乎等于全部预测High的准确率 ({all_high_acc:.4f})")
        print(f"  → 这表明模型很可能在只预测High类，或绝大多数预测是High")
    elif test_acc > all_high_acc:
        print(f"  ✓ 模型准确率 ({test_acc:.4f}) > 全部预测High的准确率 ({all_high_acc:.4f})")
        print(f"  → 模型明显学习了其他类别的模式")
    else:
        print(f"  ✗ 模型准确率 ({test_acc:.4f}) < 全部预测High的准确率 ({all_high_acc:.4f})")
        print(f"  → 模型性能较差，甚至不如简单预测多数类")

print(f"\n{'='*80}")



