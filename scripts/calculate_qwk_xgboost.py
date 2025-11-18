#!/usr/bin/env python3
"""
计算 XGBoost 模型的 QWK (Quadratic Weighted Kappa) 指标
"""

import argparse
import json
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score

# 导入数据加载函数
import sys
sys.path.append('/home/ubuntu/LLM-inference/xinze-project/dream_test')
from binary_classifier.data import load_binary_dataset, DatasetLoadError


def quadratic_weighted_kappa(y_true, y_pred):
    """
    计算 Quadratic Weighted Kappa (QWK)
    
    QWK 是 Cohen's Kappa 的加权版本，使用二次权重矩阵。
    对于二分类，QWK 和 Cohen's Kappa 相同。
    """
    # 对于二分类，quadratic weights 和 linear weights 相同
    # 但使用 'quadratic' 更符合 QWK 的定义
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def main():
    parser = argparse.ArgumentParser(description="计算 XGBoost 模型的 QWK 指标")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="数据集目录",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="XGBoost 模型路径 (model.json)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="保存 QWK 结果的 JSON 文件路径",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="binary_label",
        help="标签列名称",
    )
    parser.add_argument(
        "--concat-donor-metadata",
        action="store_true",
        help="是否使用了 donor metadata",
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print("Loading dataset...")
    try:
        dataset = load_binary_dataset(
            args.dataset_dir,
            label_column=args.label_column,
            prefer_h5ad=True,
            concat_donor_metadata=args.concat_donor_metadata,
        )
    except DatasetLoadError as exc:
        raise SystemExit(f"数据加载失败: {exc}") from exc
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    bst = xgb.Booster()
    bst.load_model(str(args.model_path))
    
    # 准备测试数据
    dtest = xgb.DMatrix(
        dataset.X_test,
        label=dataset.y_test,
        feature_names=dataset.feature_names,
    )
    
    # 获取预测
    print("Making predictions...")
    test_prob = bst.predict(dtest)
    test_pred = (test_prob >= 0.5).astype(int)
    
    # 计算 QWK
    print("Calculating QWK...")
    qwk = quadratic_weighted_kappa(dataset.y_test, test_pred)
    
    # 也计算训练集的 QWK
    dtrain = xgb.DMatrix(
        dataset.X_train,
        label=dataset.y_train,
        feature_names=dataset.feature_names,
    )
    train_prob = bst.predict(dtrain)
    train_pred = (train_prob >= 0.5).astype(int)
    train_qwk = quadratic_weighted_kappa(dataset.y_train, train_pred)
    
    # 打印结果
    print("\n" + "="*60)
    print("QWK (Quadratic Weighted Kappa) Results")
    print("="*60)
    print(f"Train QWK: {train_qwk:.6f}")
    print(f"Test QWK:  {qwk:.6f}")
    print("="*60)
    
    # 保存结果
    results = {
        "train_qwk": float(train_qwk),
        "test_qwk": float(qwk),
        "train_predictions": {
            "probabilities": test_prob.tolist() if len(test_prob) < 1000 else "too_large",
            "predictions": test_pred.tolist() if len(test_pred) < 1000 else "too_large",
        },
        "test_predictions": {
            "probabilities": test_prob.tolist() if len(test_prob) < 1000 else "too_large",
            "predictions": test_pred.tolist() if len(test_pred) < 1000 else "too_large",
        },
        "model_path": str(args.model_path),
        "dataset_dir": str(args.dataset_dir),
        "concat_donor_metadata": args.concat_donor_metadata,
    }
    
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_path}")
    else:
        # 默认保存到模型目录
        output_path = args.model_path.parent / "qwk_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

