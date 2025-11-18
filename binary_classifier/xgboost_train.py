import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, roc_curve

from .data import DatasetLoadError, load_binary_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 XGBoost 对指定数据集训练二分类模型")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="包含 cells_metadata.csv / cells_subset.h5ad 的目录",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="binary_label",
        help="标签列名称，默认 binary_label",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="保存模型与评估结果的目录（默认 dataset-dir/xgboost_output）",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.05,
        help="XGBoost 学习率（降低以防止过拟合）",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="树的最大深度（降低以防止过拟合）",
    )
    parser.add_argument(
        "--num-round",
        type=int,
        default=100,
        help=" boosting 轮数（降低以防止过拟合）",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=20,
        help="早停轮数；<=0 表示不使用早停",
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="正样本权重（处理类别不平衡）。如果为 None，则自动计算为 负样本数/正样本数",
    )
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=1.0,
        help="最小子节点权重（正则化）",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="样本采样比例（正则化）",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="特征采样比例（正则化）",
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.0,
        help="L1 正则化系数",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 正则化系数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--concat-donor-metadata",
        action="store_true",
        help="是否拼接 donor-level 统计特征到细胞特征",
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="使用验证集优化预测阈值（基于 QWK）",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="验证集比例（用于阈值优化）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        dataset = load_binary_dataset(
            args.dataset_dir,
            label_column=args.label_column,
            prefer_h5ad=True,
            concat_donor_metadata=args.concat_donor_metadata,
        )
    except DatasetLoadError as exc:
        raise SystemExit(f"数据加载失败: {exc}") from exc

    output_dir = args.output_dir or (args.dataset_dir / "xgboost_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    dtrain = xgb.DMatrix(
        dataset.X_train,
        label=dataset.y_train,
        feature_names=dataset.feature_names,
    )
    dtest = xgb.DMatrix(
        dataset.X_test,
        label=dataset.y_test,
        feature_names=dataset.feature_names,
    )

    # 自动计算 scale_pos_weight（如果未指定）
    if args.scale_pos_weight is None:
        neg_count = (dataset.y_train == 0).sum()
        pos_count = (dataset.y_train == 1).sum()
        auto_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"\n自动计算 scale_pos_weight: {auto_scale_pos_weight:.4f}")
        print(f"  负样本数: {neg_count}, 正样本数: {pos_count}")
        scale_pos_weight = auto_scale_pos_weight
    else:
        scale_pos_weight = args.scale_pos_weight

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "scale_pos_weight": scale_pos_weight,
        "seed": args.seed,
        "tree_method": "hist",
    }

    evals = [(dtrain, "train"), (dtest, "test")]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds if args.early_stopping_rounds > 0 else None,
    )

    train_prob = bst.predict(dtrain)
    test_prob = bst.predict(dtest)
    
    # 优化预测阈值
    best_threshold = 0.5
    if args.optimize_threshold:
        # 从训练集中划分验证集
        from sklearn.model_selection import train_test_split
        train_indices = np.arange(len(dataset.y_train))
        train_idx, val_idx = train_test_split(
            train_indices, 
            test_size=args.val_frac, 
            stratify=dataset.y_train, 
            random_state=args.seed
        )
        val_prob = bst.predict(xgb.DMatrix(
            dataset.X_train[val_idx],
            label=dataset.y_train[val_idx],
            feature_names=dataset.feature_names,
        ))
        
        # 在验证集上找最优阈值（基于 QWK）
        best_threshold, best_qwk = _find_best_threshold(
            dataset.y_train[val_idx], val_prob
        )
        print(f"\n阈值优化 (验证集):")
        print(f"  最佳阈值: {best_threshold:.4f}")
        print(f"  最佳 QWK: {best_qwk:.6f}")
    else:
        # 使用验证集（如果没有优化，直接从训练集划分验证集找阈值）
        from sklearn.model_selection import train_test_split
        train_indices = np.arange(len(dataset.y_train))
        train_idx, val_idx = train_test_split(
            train_indices, 
            test_size=args.val_frac, 
            stratify=dataset.y_train, 
            random_state=args.seed
        )
        val_prob = bst.predict(xgb.DMatrix(
            dataset.X_train[val_idx],
            label=dataset.y_train[val_idx],
            feature_names=dataset.feature_names,
        ))
        best_threshold, _ = _find_best_threshold(
            dataset.y_train[val_idx], val_prob
        )
    
    # 使用最佳阈值进行预测
    train_pred = (train_prob >= best_threshold).astype(int)
    test_pred = (test_prob >= best_threshold).astype(int)

    metrics = {
        "train": _collect_metrics(dataset.y_train, train_prob, threshold=best_threshold),
        "test": _collect_metrics(dataset.y_test, test_prob, threshold=best_threshold),
        "threshold": float(best_threshold),
    }
    
    # 计算 QWK (Quadratic Weighted Kappa)
    train_qwk = cohen_kappa_score(dataset.y_train, train_pred, weights='quadratic')
    test_qwk = cohen_kappa_score(dataset.y_test, test_pred, weights='quadratic')
    metrics["train"]["qwk"] = float(train_qwk)
    metrics["test"]["qwk"] = float(test_qwk)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    
    print(f"\nQWK (Quadratic Weighted Kappa) - 阈值 {best_threshold:.4f}:")
    print(f"  Train QWK: {train_qwk:.6f}")
    print(f"  Test QWK:  {test_qwk:.6f}")

    model_path = output_dir / "model.json"
    bst.save_model(model_path)

    print(f"指标已写入 {metrics_path}")
    print(f"模型已保存至 {model_path}")


def _find_best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """在验证集上找最优阈值（基于 QWK）"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_qwk = -1.0
    
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        qwk = cohen_kappa_score(y_true, pred, weights='quadratic')
        if qwk > best_qwk:
            best_qwk = qwk
            best_threshold = threshold
    
    return best_threshold, best_qwk


def _collect_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5):
    pred = (prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    metrics["support"] = int(len(y_true))
    return metrics


if __name__ == "__main__":
    main()


