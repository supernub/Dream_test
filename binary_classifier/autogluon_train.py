import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
from scipy import sparse

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    raise ImportError("请安装 autogluon: pip install autogluon")

from .data import DatasetLoadError, load_binary_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 AutoGluon 对指定数据集训练二分类模型")
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
        help="保存模型与评估结果的目录（默认 dataset-dir/autogluon_output）",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=3600,
        help="训练时间限制（秒），默认 3600 秒（1小时）",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="high_quality",
        choices=["best_quality", "high_quality", "good_quality", "medium_quality", "optimize_for_deployment"],
        help="AutoGluon 预设模式，默认 high_quality（减少过拟合）",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="roc_auc",
        help="评估指标，默认 roc_auc",
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
        "--holdout-frac",
        type=float,
        default=0.1,
        help="用于验证的 holdout 比例（0-1），默认 0.1",
    )
    parser.add_argument(
        "--num-bag-folds",
        type=int,
        default=5,
        help="Bagging 折数，默认 5（减少过拟合）",
    )
    parser.add_argument(
        "--num-stack-levels",
        type=int,
        default=1,
        help="Stacking 层数，默认 1（减少过拟合）",
    )
    return parser.parse_args()


def sparse_to_dataframe(X_sparse, feature_names=None):
    """将稀疏矩阵转换为 DataFrame，用于 AutoGluon"""
    if sparse.issparse(X_sparse):
        X_dense = X_sparse.toarray()
    else:
        X_dense = np.asarray(X_sparse)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_dense.shape[1])]
    
    df = pd.DataFrame(X_dense, columns=feature_names)
    return df


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

    output_dir = args.output_dir or (args.dataset_dir / "autogluon_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将稀疏矩阵转换为 DataFrame
    print("转换训练数据为 DataFrame...")
    train_df = sparse_to_dataframe(dataset.X_train, dataset.feature_names)
    train_df["label"] = dataset.y_train
    
    print("转换测试数据为 DataFrame...")
    test_df = sparse_to_dataframe(dataset.X_test, dataset.feature_names)
    test_df["label"] = dataset.y_test

    print(f"训练集大小: {len(train_df)}, 特征数: {len(train_df.columns) - 1}")
    print(f"测试集大小: {len(test_df)}, 特征数: {len(test_df.columns) - 1}")
    print(f"训练集标签分布: {pd.Series(train_df['label']).value_counts().to_dict()}")

    # 初始化 AutoGluon TabularPredictor
    print(f"\n初始化 AutoGluon TabularPredictor (presets={args.presets}, eval_metric={args.eval_metric})...")
    predictor = TabularPredictor(
        label="label",
        path=str(output_dir / "autogluon_model"),
        eval_metric=args.eval_metric,
        problem_type="binary",
        verbosity=2,
    )

    # 训练模型
    print(f"\n开始训练（时间限制: {args.time_limit}秒）...")
    # 设置随机种子（AutoGluon 内部会使用环境变量或 ag_args）
    import os
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    
    predictor.fit(
        train_data=train_df,
        presets=args.presets,
        time_limit=args.time_limit,
        holdout_frac=args.holdout_frac,
        num_bag_folds=args.num_bag_folds,
        num_stack_levels=args.num_stack_levels,
        hyperparameters={
            'GBM': [
                {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'min_data_in_leaf': 20},
            ],
            'CAT': [
                {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
            ],
            'XGB': [
                {'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
            ],
        },
    )

    # 查看模型排行榜
    print("\n模型排行榜:")
    leaderboard = predictor.leaderboard(train_df, silent=True)
    print(leaderboard.head(10))

    # 预测
    print("\n进行预测...")
    train_prob = predictor.predict_proba(train_df.drop(columns=["label"]))[1].values
    test_prob = predictor.predict_proba(test_df.drop(columns=["label"]))[1].values

    # 计算指标
    train_pred = (train_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)
    
    metrics = {
        "train": _collect_metrics(dataset.y_train, train_prob),
        "test": _collect_metrics(dataset.y_test, test_prob),
    }
    
    # 计算 QWK (Quadratic Weighted Kappa)
    train_qwk = cohen_kappa_score(dataset.y_train, train_pred, weights='quadratic')
    test_qwk = cohen_kappa_score(dataset.y_test, test_pred, weights='quadratic')
    metrics["train"]["qwk"] = float(train_qwk)
    metrics["test"]["qwk"] = float(test_qwk)

    # 保存指标
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 保存排行榜
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"\n指标已写入 {metrics_path}")
    print(f"模型排行榜已保存至 {leaderboard_path}")
    print(f"最佳模型保存在 {output_dir / 'autogluon_model'}")
    
    # 打印测试集指标摘要
    print("\n" + "=" * 60)
    print("测试集评估结果:")
    print("=" * 60)
    test_metrics = metrics["test"]
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("QWK (Quadratic Weighted Kappa):")
    print("=" * 60)
    print(f"Train QWK: {train_qwk:.6f}")
    print(f"Test QWK:  {test_qwk:.6f}")


def _collect_metrics(y_true: np.ndarray, prob: np.ndarray):
    pred = (prob >= 0.5).astype(int)
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

