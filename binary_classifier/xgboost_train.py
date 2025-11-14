import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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
        default=0.1,
        help="XGBoost 学习率",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="树的最大深度",
    )
    parser.add_argument(
        "--num-round",
        type=int,
        default=200,
        help=" boosting 轮数",
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
        default=1.0,
        help="正样本权重（处理类别不平衡）",
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

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "scale_pos_weight": args.scale_pos_weight,
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

    metrics = {
        "train": _collect_metrics(dataset.y_train, train_prob),
        "test": _collect_metrics(dataset.y_test, test_prob),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    model_path = output_dir / "model.json"
    bst.save_model(model_path)

    print(f"指标已写入 {metrics_path}")
    print(f"模型已保存至 {model_path}")


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


