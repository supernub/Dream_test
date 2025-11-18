from dataclasses import dataclass
from typing import Optional

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


@dataclass
class ClassifierConfig:
    """默认二分类模型的可调参数."""

    alpha: float = 1e-4
    max_iter: int = 500
    tol: float = 1e-3
    random_state: int = 42
    class_weight: Optional[str] = "balanced"


def build_default_classifier(config: Optional[ClassifierConfig] = None) -> Pipeline:
    """
    创建基于 `SGDClassifier(log_loss)` 的稀疏友好型分类器.

    组合：
        MaxAbsScaler —— 适合稀疏矩阵的缩放
        SGDClassifier —— 使用 log_loss，相当于 L2 正则的逻辑回归
    """
    cfg = config or ClassifierConfig()
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=cfg.alpha,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
    )
    return Pipeline(
        steps=[
            ("scale", MaxAbsScaler()),
            ("clf", classifier),
        ]
    )




