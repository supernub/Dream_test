import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

try:
    import scanpy as sc
except ImportError:  # pragma: no cover - scanpy 在运行环境中缺失时提示
    sc = None  # type: ignore


@dataclass
class BinaryDataset:
    """封装单个数据集的训练 / 测试拆分."""

    name: str
    X_train: sparse.csr_matrix
    y_train: np.ndarray
    X_test: sparse.csr_matrix
    y_test: np.ndarray
    train_metadata: pd.DataFrame
    test_metadata: pd.DataFrame
    feature_names: Optional[List[str]]
    label_mapping: Dict[str, int]


class DatasetLoadError(RuntimeError):
    """数据加载异常."""


def load_binary_dataset(
    dataset_dir: Path,
    label_column: str = "binary_label",
    prefer_h5ad: bool = True,
    concat_donor_metadata: bool = False,
) -> BinaryDataset:
    """
    从指定目录加载 h5ad / metadata，并基于 split_info.json 划分训练与测试集合.

    参数
    ----
    dataset_dir: Path
        单个细胞类型（或组合）数据目录，要求包含 `cells_metadata.csv`
        以及可选的 `cells_subset.h5ad` 和 `split_info.json`.
    label_column: str
        标签列名，默认读取 `binary_label`.
    prefer_h5ad: bool
        若为 True 且 h5ad 文件存在，则优先从 h5ad 中加载特征；
        否则降级为基于 metadata 的稀疏 One-Hot 特征。
    concat_donor_metadata: bool
        若为 True，将 donor-level 统计特征拼接至细胞特征。
        基于 metadata 中的 donor_id 列计算每个 donor 的统计量。
    """
    dataset_dir = Path(dataset_dir)

    csv_path = dataset_dir / "cells_metadata.csv"
    if not csv_path.exists():
        raise DatasetLoadError(f"{dataset_dir} 缺少 cells_metadata.csv")

    metadata = _load_and_normalise_metadata(csv_path)

    label_series = metadata.get(label_column)
    if label_series is None:
        raise DatasetLoadError(f"{csv_path} 中不存在标签列 `{label_column}`")

    labels, label_mapping = _normalise_labels(label_series)
    metadata = metadata.assign(_label_encoded=labels)

    split_path = dataset_dir / "split_info.json"
    if split_path.exists():
        split_config = json.loads(split_path.read_text())
    else:
        split_config = {}

    feature_matrix, feature_names = _load_feature_matrix(
        dataset_dir, metadata, prefer_h5ad=prefer_h5ad
    )

    train_mask, test_mask = _build_split_masks(metadata, split_config)
    
    # 若需要拼接 donor metadata，则计算并拼接 donor-level 特征
    # 注意：仅基于训练集 donor 计算统计特征，然后应用到所有细胞
    if concat_donor_metadata:
        donor_features, donor_feature_names = _build_donor_metadata_features(
            metadata, label_column, train_mask=train_mask
        )
        feature_matrix = sparse.hstack([feature_matrix, donor_features], format="csr")
        if feature_names is not None:
            feature_names = feature_names + donor_feature_names
        else:
            feature_names = donor_feature_names
    if not train_mask.any():
        raise DatasetLoadError(f"{dataset_dir} 训练集合为空，请检查 split_info.json")
    if not test_mask.any():
        raise DatasetLoadError(f"{dataset_dir} 测试集合为空，请检查 split_info.json")

    X_train = feature_matrix[train_mask]
    y_train = labels[train_mask]
    X_test = feature_matrix[test_mask]
    y_test = labels[test_mask]

    train_metadata = metadata.loc[train_mask].reset_index(drop=True)
    test_metadata = metadata.loc[test_mask].reset_index(drop=True)

    return BinaryDataset(
        name=dataset_dir.name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        feature_names=feature_names,
        label_mapping=label_mapping,
    )


# ---------------------------------------------------------------------------
# 内部辅助函数


def _load_and_normalise_metadata(csv_path: Path) -> pd.DataFrame:
    """读取 metadata CSV，并标准化列名 / 排序."""
    raw_df = pd.read_csv(csv_path)
    normalised_columns = {
        col: col.strip().lower().replace(" ", "_")
        for col in raw_df.columns
    }
    df = raw_df.rename(columns=normalised_columns)

    if "subset_index" in df.columns:
        df = df.sort_values("subset_index").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def _normalise_labels(label_series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """把标签转换为 0/1 并返回原始->编码映射."""
    if pd.api.types.is_numeric_dtype(label_series):
        labels = label_series.to_numpy(dtype=np.int64, copy=True)
        uniq = sorted(np.unique(labels))
        mapping = {str(v): int(v) for v in uniq}
        return labels, mapping

    values = label_series.astype(str)
    uniq = sorted(values.unique())
    mapping = {val: idx for idx, val in enumerate(uniq)}
    encoded = values.map(mapping).to_numpy(dtype=np.int64)
    return encoded, mapping


def _load_feature_matrix(
    dataset_dir: Path,
    metadata: pd.DataFrame,
    prefer_h5ad: bool = True,
) -> Tuple[sparse.csr_matrix, Optional[List[str]]]:
    """加载特征矩阵，优先使用 h5ad."""
    h5ad_path = dataset_dir / "cells_subset.h5ad"

    if prefer_h5ad and h5ad_path.exists():
        if sc is None:
            raise DatasetLoadError("需要安装 scanpy 才能读取 h5ad 文件")
        adata = sc.read_h5ad(h5ad_path)
        if adata.n_obs != len(metadata):
            raise DatasetLoadError(
                f"{h5ad_path} 中的细胞数量 ({adata.n_obs}) 与 metadata ({len(metadata)}) 不一致"
            )
        X = adata.X
        if sparse.issparse(X):
            feature_matrix = X.tocsr()
        else:
            feature_matrix = sparse.csr_matrix(np.asarray(X))
        feature_names = (
            adata.var_names.to_list() if hasattr(adata.var_names, "to_list") else None
        )
        return feature_matrix, feature_names

    # 若不存在 h5ad，则基于 metadata 构建稀疏特征
    return _build_sparse_features_from_metadata(metadata)


def _build_sparse_features_from_metadata(
    metadata: pd.DataFrame,
    label_column: str = "_label_encoded",
) -> Tuple[sparse.csr_matrix, List[str]]:
    """使用 DictVectorizer 将 metadata（除标签列外）转换成稀疏特征."""
    exclude = {label_column, "subset_index", "orig_index"}
    columns = [c for c in metadata.columns if c not in exclude]

    if not columns:
        raise DatasetLoadError("metadata 中没有可用的特征列")

    records = metadata[columns].to_dict(orient="records")
    vectorizer = DictVectorizer(sparse=True)
    feature_matrix = vectorizer.fit_transform(records).tocsr()
    feature_names = vectorizer.get_feature_names_out().tolist()
    return feature_matrix, feature_names


def _build_split_masks(
    metadata: pd.DataFrame,
    split_config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """根据 split_info.json 创建训练 / 测试掩码."""
    n = len(metadata)

    ignored_mask = _mask_from_donors(metadata, split_config.get("ignored_donors", []))

    train_mask = _resolve_mask(
        metadata,
        split_config,
        primary_keys=("train_donors", "train_cell_indices", "train_indices"),
    )
    test_mask = _resolve_mask(
        metadata,
        split_config,
        primary_keys=(
            "test_donors",
            "test_cell_indices",
            "test_indices",
            "val_donors",
            "val_cell_indices",
            "val_indices",
        ),
    )

    if train_mask is None:
        # 默认前 80% 作为训练
        default_split = int(n * 0.8)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:default_split] = True
    if test_mask is None:
        test_mask = ~train_mask

    if ignored_mask is not None:
        train_mask = train_mask & ~ignored_mask
        test_mask = test_mask & ~ignored_mask

    return train_mask, test_mask


def _mask_from_donors(metadata: pd.DataFrame, donors: Iterable[str]) -> Optional[np.ndarray]:
    donors = [d for d in donors if d]
    if not donors:
        return None
    column_candidates = ["donor_id", "donor"]
    for col in column_candidates:
        if col in metadata.columns:
            mask = metadata[col].astype(str).isin(donors)
            return mask.to_numpy(dtype=bool)
    return None


def _mask_from_indices(indices: Iterable[int], length: int) -> np.ndarray:
    idx = np.asarray(list(indices), dtype=np.int64)
    mask = np.zeros(length, dtype=bool)
    valid = idx[(0 <= idx) & (idx < length)]
    mask[valid] = True
    return mask


def _resolve_mask(
    metadata: pd.DataFrame,
    split_config: Dict,
    primary_keys: Tuple[str, ...],
) -> Optional[np.ndarray]:
    """优先使用 donor 拆分，其次使用 cell indices."""
    for key in primary_keys:
        if key.endswith("donors") and key in split_config:
            mask = _mask_from_donors(metadata, split_config[key])
            if mask is not None and mask.any():
                return mask
        elif key in split_config:
            mask = _mask_from_indices(split_config[key], len(metadata))
            if mask.any():
                return mask
    return None


def _build_donor_metadata_features(
    metadata: pd.DataFrame,
    label_column: str = "binary_label",
    train_mask: Optional[np.ndarray] = None,
) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    基于 metadata 构建 donor-level 统计特征，并将其广播到每个细胞。
    
    为每个 donor 计算：
    - 该 donor 的细胞总数（log scale）
    - 该 donor 中 binary_label=1 的比例
    - 该 donor 中 binary_label=0 的比例（如果两类都存在）
    - ADNC 值的分布统计（如果有 adnc 列）
    
    参数
    ----
    train_mask: Optional[np.ndarray]
        若提供，仅基于训练集 donor 计算统计特征（避免数据泄露）。
        测试集的 donor 将使用训练集中相同 donor 的统计量，或使用默认值。
    
    返回 (n_cells, n_donor_features) 的稀疏矩阵和特征名列表。
    """
    donor_col = None
    for col_candidate in ["donor_id", "donor", "donor id"]:
        if col_candidate in metadata.columns:
            donor_col = col_candidate
            break
    
    if donor_col is None:
        raise DatasetLoadError("metadata 中未找到 donor ID 列（期望 donor_id、donor 或 donor id）")
    
    # 若提供了 train_mask，仅基于训练集计算统计特征
    if train_mask is not None:
        train_metadata = metadata.loc[train_mask].copy()
    else:
        train_metadata = metadata.copy()
    
    # 获取每个 donor 的统计量（仅基于训练集）
    donor_stats = []
    donor_label_col = label_column.lower().replace(" ", "_")
    
    if donor_label_col in train_metadata.columns:
        donor_label_counts = train_metadata.groupby(donor_col)[donor_label_col].agg(["sum", "count"])
        donor_label_counts["positive_ratio"] = donor_label_counts["sum"] / donor_label_counts["count"]
        donor_label_counts["negative_ratio"] = 1.0 - donor_label_counts["positive_ratio"]
        donor_label_counts["log_cell_count"] = np.log1p(donor_label_counts["count"])
        donor_stats.append(donor_label_counts[["positive_ratio", "negative_ratio", "log_cell_count"]])
    
    if "adnc" in train_metadata.columns:
        adnc_dummies = pd.get_dummies(train_metadata[[donor_col, "adnc"]], columns=["adnc"], prefix="adnc")
        adnc_ratios = adnc_dummies.groupby(donor_col).mean()
        donor_stats.append(adnc_ratios)
    
    if donor_stats:
        donor_feat_df = pd.concat(donor_stats, axis=1)
    else:
        # 如果没有可用统计量，至少提供 cell count
        cell_counts = train_metadata.groupby(donor_col).size()
        donor_feat_df = pd.DataFrame({
            "log_cell_count": np.log1p(cell_counts)
        }, index=cell_counts.index)
    
    # 对于测试集中新出现的 donor，使用全局统计量的均值填充
    # 对于缺失的列，使用 0.0 填充
    donor_feat_df = donor_feat_df.fillna(0.0)
    
    # 获取所有唯一的 donor（包括测试集中新出现的）
    all_donors = metadata[donor_col].unique()
    
    # 对于测试集中新出现的 donor，使用训练集的均值作为默认值
    default_values = donor_feat_df.mean().to_dict()
    for donor in all_donors:
        if donor not in donor_feat_df.index:
            donor_feat_df.loc[donor] = default_values
    
    # 将 donor 特征广播到每个细胞
    cell_donor_features = donor_feat_df.loc[metadata[donor_col].values].values
    
    # 转换为稀疏矩阵
    donor_features_sparse = sparse.csr_matrix(cell_donor_features.astype(np.float32))
    donor_feature_names = donor_feat_df.columns.tolist()
    
    return donor_features_sparse, donor_feature_names



