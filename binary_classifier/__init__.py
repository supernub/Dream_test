"""
dream_test.binary_classifier
============================

用于基于 `dream_test/data` 子目录中的 `cells_subset.h5ad` 或 `cells_metadata.csv`
进行二分类建模的工具模块。
"""

from .data import BinaryDataset, load_binary_dataset  # noqa: F401
from .model import build_default_classifier  # noqa: F401



