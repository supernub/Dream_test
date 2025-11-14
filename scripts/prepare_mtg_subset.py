#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare a manageable MTG subset (train/test) from the large h5ad in backed mode.
It intersects provided donor split with valid ADNC labels, samples up to a cap,
and writes two smaller h5ad files for downstream CellFM fine-tuning.

Usage:
  python prepare_mtg_subset.py \
    --h5ad_path /path/to/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad \
    --split_json /path/to/donor_split.json \
    --output_dir /path/to/outputs/cellfm_mtg \
    --max_train 200000 --max_test 100000
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
ADNC_ALLOWED = set(ADNC_MAP.keys())


def normalize_adnc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        num_to_name = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}
        return series.map(num_to_name).astype(str)
    s = series.astype(str)
    map_num = {'0': 'Not AD', '1': 'Low', '2': 'Intermediate', '3': 'High'}
    return s.replace(map_num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad_path', required=True)
    ap.add_argument('--split_json', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--adnc_col', default='ADNC')
    ap.add_argument('--max_train', type=int, default=200_000)
    ap.add_argument('--max_test', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[prepare] reading backed h5ad: {args.h5ad_path}")
    adata = sc.read_h5ad(args.h5ad_path, backed='r')

    with open(args.split_json, 'r') as f:
        split = json.load(f)

    adnc_series = adata.obs[args.adnc_col]
    adnc_norm = normalize_adnc(adnc_series)
    valid_mask = adnc_norm.isin(ADNC_ALLOWED).values

    train_idx = np.array(split.get('train_indices', []), dtype=np.int64)
    test_idx = np.array(split.get('test_indices', []), dtype=np.int64)

    # intersect with valid mask
    train_idx = train_idx[valid_mask[train_idx]]
    test_idx = test_idx[valid_mask[test_idx]]

    rng = np.random.default_rng(args.seed)

    if train_idx.size > args.max_train:
        train_idx = rng.choice(train_idx, size=args.max_train, replace=False)
    if test_idx.size > args.max_test:
        test_idx = rng.choice(test_idx, size=args.max_test, replace=False)

    print(f"[prepare] sampled cells -> train: {train_idx.size}, test: {test_idx.size}")

    # write subsets; slicing a backed AnnData returns a view that can be written
    train_out = os.path.join(args.output_dir, 'train_subset.h5ad')
    test_out = os.path.join(args.output_dir, 'test_subset.h5ad')

    print(f"[prepare] writing train subset: {train_out}")
    adata[train_idx].write(train_out)
    print(f"[prepare] writing test subset:  {test_out}")
    adata[test_idx].write(test_out)

    print("[prepare] done")


if __name__ == '__main__':
    main()





