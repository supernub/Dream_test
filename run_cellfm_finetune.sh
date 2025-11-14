#!/bin/bash

# CellFM Fine-tune Script for MTG Dataset
# Usage: ./run_cellfm_finetune.sh [--testcase|--mtg]

# Parse arguments
if [ "$1" = "--testcase" ] || [ "$1" = "--testcase-4" ]; then
    DATA_TYPE="testcase-4"
    H5AD_PATH="/home/ubuntu/LLM-inference/xinze-project/test_data/SEAAD_A9_testcase_4donors.h5ad"
    OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_testcase_4"
    SPLIT_JSON="/home/ubuntu/LLM-inference/xinze-project/outputs/test_case_pipeline_4donors/donor_split.json"
elif [ "$1" = "--testcase-8" ]; then
    DATA_TYPE="testcase-8"
    H5AD_PATH="/home/ubuntu/LLM-inference/xinze-project/test_data/SEAAD_A9_testcase_8donors.h5ad"
    OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_testcase_8"
    SPLIT_JSON="/home/ubuntu/LLM-inference/xinze-project/outputs/test_case_pipeline_8donors/donor_split.json"
elif [ "$1" = "--mtg" ]; then
    DATA_TYPE="mtg"
    H5AD_PATH="/home/ubuntu/LLM-inference/xinze-project/ADNC_Pred/training_data/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad"
    OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg"
    SPLIT_JSON="/home/ubuntu/LLM-inference/xinze-project/outputs/real_data_pipeline_mtg/donor_split.json"
else
    echo "❌ Error: Please specify --testcase-4, --testcase-8, or --mtg"
    echo "Usage:"
    echo "  ./run_cellfm_finetune.sh --testcase-4   (4 donors testcase)"
    echo "  ./run_cellfm_finetune.sh --testcase-8   (8 donors testcase)"
    echo "  ./run_cellfm_finetune.sh --mtg           (Full MTG dataset)"
    exit 1
fi

# Check if checkpoint exists
CKPT_PATH="/home/ubuntu/LLM-inference/xinze-project/cellfm/CellFM_80M_weight.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "⚠️  Warning: CellFM checkpoint not found at $CKPT_PATH"
    echo "Please download from: https://huggingface.co/ShangguanNingyuan/CellFM/tree/main"
    echo "Or specify --ckpt_path when running the script"
    CKPT_ARG=""
else
    CKPT_ARG="--ckpt_path $CKPT_PATH"
fi

echo "🚀 Starting CellFM Fine-tune ($DATA_TYPE)"
echo "============================================"
echo "Data path: $H5AD_PATH"
echo "Split JSON: $SPLIT_JSON"
echo "Output dir: $OUTPUT_DIR"
echo "Checkpoint: $CKPT_PATH"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run CellFM fine-tune (using custom gene embedding version)
cd /home/ubuntu/LLM-inference/xinze-project/dream_test

if [ "$DATA_TYPE" = "testcase-4" ]; then
    python scripts/train_cellfm_adnc_custom.py \
        --h5ad_path "$H5AD_PATH" \
        --split_json "$SPLIT_JSON" \
        --output_dir "$OUTPUT_DIR" \
        $CKPT_ARG \
        --device cuda:0 \
        --batch_size 16 \
        --epochs 3 \
        --lr 1e-4 \
        --num_cls 4 \
        --eval_interval_steps 50 \
        --patience_steps 3
elif [ "$DATA_TYPE" = "testcase-8" ]; then
    python scripts/train_cellfm_adnc_custom.py \
        --h5ad_path "$H5AD_PATH" \
        --split_json "$SPLIT_JSON" \
        --output_dir "$OUTPUT_DIR" \
        $CKPT_ARG \
        --device cuda:0 \
        --batch_size 16 \
        --epochs 3 \
        --lr 1e-4 \
        --num_cls 4 \
        --eval_interval_steps 100 \
        --patience_steps 3
else
    python scripts/train_cellfm_adnc_custom.py \
        --h5ad_path "$H5AD_PATH" \
        --split_json "$SPLIT_JSON" \
        --output_dir "$OUTPUT_DIR" \
        $CKPT_ARG \
        --device cuda:0 \
        --batch_size 8 \
        --epochs 5 \
        --lr 1e-4 \
        --num_cls 4
fi

echo "✅ CellFM training completed!"
echo "Results saved to: $OUTPUT_DIR"

