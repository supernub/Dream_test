#!/bin/bash
# Run CellFM training with balanced sampling

set -e

cd /home/ubuntu/LLM-inference/xinze-project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm

# Configuration
H5AD_PATH="/home/ubuntu/LLM-inference/xinze-project/ADNC_Pred/training_data/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad"
SPLIT_JSON="/home/ubuntu/LLM-inference/xinze-project/outputs/real_data_pipeline_mtg/donor_split.json"
OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg_balanced"
CKPT_PATH="/home/ubuntu/LLM-inference/xinze-project/cellfm/CellFM_80M_weight.ckpt"
DEVICE="cuda:0"
BATCH_SIZE=16
EPOCHS=10
LR=1e-4

echo "=========================================="
echo "CellFM Balanced Training"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Prepare subset first
echo "Preparing balanced subsets..."
python dream_test/scripts/prepare_mtg_subset.py \
    --h5ad_path "$H5AD_PATH" \
    --split_json "$SPLIT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --max_train 200000 \
    --max_test 50000

echo ""
echo "Starting balanced training..."
python dream_test/scripts/train_cellfm_balanced.py \
    --h5ad_path "$H5AD_PATH" \
    --split_json "$SPLIT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Results: $OUTPUT_DIR"
echo "=========================================="



