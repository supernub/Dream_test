#!/bin/bash

# =============================================================================
# DREAM Pipeline - Donor Classification Only (Step 4)
# =============================================================================
# This script runs only the donor classification step using existing outputs
# from Steps 1-3 (data splitting, transformer training, embedding generation)
# =============================================================================

    # Parse arguments
    DATA_TYPE=""
    USE_XGBOOST=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --testcase)
                DATA_TYPE="testcase"
                OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/test_case_pipeline_8donors"
                SCREEN_NAME="dream_donor_classifier_test"
                shift
                ;;
            --mtg)
                DATA_TYPE="mtg"
                OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/real_data_pipeline_mtg"
                SCREEN_NAME="dream_donor_classifier_mtg"
                shift
                ;;
            --xgboost)
                USE_XGBOOST=true
                shift
                ;;
            *)
                echo "❌ Error: Unknown option: $1"
                echo "Usage: $0 [--testcase|--mtg] [--xgboost]"
                exit 1
                ;;
        esac
    done

if [ -z "$DATA_TYPE" ]; then
    echo "❌ Error: Please specify --testcase or --mtg"
    echo "Usage: $0 [--testcase|--mtg] [--xgboost]"
    exit 1
fi

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/donor_classification_only_${TIMESTAMP}.log"

echo "🚀 Starting DREAM Donor Classification Only ($DATA_TYPE)..."
echo "📁 Output Directory: $OUTPUT_DIR"
echo "📝 Log File: $LOG_FILE"

# Set data paths based on data type
if [ "$DATA_TYPE" = "testcase" ]; then
    H5AD_PATH="/home/ubuntu/LLM-inference/xinze-project/test_data/SEAAD_A9_testcase_8donors.h5ad"
else
    H5AD_PATH="/home/ubuntu/LLM-inference/xinze-project/ADNC_Pred/training_data/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad"
fi

# Check if required files exist
if [ "$USE_XGBOOST" = true ]; then
    # XGBoost doesn't need the original h5ad file
    REQUIRED_FILES=(
        "$OUTPUT_DIR/donor_split.json"
        "$OUTPUT_DIR/transformer_model/best_model.pt"
        "$OUTPUT_DIR/embeddings/donor_embeddings.npy"
        "$OUTPUT_DIR/embeddings/donor_labels.npy"
        "$OUTPUT_DIR/embeddings/cell_predictions.npy"
        "$OUTPUT_DIR/embeddings/cell_labels.npy"
        "$OUTPUT_DIR/embeddings/metadata.json"
    )
else
    # MLP needs the original h5ad file
    REQUIRED_FILES=(
        "$H5AD_PATH"
        "$OUTPUT_DIR/donor_split.json"
        "$OUTPUT_DIR/transformer_model/best_model.pt"
        "$OUTPUT_DIR/embeddings/donor_embeddings.npy"
        "$OUTPUT_DIR/embeddings/donor_labels.npy"
        "$OUTPUT_DIR/embeddings/cell_predictions.npy"
        "$OUTPUT_DIR/embeddings/cell_labels.npy"
        "$OUTPUT_DIR/embeddings/metadata.json"
    )
fi

echo "🔍 Checking required files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        echo "Please run the full pipeline (Steps 1-3) first!"
        exit 1
    else
        echo "✅ Found: $(basename "$file")"
    fi
done

# Run donor classification directly (no screen session for better logging)
cd /home/ubuntu/LLM-inference/xinze-project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm

echo "🚀 Starting DREAM Donor Classification Only ($DATA_TYPE)..." | tee -a $LOG_FILE
echo "📁 Using existing outputs from: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "Step 4: Donor Classification ($DATA_TYPE)..." | tee -a $LOG_FILE
echo "=============================================================" | tee -a $LOG_FILE

# Run donor classification with optimized hyperparameters
if [ "$USE_XGBOOST" = true ]; then
    echo "🚀 Using XGBoost for donor classification with optimized hyperparameters..." | tee -a $LOG_FILE
    
    if [ "$DATA_TYPE" = "testcase" ]; then
        python dream/scripts/donor_classifier.py --split_json $OUTPUT_DIR/donor_split.json --embeddings_path $OUTPUT_DIR/embeddings/donor_embeddings.npy --predictions_path $OUTPUT_DIR/embeddings/donor_labels.npy --labels_path $OUTPUT_DIR/embeddings/cell_labels.npy --output_dir $OUTPUT_DIR/donor_classifier --model_type xgboost --xgb_n_estimators 200 --xgb_early_stopping_rounds 30 2>&1 | tee -a $LOG_FILE
    else
        python dream/scripts/donor_classifier.py --split_json $OUTPUT_DIR/donor_split.json --embeddings_path $OUTPUT_DIR/embeddings/donor_embeddings.npy --predictions_path $OUTPUT_DIR/embeddings/donor_labels.npy --labels_path $OUTPUT_DIR/embeddings/cell_labels.npy --output_dir $OUTPUT_DIR/donor_classifier --model_type xgboost --xgb_n_estimators 150 --xgb_early_stopping_rounds 25 --xgb_max_depth 4 --xgb_lr 0.05 --xgb_subsample 0.9 --xgb_colsample_bytree 0.9 --xgb_reg_alpha 0.2 --xgb_reg_lambda 1.5 --xgb_min_child_weight 2 --xgb_gamma 0.05 2>&1 | tee -a $LOG_FILE
    fi
else
    echo "🚀 Using MLP for donor classification..." | tee -a $LOG_FILE
    if [ "$DATA_TYPE" = "testcase" ]; then
        python dream/scripts/donor_classifier.py --h5ad_path $H5AD_PATH --split_json $OUTPUT_DIR/donor_split.json --embeddings_path $OUTPUT_DIR/embeddings/donor_embeddings.npy --predictions_path $OUTPUT_DIR/embeddings/donor_labels.npy --labels_path $OUTPUT_DIR/embeddings/cell_labels.npy --output_dir $OUTPUT_DIR/donor_classifier --hidden_dims 32 16 --epochs 200 --batch_size 32 --lr 0.005 --weight_decay 1e-4 --loss_function coral --use_balanced_sampling --early_stopping_patience 50 --min_delta 0.001 2>&1 | tee -a $LOG_FILE
    else
        python dream/scripts/donor_classifier.py --h5ad_path $H5AD_PATH --split_json $OUTPUT_DIR/donor_split.json --embeddings_path $OUTPUT_DIR/embeddings/donor_embeddings.npy --predictions_path $OUTPUT_DIR/embeddings/donor_labels.npy --labels_path $OUTPUT_DIR/embeddings/cell_labels.npy --output_dir $OUTPUT_DIR/donor_classifier --hidden_dims 64 32 --epochs 50 --batch_size 32 --lr 0.001 --weight_decay 1e-3 --loss_function coral --use_balanced_sampling --early_stopping_patience 15 --min_delta 0.001 2>&1 | tee -a $LOG_FILE
    fi
fi

echo "✅ Donor Classification Complete!" | tee -a $LOG_FILE
echo "📊 Results saved to: $OUTPUT_DIR/donor_classifier/" | tee -a $LOG_FILE
echo "=============================================================" | tee -a $LOG_FILE

echo "✅ Donor classification completed!"
echo "📝 View full log: cat $LOG_FILE"
echo ""
echo "🔍 Required files verified:"
echo "  ✅ Original data: $H5AD_PATH"
echo "  ✅ Data split: $OUTPUT_DIR/donor_split.json"
echo "  ✅ Transformer model: $OUTPUT_DIR/transformer_model/best_model.pt"
echo "  ✅ Donor embeddings: $OUTPUT_DIR/embeddings/donor_embeddings.npy"
echo "  ✅ Donor labels: $OUTPUT_DIR/embeddings/donor_labels.npy"
echo "  ✅ Cell predictions: $OUTPUT_DIR/embeddings/cell_predictions.npy"
echo "  ✅ Cell labels: $OUTPUT_DIR/embeddings/cell_labels.npy"
echo "  ✅ Metadata: $OUTPUT_DIR/embeddings/metadata.json"
echo ""
echo "🎯 Running donor classification with optimized hyperparameters..."
