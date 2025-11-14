#!/bin/bash

# Smart pipeline runner with data type selection and skip logic
# Usage: ./run_pipeline_smart.sh --testcase  OR  ./run_pipeline_smart.sh --mtg

# Get timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse arguments
if [ "$1" = "--testcase" ]; then
    DATA_TYPE="testcase"
    H5AD_PATH="/home/ubuntu/LLM-inference/xinze-project/test_data/SEAAD_A9_testcase_8donors.h5ad"
    OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/test_case_pipeline_8donors"
    SCREEN_NAME="dream_pipeline_test"
    LOG_FILE="$OUTPUT_DIR/dream_pipeline_test_${TIMESTAMP}.log"
    DONOR_COL="Donor_space_ID"
elif [ "$1" = "--mtg" ]; then
    DATA_TYPE="mtg"
    H5AD_PATH="/home/ubuntu/LLM-inference/xinze-project/ADNC_Pred/training_data/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad"
    OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/real_data_pipeline_mtg"
    SCREEN_NAME="dream_pipeline"
    LOG_FILE="$OUTPUT_DIR/dream_pipeline_${TIMESTAMP}.log"
    DONOR_COL="Donor ID"
else
    echo "❌ Error: Please specify --testcase or --mtg"
    echo "Usage: ./run_pipeline_smart.sh --testcase  OR  ./run_pipeline_smart.sh --mtg"
    exit 1
fi

echo "🚀 Starting DREAM Pipeline ($DATA_TYPE) (Steps 1→2→3→4)..."

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Kill any existing session
screen -S $SCREEN_NAME -X quit 2>/dev/null

# Check if data split already exists
SPLIT_FILE="$OUTPUT_DIR/donor_split.json"
SKIP_SPLIT=false

if [ -f "$SPLIT_FILE" ]; then
    echo "✅ Found existing split file: $SPLIT_FILE"
    echo "⏭️  Skipping Step 1 (Data Splitting)"
    SKIP_SPLIT=true
else
    echo "📊 No existing split found, will run Step 1 (Data Splitting)"
fi

# Start pipeline in screen session
screen -S $SCREEN_NAME -dm bash -c "
    cd /home/ubuntu/LLM-inference/xinze-project
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate env_vllm
    echo '🚀 Starting DREAM Pipeline ($DATA_TYPE) (Steps 1→2→3→4)...' | tee -a $LOG_FILE
    
    if [ '$SKIP_SPLIT' = 'false' ]; then
        echo 'Step 1: Data Splitting ($DATA_TYPE)...' | tee -a $LOG_FILE
        python dream/scripts/data_split.py --h5ad_path $H5AD_PATH --output_path $OUTPUT_DIR/donor_split.json --split_type donor --adnc_col ADNC --donor_col \"$DONOR_COL\" 2>&1 | tee -a $LOG_FILE
    else
        echo 'Step 1: SKIPPED (using existing split)' | tee -a $LOG_FILE
    fi
    
    echo 'Step 2: Transformer Training ($DATA_TYPE)...' | tee -a $LOG_FILE
    export CUDA_VISIBLE_DEVICES=0
    if [ "$DATA_TYPE" = "testcase" ]; then
        accelerate launch --num_processes 1 dream/scripts/train_transformer.py --h5ad_path $H5AD_PATH --split_json $OUTPUT_DIR/donor_split.json --output_dir $OUTPUT_DIR/transformer_model --embedding_dim 8 --depth 1 --epochs 4 --batch_size 256 --max_seq_len 1024 --lr 5e-4 --weight_decay 1e-5 --early_stopping --patience 10 --val_patience 3 --monitor_metric val_mae --use_attention_pooling --use_layer_norm 2>&1 | tee -a $LOG_FILE
    else
        accelerate launch --num_processes 1 dream/scripts/train_transformer.py --h5ad_path $H5AD_PATH --split_json $OUTPUT_DIR/donor_split.json --output_dir $OUTPUT_DIR/transformer_model --embedding_dim 128 --depth 6 --epochs 10 --batch_size 128 --max_seq_len 1024 --lr 1e-4 --weight_decay 1e-4 --early_stopping --patience 10 --val_patience 3 --monitor_metric val_mae --use_attention_pooling --use_layer_norm 2>&1 | tee -a $LOG_FILE
    fi
    
    echo 'Step 3: Embedding Extraction ($DATA_TYPE)...' | tee -a $LOG_FILE
    export CUDA_VISIBLE_DEVICES=0
    if [ "$DATA_TYPE" = "testcase" ]; then
        accelerate launch --num_processes 1 dream/scripts/extract_embeddings_optimized.py --h5ad_path $H5AD_PATH --split_json $OUTPUT_DIR/donor_split.json --checkpoint_path $OUTPUT_DIR/transformer_model/best_model.pt --output_dir $OUTPUT_DIR/embeddings --k_samples 100 --num_repetitions 3 --batch_size 128 2>&1 | tee -a $LOG_FILE
    else
        accelerate launch --num_processes 1 dream/scripts/extract_embeddings_optimized.py --h5ad_path $H5AD_PATH --split_json $OUTPUT_DIR/donor_split.json --checkpoint_path $OUTPUT_DIR/transformer_model/best_model.pt --output_dir $OUTPUT_DIR/embeddings --k_samples 100 --num_repetitions 16 --batch_size 128 2>&1 | tee -a $LOG_FILE
    fi
    
    echo 'Step 4: Donor Classification with XGBoost ($DATA_TYPE)...' | tee -a $LOG_FILE
    if [ "$DATA_TYPE" = "testcase" ]; then
        python dream/scripts/donor_classifier.py --split_json $OUTPUT_DIR/donor_split.json --embeddings_path $OUTPUT_DIR/embeddings/donor_embeddings.npy --predictions_path $OUTPUT_DIR/embeddings/donor_labels.npy --labels_path $OUTPUT_DIR/embeddings/cell_labels.npy --output_dir $OUTPUT_DIR/donor_classifier --model_type xgboost --xgb_n_estimators 200 --xgb_early_stopping_rounds 30 2>&1 | tee -a $LOG_FILE
    else
        python dream/scripts/donor_classifier.py --split_json $OUTPUT_DIR/donor_split.json --embeddings_path $OUTPUT_DIR/embeddings/donor_embeddings.npy --predictions_path $OUTPUT_DIR/embeddings/donor_labels.npy --labels_path $OUTPUT_DIR/embeddings/cell_labels.npy --output_dir $OUTPUT_DIR/donor_classifier --model_type xgboost --xgb_n_estimators 100 --xgb_early_stopping_rounds 20 --xgb_max_depth 3 --xgb_lr 0.1 --xgb_subsample 0.8 --xgb_colsample_bytree 0.8 --xgb_reg_alpha 0.5 --xgb_reg_lambda 2.0 --xgb_min_child_weight 3 --xgb_gamma 0.1 2>&1 | tee -a $LOG_FILE
    fi
    
    echo '✅ Pipeline ($DATA_TYPE) completed!' | tee -a $LOG_FILE
"

echo "✓ Pipeline ($DATA_TYPE) started in screen session '$SCREEN_NAME'"
echo "  Access with: screen -r $SCREEN_NAME"
echo "  View log: tail -f $LOG_FILE"
echo "  Output dir: $OUTPUT_DIR"
