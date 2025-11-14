# DREAM Pipeline: Deep Learning for ADNC Prediction

This repository contains the DREAM (Deep RNA Expression Analysis for Medical) pipeline for predicting ADNC (Alzheimer's Disease Neuropathologic Change) levels from single-cell RNA sequencing data using transformer-based embeddings and machine learning classifiers.

## Overview

The DREAM pipeline consists of 4 main steps:
1. **Data Splitting**: Split donors into train/test sets
2. **Transformer Training**: Train a transformer model to predict ADNC from gene expression
3. **Embedding Extraction**: Extract cell-type-specific embeddings from the trained transformer
4. **Donor Classification**: Train a classifier on donor-level aggregated embeddings

## Prerequisites

- Python 3.12 with conda environment `env_vllm`
- CUDA-compatible GPU (recommended: NVIDIA GH200 or similar)
- Required Python packages: PyTorch, XGBoost, scikit-learn, scanpy, accelerate

## Quick Start

### Option 1: Full Pipeline (Recommended)

Run the complete pipeline from data splitting to donor classification:

```bash
# For test case (8 donors)
./run_pipeline_smart.sh --testcase

# For real data (MTG dataset)
./run_pipeline_smart.sh --mtg
```

### Option 2: Donor Classification Only

If you already have results from Steps 1-3, run only the donor classification:

```bash
# For test case with MLP
./run_donor_classification_only.sh --testcase

# For test case with XGBoost
./run_donor_classification_only.sh --testcase --xgboost

# For real data with MLP
./run_donor_classification_only.sh --mtg

# For real data with XGBoost
./run_donor_classification_only.sh --mtg --xgboost
```

## Data Types

### Test Case (`--testcase`)
- **Dataset**: 8 donors from SEAAD_A9_testcase_8donors.h5ad
- **Purpose**: Quick testing and development
- **Output**: `/home/ubuntu/LLM-inference/xinze-project/outputs/test_case_pipeline_8donors/`

### Real Data (`--mtg`)
- **Dataset**: Full MTG dataset from SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad
- **Purpose**: Production analysis
- **Output**: `/home/ubuntu/LLM-inference/xinze-project/outputs/real_data_pipeline_mtg/`

## Pipeline Steps

### Step 1: Data Splitting
- Splits donors into train/test sets (80/20 split)
- Ensures no overlap between train and test donors
- Saves split information to `donor_split.json`

### Step 2: Transformer Training
- Trains a transformer model to predict ADNC levels from gene expression
- Uses attention pooling and layer normalization
- Implements early stopping based on validation MAE
- Saves best model checkpoint

**Hyperparameters:**
- **Test Case**: embedding_dim=8, depth=1, epochs=4, batch_size=256, lr=5e-4
- **Real Data**: embedding_dim=128, depth=6, epochs=10, batch_size=128, lr=1e-4

### Step 3: Embedding Extraction
- Extracts cell-type-specific embeddings from trained transformer
- Creates multiple repetitions for training donors (3 for test case, 16 for real data)
- Aggregates embeddings by donor and cell type
- Saves embeddings as numpy arrays

### Step 4: Donor Classification
- Trains a classifier on donor-level aggregated embeddings
- Supports both MLP and XGBoost models
- Uses test donors as validation during training
- Implements early stopping based on validation QWK

## Model Options

### MLP Classifier
- Multi-layer perceptron with CORAL loss for ordinal regression
- Balanced sampling and class weights
- Early stopping based on validation QWK

**Hyperparameters:**
- **Test Case**: hidden_dims=[32,16], epochs=200, batch_size=32, lr=0.005
- **Real Data**: hidden_dims=[64,32], epochs=50, batch_size=32, lr=0.001

### XGBoost Classifier
- Gradient boosting with ordinal regression
- Custom QWK evaluation metric
- Early stopping based on validation loss

**Hyperparameters:**
- **Test Case**: n_estimators=200, early_stopping_rounds=30
- **Real Data**: n_estimators=150, max_depth=4, lr=0.05, reg_alpha=0.2, reg_lambda=1.5

## Output Files

### Transformer Model
- `best_model.pt`: Best transformer checkpoint
- `training_results.json`: Training metrics and hyperparameters

### Embeddings
- `donor_embeddings.npy`: Donor-level aggregated embeddings
- `donor_labels.npy`: Donor-level ADNC labels
- `cell_embeddings.npy`: Cell-level embeddings
- `cell_predictions.npy`: Cell-level predictions
- `cell_labels.npy`: Cell-level labels
- `metadata.json`: Embedding metadata

### Donor Classification
- `donor_classifier.pt`: Trained classifier (MLP) or model file (XGBoost)
- `donor_classifier_results.json`: Classification metrics and results

## Monitoring Progress

### Screen Sessions
The full pipeline runs in a screen session for long-running processes:

```bash
# Check running pipeline
screen -r dream_pipeline        # For real data
screen -r dream_pipeline_test   # For test case

# Detach from screen: Ctrl+A, then D
```

### Log Files
Monitor progress through log files:

```bash
# Real data logs
tail -f /home/ubuntu/LLM-inference/xinze-project/outputs/real_data_pipeline_mtg/dream_pipeline_*.log

# Test case logs
tail -f /home/ubuntu/LLM-inference/xinze-project/outputs/test_case_pipeline_8donors/dream_pipeline_test_*.log

# Donor classification only logs
tail -f /home/ubuntu/LLM-inference/xinze-project/outputs/*/donor_classification_only_*.log
```

## Key Metrics

### Transformer Training
- **Validation MAE**: Mean Absolute Error on validation set
- **Validation Accuracy**: Classification accuracy
- **Validation F1**: Weighted F1 score

### Donor Classification
- **Val QWK**: Quadratic Weighted Kappa (primary metric)
- **Val Accuracy**: Classification accuracy
- **Val F1**: Weighted F1 score
- **Detailed Predictions**: Individual donor results with ground truth vs predictions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in transformer training
   - Use smaller embedding dimensions

2. **Poor QWK Scores**
   - Check class imbalance in validation set
   - Try different hyperparameters
   - Consider using MLP instead of XGBoost

3. **Training Stuck**
   - Check early stopping parameters
   - Monitor validation metrics
   - Adjust learning rate

### File Requirements

For `run_donor_classification_only.sh`, ensure these files exist:
- `donor_split.json`
- `transformer_model/best_model.pt`
- `embeddings/donor_embeddings.npy`
- `embeddings/donor_labels.npy`
- `embeddings/cell_labels.npy`
- `embeddings/metadata.json`

## Performance Expectations

### Test Case (8 donors)
- **Training Time**: ~5-10 minutes
- **Expected QWK**: 0.3-0.6
- **Expected Accuracy**: 60-80%

### Real Data (MTG)
- **Training Time**: ~2-4 hours
- **Expected QWK**: 0.2-0.4
- **Expected Accuracy**: 50-70%

## Advanced Usage

### Custom Hyperparameters
Modify hyperparameters directly in the script files:
- `run_pipeline_smart.sh`: Transformer and XGBoost parameters
- `run_donor_classification_only.sh`: MLP and XGBoost parameters

### Model Selection
- **MLP**: Better for high-dimensional embeddings, more stable training
- **XGBoost**: Faster training, good feature importance analysis

### Validation Strategy
The pipeline uses test donors as validation during training to optimize for the specific test set, which is appropriate for this research context.

## Citation

If you use this pipeline in your research, please cite the original DREAM paper and acknowledge this implementation.

## Support

For issues or questions, please check the log files first and ensure all prerequisites are met.