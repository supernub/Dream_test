# DREAM Pipeline - Real Data Execution

This document describes the enhanced DREAM pipeline for running on the real MTG dataset with improved performance and automated execution.

## Overview

The real data pipeline is now available in two execution modes:

### 🚀 **Enhanced Mode (Recommended)**
- **Single Command**: `./start_real_data_pipeline.sh start-all`
- **Automatic Sequencing**: Steps 1→2→3→4 with proper dependencies
- **Enhanced Features**: Early stopping, ordinal regression, focal loss
- **Background Monitoring**: Non-blocking execution with progress tracking

### 🔧 **Manual Mode (Advanced)**
- **Individual Scripts**: Run each step separately
- **Full Control**: Custom hyperparameters and monitoring
- **Debugging**: Step-by-step execution for troubleshooting

## Quick Start (Enhanced Mode)

```bash
# Complete pipeline with enhanced features
./start_real_data_pipeline.sh start-all

# Choose execution mode:
#   1) Sequential (wait for completion) - Blocks terminal
#   2) Background (monitor in background) - Non-blocking
```

### Enhanced Features
- ✅ **Sequential Execution**: Proper 1→2→3→4 dependency management
- ✅ **Early Stopping**: Prevents overfitting (patience=3, monitor=val_mae)
- ✅ **Ordinal Regression**: CORAL loss for ordered classes (High > Intermediate > Low > Not AD)
- ✅ **Focal Loss**: Handles class imbalance (α=0.25, γ=2.0)
- ✅ **Enhanced Architecture**: Attention pooling + layer normalization
- ✅ **Optimized Hyperparameters**: Better convergence and performance

## Dataset Information

- **Dataset**: SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad
- **Location**: `/home/ubuntu/LLM-inference/xinze-project/ADNC_Pred/training_data/`
- **Size**: 1,240,908 cells × 36,601 genes
- **Donors**: 84 unique donors
- **ADNC Classes**: 4 (High, Intermediate, Low, Not AD)
- **Cell Types**: 139 unique supertypes

## Enhanced Pipeline Starter

**File**: `start_real_data_pipeline.sh`

### Commands

```bash
# Complete pipeline (recommended)
./start_real_data_pipeline.sh start-all

# Individual steps
./start_real_data_pipeline.sh start-1-2    # Steps 1-2 only
./start_real_data_pipeline.sh start-3-4    # Steps 3-4 only (requires 1-2 completed)

# Monitoring and status
./start_real_data_pipeline.sh monitor       # Real-time progress monitoring
./start_real_data_pipeline.sh status        # Check session status
./start_real_data_pipeline.sh help          # Show all commands
```

### Execution Modes

#### Mode 1: Sequential (Blocking)
- **Use Case**: When you want to wait for completion
- **Behavior**: Terminal blocks until pipeline finishes
- **Monitoring**: Real-time progress updates
- **Best For**: Interactive sessions, debugging

#### Mode 2: Background (Non-blocking)
- **Use Case**: When you want to run in background
- **Behavior**: Terminal is free for other tasks
- **Monitoring**: Background monitor handles sequencing
- **Best For**: Long runs, server environments

### Enhanced Hyperparameters

| Parameter | Old Value | New Value | Improvement |
|-----------|-----------|-----------|-------------|
| **Model Depth** | 4 layers | 6 layers | Better representation |
| **Learning Rate** | 2e-4 | 1e-4 | More stable training |
| **Batch Size** | 256 | 128 | Better memory usage |
| **Epochs** | 2 | 10 (with early stopping) | More training, no overfitting |
| **Early Stopping** | None | Patience=3 | Prevents overfitting |
| **Loss Function** | Basic CORAL | CORAL + Focal | Handles class imbalance |
| **Architecture** | Basic | Attention + LayerNorm | Better performance |

### Pipeline Dependencies

```
Step 1: Data Splitting
  ↓ (creates donor_split.json)
Step 2: Transformer Training  
  ↓ (creates best_model.pt)
Step 3: Embedding Extraction
  ↓ (creates cell_embeddings.npy)
Step 4: Donor Classification
  ↓ (creates final results)
```

### Monitoring Options

```bash
# Real-time monitoring
./start_real_data_pipeline.sh monitor

# Direct session access
screen -r real_data_steps_1_2      # Steps 1-2
screen -r real_data_steps_3_4      # Steps 3-4
screen -r pipeline_monitor         # Background monitor

# View logs
tail -f real_data_steps_1_2.log
tail -f real_data_steps_3_4.log
```

## Manual Mode (Advanced Users)

### Script 1: Data Splitting and Transformer Training

**File**: `run_real_data_steps_1_2.sh`

### Purpose
- Split data by donor (80% train, 20% test)
- Train transformer model with balanced sampling
- Save model checkpoint and split information

### Enhanced Hyperparameters
- **Embedding Dimension**: 128
- **Model Depth**: 6 layers (increased from 4)
- **Epochs**: 10 (with early stopping, patience=3)
- **Batch Size**: 128 (reduced from 256 for stability)
- **Max Sequence Length**: 1024
- **Learning Rate**: 1e-4 (reduced from 2e-4)
- **Weight Decay**: 1e-4
- **Early Stopping**: Enabled (patience=3, monitor=val_mae)
- **Focal Loss**: α=0.25, γ=2.0 (handles class imbalance)
- **Attention Pooling**: Enabled
- **Layer Normalization**: Enabled

### Usage
```bash
# Use all available GPUs
./run_real_data_steps_1_2.sh

# Use specific GPUs
./run_real_data_steps_1_2.sh --gpu-ids 0,1,2,3

# Use first N GPUs
./run_real_data_steps_1_2.sh --num-gpus 2
```

### Output Files
- `outputs/real_data_pipeline_mtg/donor_split.json`
- `outputs/real_data_pipeline_mtg/transformer_model/best_model.pt`
- `outputs/real_data_pipeline_mtg/transformer_model/training_log.json`

## Script 2: Embedding Extraction and Donor Classification

**File**: `run_real_data_steps_3_4.sh`

### Purpose
- Extract embeddings using trained transformer
- Train donor-level classifier
- Generate final results

### Enhanced Parameters
- **K Samples**: 100 (for donor aggregation)
- **Repetitions**: 16 (increased from 3 for better training)
- **Ordinal Regression**: CORAL loss for ordered classes
- **Batch Normalization**: Enabled for training stability
- **Enhanced Regularization**: Dropout + weight decay

### Usage
```bash
# Use all available GPUs
./run_real_data_steps_3_4.sh

# Use specific GPUs
./run_real_data_steps_3_4.sh --gpu-ids 0,1,2,3

# Use first N GPUs
./run_real_data_steps_3_4.sh --num-gpus 2
```

### Prerequisites
- Must run Script 1 first
- Requires `donor_split.json` and `best_model.pt`

### Output Files
- `outputs/real_data_pipeline_mtg/embeddings/cell_embeddings.npy`
- `outputs/real_data_pipeline_mtg/embeddings/cell_predictions.npy`
- `outputs/real_data_pipeline_mtg/embeddings/cell_labels.npy`
- `outputs/real_data_pipeline_mtg/embeddings/donor_embeddings.npy`
- `outputs/real_data_pipeline_mtg/embeddings/donor_labels.npy`
- `outputs/real_data_pipeline_mtg/donor_classifier/donor_classifier.pt`
- `outputs/real_data_pipeline_mtg/donor_classifier/donor_classifier_results.json`

## Key Features

### Balanced Sampling
- Each training batch contains equal samples from all ADNC classes
- Prevents class imbalance issues during training
- Batch size (256) is divisible by number of classes (4)

### Multi-GPU Support
- Automatic detection of available GPUs
- Uses `accelerate` for distributed training
- Fallback to single-GPU if `accelerate` not available

### Flexible Paths
- No hardcoded paths
- Works from any directory
- Relative path resolution

### Error Handling
- Comprehensive error checking
- Clear success/failure messages
- Prerequisites validation

## Expected Performance

### Enhanced Performance Improvements
- **Better Convergence**: Early stopping prevents overfitting
- **Improved Accuracy**: Deeper model + attention pooling
- **Ordinal Consistency**: CORAL loss respects class ordering
- **Class Balance**: Focal loss handles imbalanced data
- **Training Stability**: Layer norm + reduced learning rate

### Computational Requirements
- **GPU Memory**: ~101.5 GB available
- **Batch Size**: 128 (optimized for stability)
- **Training Time**: ~3-6 hours (with early stopping)
- **Embedding Extraction**: ~1-2 hours
- **Donor Classification**: ~30 minutes
- **Total Pipeline**: ~4-8 hours (depending on early stopping)

### Memory Usage
- **Transformer Training**: ~0.1 GB per batch
- **Embedding Extraction**: ~2-4 GB peak
- **Donor Classification**: ~1 GB

## Usage Examples

### 🚀 **Recommended: Enhanced Pipeline**

```bash
# Complete pipeline with all enhancements
./start_real_data_pipeline.sh start-all

# Choose execution mode:
#   1) Sequential (wait for completion)
#   2) Background (monitor in background)
```

### 🔧 **Manual Execution (Advanced)**

```bash
# Step-by-step execution
./start_real_data_pipeline.sh start-1-2    # Steps 1-2
./start_real_data_pipeline.sh start-3-4    # Steps 3-4 (after 1-2 complete)

# Or use individual scripts directly
./run_real_data_steps_1_2.sh --num-gpus 1
./run_real_data_steps_3_4.sh --num-gpus 1
```

### 📊 **Monitoring and Management**

```bash
# Check overall status
./start_real_data_pipeline.sh status

# Real-time monitoring
./start_real_data_pipeline.sh monitor

# View specific logs
tail -f real_data_steps_1_2.log
tail -f real_data_steps_3_4.log

# Attach to running sessions
screen -r real_data_steps_1_2
screen -r real_data_steps_3_4
```

### 🛑 **Stopping the Pipeline**

```bash
# Stop specific steps
screen -S real_data_steps_1_2 -X quit
screen -S real_data_steps_3_4 -X quit

# Stop background monitor
screen -S pipeline_monitor -X quit
```

## Execution Order

### Enhanced Mode (Automatic)
1. **Single Command**: `./start_real_data_pipeline.sh start-all`
2. **Automatic Sequencing**: 1→2→3→4 with dependency management
3. **Background Monitoring**: Optional non-blocking execution

### Manual Mode (Step-by-step)
1. **First**: Run `./run_real_data_steps_1_2.sh` (or `./start_real_data_pipeline.sh start-1-2`)
2. **Then**: Run `./run_real_data_steps_3_4.sh` (or `./start_real_data_pipeline.sh start-3-4`)

## Monitoring Progress

Both scripts provide detailed progress information:
- Training progress bars
- Loss and accuracy metrics
- GPU memory usage
- Step completion status

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size to 128 or 64
2. **Missing Files**: Ensure Script 1 completes before Script 2
3. **GPU Not Found**: Check CUDA installation and GPU availability

### Performance Optimization
- Use multiple GPUs for faster training
- Adjust `num_workers` based on CPU cores
- Monitor GPU memory usage during execution

## Results Interpretation

The final results will include:
- **Test Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **QWK**: Quadratic Weighted Kappa for ordinal classification
- **Train/Test Donor Counts**: Data split information

## Quick Reference Guide

### 🚀 **Most Common Commands**

```bash
# Start complete pipeline (recommended)
./start_real_data_pipeline.sh start-all

# Monitor progress
./start_real_data_pipeline.sh monitor

# Check status
./start_real_data_pipeline.sh status

# View help
./start_real_data_pipeline.sh help
```

### 📊 **Monitoring Commands**

```bash
# Real-time monitoring with system resources
./start_real_data_pipeline.sh monitor

# Direct session access
screen -r real_data_steps_1_2      # Steps 1-2
screen -r real_data_steps_3_4      # Steps 3-4
screen -r pipeline_monitor         # Background monitor

# View logs
tail -f real_data_steps_1_2.log
tail -f real_data_steps_3_4.log
```

### 🛑 **Management Commands**

```bash
# Stop all sessions
screen -S real_data_steps_1_2 -X quit
screen -S real_data_steps_3_4 -X quit
screen -S pipeline_monitor -X quit

# List all screen sessions
screen -list

# Kill all screen sessions
screen -wipe
```

### 🔧 **Troubleshooting Commands**

```bash
# Check GPU availability
nvidia-smi

# Check disk space
df -h

# Check memory usage
free -h

# Check running processes
ps aux | grep python
```

## Summary

The enhanced DREAM pipeline provides:

- ✅ **One-Command Execution**: `./start_real_data_pipeline.sh start-all`
- ✅ **Automatic Sequencing**: Proper 1→2→3→4 dependency management
- ✅ **Enhanced Performance**: Early stopping, ordinal regression, focal loss
- ✅ **Flexible Execution**: Sequential or background modes
- ✅ **Real-time Monitoring**: Progress tracking and system resources
- ✅ **Error Handling**: Comprehensive validation and error reporting

**Recommended Usage**: Use the enhanced mode (`./start_real_data_pipeline.sh start-all`) for the best experience with all performance improvements and automated execution.
