# CellFM CUDA 环境配置说明

## 环境信息

- **环境名称**: `cellfm_cuda`
- **Python 版本**: 3.10.19
- **系统架构**: ARM64 (aarch64) - NVIDIA GH200

## 已安装的包

- **PyTorch**: 2.0.1 (CPU版本，ARM架构暂不支持CUDA)
- **MindSpore**: 2.7.1 (用于加载CellFM预训练权重)
- **Scanpy**: 1.11.5
- **Pandas**: 2.3.3
- **NumPy**: 1.26.4
- **Scikit-learn**: 1.7.2
- **其他依赖**: anndata, h5py, tqdm, scipy

## CUDA 状态

⚠️ **当前状态**: PyTorch 未检测到 CUDA 支持

**原因**: 
- 系统是 ARM 架构 (aarch64)
- PyTorch 官方对 ARM 架构的 CUDA 支持有限
- GH200 可能需要特殊的 PyTorch 构建

**解决方案**:
1. 等待 NVIDIA 官方发布支持 ARM CUDA 的 PyTorch 构建
2. 从源码编译 PyTorch（需要 CUDA toolkit）
3. 暂时使用 CPU 模式运行（会很慢，但可以测试脚本逻辑）

## 使用方法

### 激活环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellfm_cuda
```

### 运行 CellFM 训练

```bash
cd /home/ubuntu/LLM-inference/xinze-project/dream_test

python scripts/train_cellfm_binary.py \
  --h5ad_path data/top4_binary/Oligodendrocyte/cells_subset.h5ad \
  --split_json data/top4_binary/Oligodendrocyte/split_info.json \
  --output_dir output/cellfm/Oligodendrocyte_donor_metadata \
  --ckpt_path /home/ubuntu/LLM-inference/xinze-project/cellfm/CellFM_80M_weight.ckpt \
  --device cpu \
  --batch_size 8 \
  --epochs 3 \
  --lr 1e-4 \
  --num_cls 2 \
  --label-column binary_label \
  --concat-donor-metadata \
  --eval_interval_steps 50 \
  --patience_steps 3
```

**注意**: 
- 如果 CUDA 可用，脚本会自动使用 `cuda:0`
- 如果 CUDA 不可用，会自动切换到 `cpu` 模式
- CPU 模式下建议减小 `batch_size` 和 `epochs`

## 检查 CUDA 支持

```bash
conda activate cellfm_cuda
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 未来改进

如果 NVIDIA 发布了支持 ARM CUDA 的 PyTorch 构建，可以：

```bash
conda activate cellfm_cuda
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url <NVIDIA官方ARM CUDA构建URL>
```

