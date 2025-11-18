# CellFM CUDA 支持研究 - ARM 架构 (GH200)

## 当前状态

### 系统信息
- **GPU**: NVIDIA GH200 480GB
- **CUDA 版本**: 12.8.61
- **驱动版本**: 570.124.06
- **架构**: ARM64 (aarch64)
- **Compute Capability**: 9.0

### PyTorch 状态
- **当前版本**: 2.0.1 (CPU only)
- **CUDA 支持**: ❌ 不可用
- **CUDA 编译**: None
- **问题**: PyTorch 官方未提供 ARM 架构的 CUDA 构建

### CUDA 库检查
- ✅ CUDA runtime 库存在: `/usr/lib/aarch64-linux-gnu/libcudart.so.12.8.57`
- ✅ CUDA 路径存在: `/usr/local/cuda`, `/usr/lib/cuda`
- ❌ PyTorch 无法链接到 CUDA 库（因为是 CPU 版本）

## 问题分析

### 根本原因
PyTorch 官方目前**不提供** ARM 架构 (aarch64) 的 CUDA 支持构建。所有从官方源安装的 PyTorch 都是 CPU 版本。

### 验证方法
```bash
# 检查 PyTorch 版本
python -c "import torch; print(torch.__version__)"  # 输出: 2.0.1+cpu

# 检查 CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"  # 输出: False
```

## 可能的解决方案

### 方案 1: 等待官方支持 ⏳
- **状态**: 等待 PyTorch 官方发布 ARM CUDA 构建
- **时间**: 未知
- **优点**: 最简单，无需额外工作
- **缺点**: 可能需要等待较长时间

### 方案 2: 从源码编译 PyTorch 🔧
- **难度**: ⭐⭐⭐⭐⭐ (非常困难)
- **要求**:
  - CUDA Toolkit 12.8
  - cuDNN
  - 大量编译时间（可能需要数小时）
  - 足够的磁盘空间（>50GB）
- **步骤**:
  1. 克隆 PyTorch 源码
  2. 配置编译选项（启用 CUDA，ARM 架构）
  3. 编译（可能需要数小时）
  4. 安装编译好的包

**参考命令**:
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST="9.0"  # GH200 compute capability
python setup.py install
```

### 方案 3: 使用 NVIDIA 官方构建（如果有）🔍
- **检查**: NVIDIA 可能为 GH200 提供特殊构建
- **查找位置**:
  - NVIDIA Developer Portal
  - NVIDIA Container Registry
  - GH200 专用软件仓库

### 方案 4: 使用 Docker 容器 🐳
- **前提**: 需要 NVIDIA Container Toolkit
- **查找**: NVIDIA 官方或社区维护的 ARM CUDA PyTorch 镜像
- **命令示例**:
```bash
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
```

### 方案 5: 使用替代框架 ⚠️
- **JAX**: 可能对 ARM CUDA 有更好的支持
- **TensorFlow**: 检查是否有 ARM CUDA 构建
- **限制**: 需要修改 CellFM 代码

## 当前可行的方案

### 临时方案: CPU 模式运行
- **状态**: ✅ 已实现
- **性能**: 慢（每个 batch ~12.8 秒）
- **适用**: 测试脚本逻辑，小规模实验

### 优化建议
1. **减小 batch size**: 当前 8，可以尝试 4
2. **减少 epochs**: 当前 3，可以尝试 1-2
3. **增加 eval_interval**: 减少评估频率
4. **使用混合精度**: 虽然 CPU 不支持，但可以优化代码

## 环境变量设置（如果未来有 CUDA 支持）

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

## 检查清单

- [x] CUDA 驱动已安装
- [x] CUDA runtime 库存在
- [x] PyTorch 已安装（CPU 版本）
- [ ] PyTorch CUDA 支持（需要官方或自定义构建）
- [ ] 验证 GPU 可用性

## 下一步行动

1. **监控 PyTorch 官方更新**: 关注 ARM CUDA 支持进展
2. **联系 NVIDIA 支持**: 询问 GH200 的 PyTorch CUDA 构建
3. **考虑从源码编译**: 如果有足够时间和资源
4. **继续使用 CPU 模式**: 用于开发和测试

## 参考资源

- PyTorch GitHub: https://github.com/pytorch/pytorch
- NVIDIA Developer Forums: https://forums.developer.nvidia.com/
- PyTorch Installation Guide: https://pytorch.org/get-started/locally/

## 更新日志

- 2025-11-18: 初始研究，确认 PyTorch 官方无 ARM CUDA 构建
- 2025-11-18: 验证 CUDA 库存在，但 PyTorch 无法链接

