# CellFM Fine-tune 设置说明

## 当前状态

✅ **已完成:**
1. 克隆了 CellFM-torch 仓库到 `/home/ubuntu/LLM-inference/xinze-project/cellfm/`
2. 分析了 CellFM 的结构和 fine-tune 方法
3. 创建了 CellFM 微调脚本: `scripts/train_cellfm_adnc.py`
4. 创建了运行脚本: `run_cellfm_finetune.sh`

⚠️ **待处理:**
1. 需要下载 CellFM 预训练权重 (80M)
2. 需要配置 CellFM 的数据格式要求
3. 测试运行脚本

## CellFM 结构分析

### 模型架构
- **基础模型**: CellFM 80M (预训练权重)
- **微调架构**: `Finetune_Cell_FM` 
  - 使用预训练的 CellFM 作为特征提取器
  - 添加分类头用于 ADNC 预测
  - 只训练分类头和encoder部分

### 数据要求
1. **h5ad 文件格式**:
   - 必须包含列: `celltype`, `feat` (类别标签), `batch_id`
   - `celltype`: 细胞类型信息
   - `feat`: ADNC 类别 (0-3 对应 Not AD, Low, Intermediate, High)
   - `train`: 0 (训练) / 2 (测试)

2. **基因映射**:
   - CellFM 需要映射到其基因词汇表 (27,855 个基因)
   - 使用 `expand_gene_info.csv` 进行映射
   - 自动处理基因别名匹配

3. **批次设置**:
   - 每个 batch 需要 `batch_id`
   - 训练时使用 `batch_size` 参数控制

## 使用方法

### 1. 下载预训练权重

```bash
# 从 HuggingFace 下载 CellFM 80M 权重
# https://huggingface.co/ShangguanNingyuan/CellFM/tree/main

# 或者如果已经下载，保存到:
# /home/ubuntu/LLM-inference/xinze-project/cellfm/CellFM_80M_weight.ckpt
```

### 2. 运行 Fine-tune

**测试数据 (8 donors):**
```bash
cd /home/ubuntu/LLM-inference/xinze-project/dream_test
./run_cellfm_finetune.sh --testcase
```

**实际数据 (MTG):**
```bash
cd /home/ubuntu/LLM-inference/xinze-project/dream_test
./run_cellfm_finetune.sh --mtg
```

### 3. 检查结果

训练完成后，结果保存在:
- 测试数据: `dream_test/outputs/cellfm_testcase/`
- MTG数据: `dream_test/outputs/cellfm_mtg/`

包含:
- `best_model.pth`: 最佳模型权重
- `checkpoint_epoch_X.pth`: 每个epoch的checkpoint
- `training_results.json`: 训练历史和准确率

## 关键配置

### 超参数 (Test Case)
```python
batch_size = 16
epochs = 3
lr = 1e-4
num_classes = 4
```

### 超参数 (MTG)
```python
batch_size = 8  # 更大数据集，使用较小batch
epochs = 5
lr = 1e-4
num_classes = 4
```

### 模型参数
```python
enc_dims = 1536        # 编码器维度
enc_nlayers = 2        # 编码器层数
enc_num_heads = 48     # 注意力头数
```

## 与现有 DREAM Pipeline 的差异

| 特性 | DREAM Pipeline | CellFM |
|------|---------------|--------|
| 模型架构 | 自定义 Transformer | 预训练 CellFM 80M |
| 初始化 | 随机初始化 | 预训练权重 |
| 数据格式 | 可变长度序列 | 固定长度 (2048 基因) |
| 基因映射 | 直接使用 | 需要映射到固定词汇表 |
| 训练方式 | 从头训练 | 微调预训练模型 |

## 注意事项

1. **数据兼容性**: 
   - CellFM 需要特定的数据预处理步骤
   - 需要将当前数据格式转换为 CellFM 格式
   
2. **基因对齐**:
   - CellFM 使用 27,855 个基因的固定词汇表
   - 需要映射当前数据的基因名到 CellFM 的基因ID
   
3. **批次大小**:
   - 预训练模型权重较大，建议使用较小的 batch_size
   - MTG 数据使用 batch_size=8
   
4. **预训练权重**:
   - 需要从 HuggingFace 下载 CellFM_80M_weight.ckpt
   - 权重文件约 320MB

## 下一步

1. **下载预训练权重**: 从 HuggingFace 获取 CellFM 80M 权重
2. **修复数据兼容性**: 确保我们的 h5ad 数据能被 CellFM 正确加载
3. **测试运行**: 先在测试数据上验证流程
4. **MTG 数据训练**: 在完整数据上微调并评估准确率

## 相关文件

- `/home/ubuntu/LLM-inference/xinze-project/cellfm/`: CellFM 源码
- `/home/ubuntu/LLM-inference/xinze-project/dream_test/scripts/train_cellfm_adnc.py`: 微调脚本
- `/home/ubuntu/LLM-inference/xinze-project/dream_test/run_cellfm_finetune.sh`: 运行脚本

## 参考资源

- CellFM-torch GitHub: https://github.com/biomed-AI/CellFM-torch
- CellFM HuggingFace: https://huggingface.co/ShangguanNingyuan/CellFM
- 数据路径: `/home/ubuntu/LLM-inference/xinze-project/ADNC_Pred/training_data/`




