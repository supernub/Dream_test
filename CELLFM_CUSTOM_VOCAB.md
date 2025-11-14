# CellFM Fine-tune with Custom Gene Embedding

## 问题说明

### 问题
CellFM 使用的是固定词汇表（27,855个基因），而我们的数据使用的是不同的基因集合。这导致了以下问题：

1. **Gene embedding 维度不匹配**: CellFM的 `gene_emb` 层是预训练好的，维度为 `(27856, 1536)`
2. **基因索引不匹配**: 即使维度相同，基因的对应关系也可能不对
3. **直接加载会导致错误**: 无法直接使用预训练权重

### 解决方案

**老师的建议**（核心要点）:
> "finetune cellfm 有一个要点，需要特别注意，就是 gene embedding 第一层是有问题的，就是 vocab 和我们的数据不一样。为了简单期间，你需要重新 init 一个我们 data vocab 的 gene embedding，然后其他 cellfm 的 layer 要 init with original weight，只 load 除了 emb 以外的部分"

**实现方式**:
1. ✅ **Gene embedding**: 重新初始化，使用我们的基因数量
2. ✅ **其他层**: 加载预训练权重（encoder, decoder 等）
3. ✅ **训练**: 只训练可训练的参数（cls 和 encoder）

## 修改内容

### 1. 输出目录

**修改前**:
```bash
OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/dream_test/outputs/cellfm_..."
```

**修改后**:
```bash
OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_..."
```

输出将统一放在 `/outputs/` 目录下，与其他实验结果一致。

### 2. Gene Embedding 处理

创建了新脚本: `train_cellfm_adnc_custom.py`

**关键修改**:

```python
# 1. 重新初始化 gene embedding
print("Re-initializing gene embedding...")
with torch.no_grad():
    emb_dim = net.extractor.net.gene_emb.shape[1]
    
    # 创建新的 embedding（根据我们的基因数量）
    n_genes = len(our_genes)  # 从我们的数据获取
    pad_to = ((n_genes - 1) // 8 + 1) * 8  # CellFM 需要8的倍数
    
    net.extractor.net.gene_emb = nn.Parameter(
        torch.empty(pad_to, emb_dim)
    )
    nn.init.xavier_normal_(net.extractor.net.gene_emb)
    net.extractor.net.gene_emb.data[0, :] = 0  # pad token 始终为0

# 2. 加载预训练权重（排除 gene embedding）
def load_pretrained_weights_without_gene_emb(model, ckpt_path):
    ms_ckpt = load_checkpoint(ckpt_path)
    
    for ms_key, ms_param in ms_ckpt.items():
        pt_key = map_ms_to_pt(ms_key)
        
        # 跳过 gene embedding
        if 'gene_emb' in pt_key:
            print(f"Skipping gene embedding: {pt_key}")
            continue
        
        torch_state_dict[pt_key] = torch.tensor(ms_param.asnumpy())
    
    model.extractor.net.load_state_dict(torch_state_dict, strict=False)
```

### 3. 训练参数设置

```python
# 只训练分类头和encoder
for name, param in net.named_parameters():
    param.requires_grad = "cls." in name or "encoder" in name
```

- ✅ **训练**: `cls` 分类器、`encoder` 层
- ❌ **冻结**: `gene_emb` (新初始化的)、其他预训练层

## 使用方法

### 1. 准备预训练权重

下载 CellFM 80M 权重并保存到:
```bash
/home/ubuntu/LLM-inference/xinze-project/cellfm/CellFM_80M_weight.ckpt
```

### 2. 运行 Fine-tune

**测试数据 (8 donors):**
```bash
cd /home/ubuntu/LLM-inference/xinze-project/dream_test
./run_cellfm_finetune.sh --testcase
```

**MTG 数据:**
```bash
cd /home/ubuntu/LLM-inference/xinze-project/dream_test
./run_cellfm_finetune.sh --mtg
```

### 3. 查看结果

**测试数据输出**:
```
/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_testcase/
├── best_model.pth              # 最佳模型
├── checkpoint_epoch_1.pth      # Epoch checkpoints
├── training_results.json       # 训练历史
├── train.h5ad                  # 训练数据
└── test.h5ad                   # 测试数据
```

**MTG 数据输出**:
```
/home/ubuntu/LLM-inference/xinze-project/outputs/cellfm_mtg/
├── best_model.pth
├── ...
```

## 技术细节

### Gene Embedding 维度

1. **CellFM 原始**: `(27856, 1536)`
2. **我们的数据**: 假设 `n_genes` 个基因
3. **实际维度**: `(pad_to, 1536)` 其中 `pad_to = ((n_genes - 1) // 8 + 1) * 8`

### 为什么需要 padding？

CellFM 的代码要求基因数量是8的倍数，因为使用了某种对齐优化。

### 权重加载逻辑

```python
# 加载顺序
1. 初始化整个模型
2. 重新初始化 gene_emb (我们的自定义实现)
3. 加载预训练权重（排除 gene_emb）
4. 设置可训练参数
5. 开始训练
```

### 训练策略

- **Fine-tune**: 只训练分类器和部分 encoder
- **Freeze**: gene embedding（新初始化）+ 其他预训练层
- **Reason**: 保留预训练知识，学习领域特定的特征

## 预期结果

### 训练指标

```json
{
  "train_acc_history": [0.25, 0.35, ...],
  "test_acc_history": [0.28, 0.32, ...],
  "best_test_acc": 0.45,
  "n_genes": 36601,
  "gene_embedding": "reinitialized_for_our_vocab"
}
```

### 性能期望

- **合理范围**: 40-60% accuracy（取决于数据质量）
- **对比基准**: 可以与 DREAM pipeline 的 58.8% 对比
- **优势**: 使用预训练模型，可能学到更好的细胞特征

## 故障排除

### 问题1: Gene embedding 维度错误
**症状**: `RuntimeError: size mismatch`
**解决**: 检查 `pad_to` 计算是否正确

### 问题2: 无法加载预训练权重
**症状**: `KeyError: xxx not in state_dict`
**解决**: 确保 MindSpore checkpoint 路径正确

### 问题3: 内存不足
**症状**: `CUDA out of memory`
**解决**: 减小 `batch_size` (如从16减到8)

## 总结

✅ **输出目录**: 已改为 `/outputs/` 统一管理
✅ **Gene embedding**: 重新初始化适配我们的数据
✅ **其他层**: 加载预训练权重
✅ **训练策略**: 只训练必要部分
✅ **代码**: 已创建 `train_cellfm_adnc_custom.py`

现在可以直接运行测试了！




