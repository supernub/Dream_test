# 模型结果总结（Oligodendrocyte Not AD vs High）

## 数据条件
- 数据集: `data/top4_binary/Oligodendrocyte`
- Split: donor-level（使用 `split_info.json`）
- 特征:
  - 基因表达 (36,601 genes)
  - Donor metadata：仅 `log_cell_count` + 自动提取的分类统计
  - **严格排除** ADNC stage 信息，防止数据泄露
- 训练集: 44,442 细胞（High 74.39%，Not AD 25.61%）
- 测试集: 15,602 细胞（High 86.94%，Not AD 13.06%）

## XGBoost（推荐方案）
- 配置: `max_depth=3`, `eta=0.03`, `scale_pos_weight` 自动计算, 正则化增强
- 阈值: 0.5（验证集最优阈值 0.41 在测试集表现差）
- 测试集表现:
  - Accuracy: **0.7729**
  - Precision: 0.9214
  - Recall: 0.8075
  - F1: 0.8605
  - ROC-AUC: 0.5728
  - **QWK: 0.2581**
  - 混淆矩阵 (Not AD / High): TN=1102, FP=935, FN=2612, TP=10953
- 说明: 能有效区分两类，相比基线 QWK=0.0207 提升 12.5 倍

## AutoGluon（同等数据条件）
- 配置: `presets=high_quality`, `time_limit=3600s`, bagging=5, stacking=1, `eval_metric=roc_auc`
- 最佳模型: LightGBM_BAG_L1_FULL（自动堆叠）
- 测试集表现:
  - Accuracy: 0.4863
  - Precision: 0.7883
  - Recall: 0.5593
  - F1: 0.6544
  - ROC-AUC: 0.1825
  - **QWK: -0.2419**
- 问题定位:
  - Train QWK = 1.0, Test QWK = -0.24 → 严重过拟合
  - ROC-AUC 低于随机 (0.1825)
  - Accuracy 低于多数类比例（模型预测偏向少数类）

## 结论
1. 在当前数据处理和 split 条件下，**XGBoost** 明显优于 AutoGluon。
2. AutoGluon 需额外调参（阈值优化、改用 F1/平衡准确率、加大正则等）才能避免过拟合。
3. 推荐继续使用 XGBoost + Gene + Donor Metadata（不含 ADNC）方案。

---

## CellFM CUDA 配置说明
- 环境: `cellfm_cuda` / `new_cellfm_finetune`
- 硬件: NVIDIA GPU + CUDA 12.8，操作系统为 ARM (aarch64)
- 问题: PyTorch 官方未提供 ARM CUDA 支持的预编译包
- 结论: 虽然系统有 CUDA，但 PyTorch 仅能使用 CPU，无法在 GPU 上跑 CellFM。运行时会显示 “torch.cuda.is_available() = False”
- 临时方案: 使用 CPU 训练（非常慢），或改用 x86-64 架构服务器
