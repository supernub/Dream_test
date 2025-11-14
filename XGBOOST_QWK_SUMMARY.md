# XGBoost 和 QWK 相关代码总结

## 主要文件

### 1. `/home/ubuntu/LLM-inference/xinze-project/dream_test/scripts/donor_classifier.py`

这是最主要的文件，包含了：
- **XGBoost 分类器实现**
- **QWK (Quadratic Weighted Kappa) 评估指标**
- **多种损失函数**（包括 QWK 损失）

#### XGBoost 相关

```python
import xgboost as xgb

# XGBoost 训练函数 (行265-301)
def train_xgboost_classifier(train_embeddings, train_labels, val_embeddings, val_labels):
    # 准备数据
    dtrain = xgb.DMatrix(train_embeddings, label=train_labels)
    dval = xgb.DMatrix(val_embeddings, label=val_labels)
    
    # 训练参数
    params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'mlogloss',
    }
    
    # 自定义 QWK 评估函数
    def qwk_eval(y_pred, y_true):
        y_true = y_true.get_label()
        y_pred_class = np.argmax(y_pred.reshape(-1, 4), axis=1)
        qwk = cohen_kappa_score(y_true, y_pred_class, weights='quadratic')
        return 'qwk', qwk
    
    # 训练
    model = xgb.train(
        params,
        dtrain,
        evals=[(dtrain, 'train'), (dval, 'val')],
        feval=qwk_eval,
        maximize=True,
        early_stopping_rounds=20,
        num_boost_round=200,
        verbose_eval=10
    )
```

#### QWK 相关

```python
from sklearn.metrics import cohen_kappa_score

# QWK 评估
val_qwk = cohen_kappa_score(val_labels, val_preds, weights='quadratic')

# QWK 损失函数 (行127-147)
def qwk_loss(logits, targets, reduction="mean"):
    """
    Quadratic Weighted Kappa loss for ordinal regression.
    This directly optimizes for QWK metric by penalizing predictions 
    based on their distance from true labels.
    """
    B, Km1 = logits.shape
    num_classes = Km1 + 1
    
    # Convert logits to probabilities using sigmoid
    probs = torch.sigmoid(logits)  # [B, K-1]
    
    # ... QWK 损失计算逻辑 ...
```

## 相关文件

### 2. `/home/ubuntu/LLM-inference/xinze-project/dream_test/scripts/test_case_example.py`
- 包含 QWK 测试用例

### 3. `/home/ubuntu/LLM-inference/xinze-project/dream_test/scripts/run_full_pipeline.py`
- 包含 QWK 评估

### 4. `/home/ubuntu/LLM-inference/xinze-project/dream_test/README.md`
- 文档说明

## 使用方式

### 运行 donor classifier
```bash
python scripts/donor_classifier.py \
    --embedding_dir /path/to/embeddings \
    --output_dir /path/to/output \
    --loss_function qwk
```

### 关键参数
- `--loss_function`: 可选择 "coral", "emd", "qwk", "ordinal_mae"
- `--use_class_weights`: 使用类别权重
- `--use_balanced_sampling`: 平衡采样

## 评估指标

donor_classifier.py 中包含了以下评估指标：
- Accuracy
- F1 Score
- **Quadratic Weighted Kappa (QWK)**
- Cohen Kappa Score

## 代码行数统计

- XGBoost 相关代码：约 40 行 (行265-301)
- QWK 相关代码：约 50 行 (行127-147, 265-270, 752, 817-822)
- 总文件大小：约 41KB



