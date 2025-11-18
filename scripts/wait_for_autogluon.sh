#!/bin/bash
# 等待 AutoGluon 训练完成并显示结果

LOG_FILE="/tmp/autogluon_training.log"
OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/dream_test/output/autogluon/Oligodendrocyte_gene_donor_metadata_no_adnc"
METRICS_FILE="$OUTPUT_DIR/metrics.json"

echo "=========================================="
echo "等待 AutoGluon 训练完成..."
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查训练进程是否还在运行
check_process() {
    ps aux | grep -E "autogluon|python.*autogluon" | grep -v grep > /dev/null 2>&1
    return $?
}

# 检查是否完成
check_complete() {
    if [ -f "$METRICS_FILE" ]; then
        return 0
    else
        return 1
    fi
}

# 等待循环
while true; do
    if check_complete; then
        echo ""
        echo "=========================================="
        echo "✓ 训练已完成！"
        echo "=========================================="
        echo ""
        echo "=== 测试集评估结果 ==="
        if command -v python3 > /dev/null; then
            python3 << EOF
import json
import sys

try:
    with open('$METRICS_FILE', 'r') as f:
        metrics = json.load(f)
    
    print("\n训练集指标:")
    train = metrics.get('train', {})
    for key, value in sorted(train.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n测试集指标:")
    test = metrics.get('test', {})
    for key, value in sorted(test.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # QWK 对比
    if 'qwk' in test:
        print("\n" + "="*60)
        print("QWK (Quadratic Weighted Kappa) 结果:")
        print("="*60)
        print(f"Train QWK: {train.get('qwk', 'N/A')}")
        print(f"Test QWK:  {test.get('qwk', 'N/A')}")
        print("="*60)
        
        # 与 XGBoost 对比
        xgboost_qwk = 0.2581
        autogluon_qwk = test.get('qwk', 0)
        print(f"\n与 XGBoost 对比 (Gene + Donor Metadata, 不含 ADNC):")
        print(f"  XGBoost Test QWK: {xgboost_qwk:.6f}")
        print(f"  AutoGluon Test QWK: {autogluon_qwk:.6f}")
        if autogluon_qwk > xgboost_qwk:
            improvement = ((autogluon_qwk - xgboost_qwk) / xgboost_qwk) * 100
            print(f"  ✓ AutoGluon 提升了 {improvement:.2f}%")
        elif autogluon_qwk < xgboost_qwk:
            decrease = ((xgboost_qwk - autogluon_qwk) / xgboost_qwk) * 100
            print(f"  ✗ AutoGluon 降低了 {decrease:.2f}%")
        else:
            print(f"  = 性能相同")
        
except Exception as e:
    print(f"Error reading metrics: {e}")
    sys.exit(1)
EOF
        else
            cat "$METRICS_FILE"
        fi
        
        echo ""
        echo "=== 详细结果文件 ==="
        echo "Metrics: $METRICS_FILE"
        if [ -f "$OUTPUT_DIR/leaderboard.csv" ]; then
            echo "Leaderboard: $OUTPUT_DIR/leaderboard.csv"
        fi
        echo ""
        echo "=========================================="
        exit 0
    fi
    
    if ! check_process; then
        # 进程已停止但结果文件不存在，可能出错
        if ! check_complete; then
            echo ""
            echo "⚠ 警告: 训练进程已停止，但未找到结果文件"
            echo "请检查日志: tail -100 $LOG_FILE"
            exit 1
        fi
    fi
    
    # 每30秒检查一次
    sleep 30
done

