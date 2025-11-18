#!/bin/bash
# AutoGluon 训练实时监控脚本

LOG_FILE="/tmp/autogluon_training.log"
OUTPUT_DIR="/home/ubuntu/LLM-inference/xinze-project/dream_test/output/autogluon/Oligodendrocyte_gene_donor_metadata_no_adnc"

echo "=========================================="
echo "AutoGluon 训练实时监控"
echo "=========================================="
echo ""

# 检查训练进程
check_process() {
    if ps aux | grep -E "autogluon|python.*autogluon" | grep -v grep > /dev/null; then
        return 0
    else
        return 1
    fi
}

# 显示训练进度摘要
show_progress() {
    echo "=== 当前时间: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo ""
    
    if check_process; then
        echo "✓ 训练进程运行中"
        echo ""
        
        # 显示最新进度
        echo "=== 最新训练日志 (最后 20 行) ==="
        if [ -f "$LOG_FILE" ]; then
            tail -20 "$LOG_FILE" | grep -E "(Beginning AutoGluon|Train Data|Problem Type|Fitting|Training|model|Best|leaderboard|QWK|Test QWK|测试集评估|Completed|Time limit|Finished|score_val|score_test)" || tail -10 "$LOG_FILE"
        fi
        echo ""
        
        # 检查是否完成
        if [ -f "$OUTPUT_DIR/metrics.json" ]; then
            echo "✓ 训练已完成！结果已保存"
            echo ""
            echo "=== 测试集结果 ==="
            cat "$OUTPUT_DIR/metrics.json" | python3 -m json.tool 2>/dev/null | grep -A 10 '"test"'
            echo ""
            echo "=== QWK 结果 ==="
            cat "$OUTPUT_DIR/metrics.json" | python3 -m json.tool 2>/dev/null | grep -E '"qwk"'
            exit 0
        fi
        
    else
        echo "✗ 训练进程未运行"
        if [ -f "$OUTPUT_DIR/metrics.json" ]; then
            echo "✓ 但结果文件存在，训练可能已完成"
            echo ""
            echo "=== 最终结果 ==="
            cat "$OUTPUT_DIR/metrics.json" | python3 -m json.tool 2>/dev/null
            exit 0
        fi
        exit 1
    fi
    
    echo "=========================================="
    echo ""
}

# 监控循环
if [ "$1" = "watch" ]; then
    while true; do
        clear
        show_progress
        sleep 30
    done
else
    show_progress
    echo ""
    echo "提示: 使用 '$0 watch' 进行实时监控（每30秒刷新）"
fi

