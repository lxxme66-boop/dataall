#!/bin/bash

# 半导体QA生成系统 - 增强版运行脚本
# 保持原有生成逻辑，添加断点、火山API、多线程支持

echo "=================================================="
echo "半导体QA生成系统 - 增强版"
echo "保持原有流程(1.1-3.1)，添加新功能："
echo "1. 断点续跑"
echo "2. 火山API支持（可选）"
echo "3. 多线程并行处理"
echo "=================================================="

# 默认参数
CONFIG_FILE="config_enhanced.json"
MAX_WORKERS=4
CHECKPOINT_DIR="checkpoints"
USE_VOLCANO=false
VOLCANO_API_KEY=""
RESUME=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --use-volcano)
            USE_VOLCANO=true
            shift
            ;;
        --volcano-api-key)
            VOLCANO_API_KEY="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "使用方法："
            echo "  $0 [选项]"
            echo ""
            echo "选项："
            echo "  --config FILE           配置文件路径 (默认: config_enhanced.json)"
            echo "  --max-workers N         最大工作线程数 (默认: 4)"
            echo "  --use-volcano           使用火山API"
            echo "  --volcano-api-key KEY   火山API密钥"
            echo "  --resume                从断点恢复"
            echo "  --checkpoint-dir DIR    断点目录 (默认: checkpoints)"
            echo "  --help                  显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 创建必要的目录
mkdir -p data/output
mkdir -p "$CHECKPOINT_DIR"
mkdir -p logs

# 设置日志文件
LOG_FILE="logs/enhanced_run_$(date +%Y%m%d_%H%M%S).log"

# 构建Python命令
PYTHON_CMD="python run_semiconductor_qa_enhanced.py"
PYTHON_CMD="$PYTHON_CMD --config $CONFIG_FILE"
PYTHON_CMD="$PYTHON_CMD --max-workers $MAX_WORKERS"
PYTHON_CMD="$PYTHON_CMD --checkpoint-dir $CHECKPOINT_DIR"

if [ "$USE_VOLCANO" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --use-volcano"
    if [ -n "$VOLCANO_API_KEY" ]; then
        PYTHON_CMD="$PYTHON_CMD --volcano-api-key $VOLCANO_API_KEY"
    fi
fi

if [ "$RESUME" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --resume"
fi

# 显示运行配置
echo ""
echo "运行配置："
echo "  配置文件: $CONFIG_FILE"
echo "  最大线程数: $MAX_WORKERS"
echo "  使用火山API: $USE_VOLCANO"
echo "  断点恢复: $RESUME"
echo "  断点目录: $CHECKPOINT_DIR"
echo "  日志文件: $LOG_FILE"
echo ""

# 运行主程序
echo "开始执行..."
echo "命令: $PYTHON_CMD"
echo ""

# 执行并记录日志
$PYTHON_CMD 2>&1 | tee "$LOG_FILE"

# 检查执行结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "执行成功完成！"
    echo "日志已保存到: $LOG_FILE"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "执行过程中出现错误"
    echo "请检查日志文件: $LOG_FILE"
    echo "可以使用 --resume 参数从断点恢复"
    echo "=================================================="
    exit 1
fi