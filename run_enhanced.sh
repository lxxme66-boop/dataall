#!/bin/bash

# 半导体QA生成系统 - 增强版运行脚本
# 支持断点续传、火山API和多线程

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
CONFIG_FILE="config.json"
MODEL="deepseek-r1"
QUALITY_THRESHOLD=0.7
BATCH_SIZE=16
MAX_WORKERS=8
CHECKPOINT_DIR="checkpoints"
USE_VOLCANO=false
VOLCANO_API_KEY=""
RESUME=false
RESUME_FROM=""

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --input-dir DIR          输入目录 (默认: data/input)"
    echo "  --output-dir DIR         输出目录 (默认: data/output)"
    echo "  --config FILE           配置文件 (默认: config.json)"
    echo "  --model NAME            模型名称 (默认: deepseek-r1)"
    echo "  --quality-threshold NUM  质量阈值 (默认: 0.7)"
    echo "  --batch-size NUM        批处理大小 (默认: 16)"
    echo "  --max-workers NUM       最大线程数 (默认: 8)"
    echo "  --checkpoint-dir DIR    断点目录 (默认: checkpoints)"
    echo "  --use-volcano          使用火山API"
    echo "  --volcano-api-key KEY   火山API密钥"
    echo "  --resume               从断点恢复"
    echo "  --resume-from STEP     从指定步骤恢复 (1.1-3.1)"
    echo "  --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 首次运行，使用火山API和多线程"
    echo "  $0 --use-volcano --volcano-api-key YOUR_KEY --max-workers 8"
    echo ""
    echo "  # 从断点恢复"
    echo "  $0 --resume"
    echo ""
    echo "  # 从特定步骤恢复"
    echo "  $0 --resume --resume-from 2.1"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --quality-threshold)
            QUALITY_THRESHOLD="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
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
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查Python环境
echo -e "${GREEN}检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python3${NC}"
    exit 1
fi

# 检查必要的Python包
echo -e "${GREEN}检查依赖包...${NC}"
python3 -c "import aiohttp" 2>/dev/null || {
    echo -e "${YELLOW}安装aiohttp...${NC}"
    pip install aiohttp
}

# 创建必要的目录
echo -e "${GREEN}创建目录结构...${NC}"
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p logs

# 构建Python命令
CMD="python3 run_semiconductor_qa_enhanced_full.py"
CMD="$CMD --input-dir $INPUT_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --config $CONFIG_FILE"
CMD="$CMD --model $MODEL"
CMD="$CMD --quality-threshold $QUALITY_THRESHOLD"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --max-workers $MAX_WORKERS"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"

if [ "$USE_VOLCANO" = true ]; then
    CMD="$CMD --use-volcano"
    if [ -n "$VOLCANO_API_KEY" ]; then
        CMD="$CMD --volcano-api-key $VOLCANO_API_KEY"
    else
        echo -e "${YELLOW}警告: 使用火山API但未提供API密钥${NC}"
    fi
fi

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
    if [ -n "$RESUME_FROM" ]; then
        CMD="$CMD --resume-from $RESUME_FROM"
    fi
fi

# 显示运行信息
echo -e "${GREEN}=== 半导体QA生成系统 - 增强版 ===${NC}"
echo -e "输入目录: $INPUT_DIR"
echo -e "输出目录: $OUTPUT_DIR"
echo -e "配置文件: $CONFIG_FILE"
echo -e "模型: $MODEL"
echo -e "质量阈值: $QUALITY_THRESHOLD"
echo -e "批处理大小: $BATCH_SIZE"
echo -e "最大线程数: $MAX_WORKERS"
echo -e "断点目录: $CHECKPOINT_DIR"
echo -e "使用火山API: $USE_VOLCANO"
echo -e "从断点恢复: $RESUME"
if [ -n "$RESUME_FROM" ]; then
    echo -e "恢复步骤: $RESUME_FROM"
fi
echo -e "${GREEN}=================================${NC}"
echo ""

# 运行主程序
LOG_FILE="logs/enhanced_run_$(date +%Y%m%d_%H%M%S).log"
echo -e "${GREEN}开始运行，日志保存至: $LOG_FILE${NC}"
echo -e "${YELLOW}执行命令: $CMD${NC}"
echo ""

# 执行并保存日志
$CMD 2>&1 | tee "$LOG_FILE"

# 检查退出状态
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== 流程成功完成 ===${NC}"
    echo -e "输出文件位于: $OUTPUT_DIR"
    echo -e "日志文件: $LOG_FILE"
    
    # 显示统计信息
    if [ -f "$OUTPUT_DIR/pipeline_stats.json" ]; then
        echo ""
        echo -e "${GREEN}统计信息:${NC}"
        python3 -c "
import json
with open('$OUTPUT_DIR/pipeline_stats.json', 'r') as f:
    stats = json.load(f)
    print(f\"  - 处理文本: {stats.get('stage1_text_processing', {}).get('total_processed', 0)}\")
    print(f\"  - 合格文本: {stats.get('stage1_text_processing', {}).get('qualified_texts', 0)}\")
    print(f\"  - 生成QA: {stats.get('stage2_qa_generation', {}).get('total_qa_generated', 0)}\")
    print(f\"  - 高质量QA: {stats.get('stage2_qa_generation', {}).get('high_quality_qa', 0)}\")
    print(f\"  - 最终数据集: {stats.get('stage3_enhancement', {}).get('final_dataset_size', 0)}\")
"
    fi
else
    echo ""
    echo -e "${RED}=== 流程执行失败 ===${NC}"
    echo -e "请查看日志文件: $LOG_FILE"
    echo ""
    echo -e "${YELLOW}提示: 您可以使用 --resume 参数从断点恢复${NC}"
    
    # 显示最新断点信息
    if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
        LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.pkl 2>/dev/null | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
            STEP_NUMBER=$(echo "$CHECKPOINT_NAME" | cut -d'_' -f1)
            echo -e "${YELLOW}最新断点: 步骤 $STEP_NUMBER${NC}"
            echo -e "${YELLOW}恢复命令: $0 --resume${NC}"
        fi
    fi
    
    exit 1
fi