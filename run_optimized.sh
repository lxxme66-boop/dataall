#!/bin/bash
# 半导体QA生成系统 - 优化版本启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 半导体QA生成系统 - 优化版本 ===${NC}"
echo ""

# 默认参数
INPUT_DIR="data/texts"
OUTPUT_DIR="data/output"
MODEL="vllm_http"
BATCH_SIZE=32
MAX_WORKERS=16
QUALITY_THRESHOLD=0.7
USE_VOLCANO=false
VOLCANO_API_KEY=""
NO_RESUME=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --volcano)
            USE_VOLCANO=true
            shift
            ;;
        --api-key)
            VOLCANO_API_KEY="$2"
            shift 2
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-resume)
            NO_RESUME=true
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --volcano         使用火山API (deepseek-r1)"
            echo "  --api-key KEY     火山API密钥"
            echo "  --input DIR       输入目录 (默认: data/texts)"
            echo "  --output DIR      输出目录 (默认: data/output)"
            echo "  --workers N       最大工作线程数 (默认: 16)"
            echo "  --batch N         批处理大小 (默认: 32)"
            echo "  --no-resume       禁用断点续跑"
            echo "  --help           显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建命令
CMD="python run_semiconductor_qa_optimized.py"
CMD="$CMD --input-dir $INPUT_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --max-workers $MAX_WORKERS"
CMD="$CMD --quality-threshold $QUALITY_THRESHOLD"

# 选择API模式
if [ "$USE_VOLCANO" = true ]; then
    echo -e "${YELLOW}使用火山API (deepseek-r1)${NC}"
    
    # 检查API密钥
    if [ -z "$VOLCANO_API_KEY" ]; then
        # 尝试从环境变量获取
        if [ -z "$VOLCANO_API_KEY" ]; then
            echo -e "${RED}错误: 未提供火山API密钥${NC}"
            echo "请使用 --api-key 参数或设置 VOLCANO_API_KEY 环境变量"
            exit 1
        fi
    fi
    
    CMD="$CMD --use-volcano --volcano-api-key $VOLCANO_API_KEY"
else
    echo -e "${YELLOW}使用vLLM HTTP模式${NC}"
    
    # 检查vLLM服务器是否运行
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${YELLOW}警告: vLLM服务器未运行${NC}"
        echo "请先运行: python start_vllm_server.py"
        read -p "是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    CMD="$CMD --config config_vllm_http.json"
fi

# 断点续跑选项
if [ "$NO_RESUME" = true ]; then
    CMD="$CMD --no-resume"
    echo -e "${YELLOW}禁用断点续跑 - 将重新开始${NC}"
else
    # 检查是否有进度文件
    if [ -f "$OUTPUT_DIR/.checkpoints/progress_checkpoint.json" ]; then
        echo -e "${GREEN}检测到进度文件，将从断点继续${NC}"
    else
        echo -e "${YELLOW}首次运行 - 将自动保存进度${NC}"
    fi
fi

# 显示配置
echo ""
echo "配置信息:"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  批处理大小: $BATCH_SIZE"
echo "  工作线程数: $MAX_WORKERS"
echo "  质量阈值: $QUALITY_THRESHOLD"
echo ""

# 显示将要执行的命令
echo -e "${GREEN}执行命令:${NC}"
echo "$CMD"
echo ""

# 确认执行
read -p "是否开始执行？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)

echo ""
echo -e "${GREEN}开始执行...${NC}"
echo "="*50

# 执行命令
$CMD

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 转换为时分秒
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "="*50
echo ""
echo -e "${GREEN}执行完成！${NC}"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"

# 运行验证脚本
if [ -f "test_flow_verification.py" ]; then
    echo ""
    echo -e "${YELLOW}运行流程验证...${NC}"
    python test_flow_verification.py --output-dir "$OUTPUT_DIR"
fi

echo ""
echo -e "${GREEN}输出文件位于: $OUTPUT_DIR${NC}"