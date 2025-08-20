#!/bin/bash

# 火山引擎 DeepSeek-R1 API 设置脚本
# 用于配置环境变量和准备运行环境

echo "======================================"
echo "火山引擎 DeepSeek-R1 QA生成系统设置"
echo "======================================"

# 检查是否已设置必要的环境变量
check_env_vars() {
    local missing_vars=()
    
    if [ -z "$VOLCANO_API_KEY" ]; then
        missing_vars+=("VOLCANO_API_KEY")
    fi
    
    if [ -z "$VOLCANO_ENDPOINT_ID" ]; then
        missing_vars+=("VOLCANO_ENDPOINT_ID")
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "❌ 缺少必要的环境变量："
        for var in "${missing_vars[@]}"; do
            echo "   - $var"
        done
        echo ""
        echo "请设置以下环境变量："
        echo "  export VOLCANO_API_KEY='your-api-key'"
        echo "  export VOLCANO_ENDPOINT_ID='your-endpoint-id'"
        return 1
    fi
    
    echo "✅ 环境变量检查通过"
    return 0
}

# 创建必要的目录结构
create_directories() {
    echo "创建目录结构..."
    
    # 数据目录
    mkdir -p data/texts
    mkdir -p data/volcano_output
    
    # 缓存目录
    mkdir -p cache/volcano_deepseek
    
    # 日志目录
    mkdir -p logs
    
    # 临时目录
    mkdir -p temp/volcano
    
    # 检查点目录
    mkdir -p checkpoints/volcano
    
    # 指标目录
    mkdir -p metrics
    
    echo "✅ 目录结构创建完成"
}

# 安装Python依赖
install_dependencies() {
    echo "检查Python依赖..."
    
    # 检查是否有requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "安装依赖包..."
        pip install -r requirements.txt
    else
        echo "⚠️  未找到requirements.txt，跳过依赖安装"
    fi
    
    # 安装火山引擎SDK（如果需要）
    echo "检查火山引擎SDK..."
    python -c "import volcenginesdkarkruntime" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "安装火山引擎SDK..."
        pip install volcenginesdkarkruntime
    else
        echo "✅ 火山引擎SDK已安装"
    fi
}

# 验证配置文件
validate_config() {
    echo "验证配置文件..."
    
    if [ -f "config_volcano_deepseek.json" ]; then
        python -c "import json; json.load(open('config_volcano_deepseek.json'))" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✅ 配置文件验证通过"
        else
            echo "❌ 配置文件格式错误"
            return 1
        fi
    else
        echo "❌ 配置文件不存在: config_volcano_deepseek.json"
        return 1
    fi
    
    return 0
}

# 创建示例数据
create_sample_data() {
    echo "创建示例数据..."
    
    # 创建示例文本文件
    cat > data/texts/sample_semiconductor.txt << 'EOF'
半导体显示技术是现代电子设备的核心技术之一。OLED（有机发光二极管）技术因其自发光特性，
具有高对比度、快速响应时间和宽视角等优势。与传统的LCD技术相比，OLED不需要背光源，
可以实现更薄的显示面板设计。

TFT（薄膜晶体管）技术是驱动现代显示器的关键技术。TFT-LCD通过控制液晶分子的排列来调节光的透过率，
而AMOLED则使用TFT来控制每个OLED像素的电流。低温多晶硅（LTPS）TFT技术因其高电子迁移率，
被广泛应用于高分辨率显示器中。

量子点显示技术（QLED）是另一种新兴的显示技术。量子点是纳米级的半导体晶体，
可以通过调节其尺寸来精确控制发光颜色。QLED技术结合了LCD的成熟制造工艺和量子点的优异光学特性，
能够实现更广的色域覆盖和更高的能效。

柔性显示技术是未来显示技术的重要发展方向。通过使用柔性基板材料如聚酰亚胺（PI），
可以制造可弯曲、可折叠的显示器。这需要解决多个技术挑战，包括柔性封装、应力管理和可靠性保证。

Micro-LED技术被认为是下一代显示技术的有力竞争者。与OLED相比，Micro-LED具有更高的亮度、
更长的寿命和更好的稳定性。然而，巨量转移技术仍然是Micro-LED商业化的主要瓶颈。
EOF
    
    echo "✅ 示例数据创建完成"
}

# 显示使用说明
show_usage() {
    echo ""
    echo "======================================"
    echo "使用说明"
    echo "======================================"
    echo ""
    echo "1. 基础运行（处理示例数据）:"
    echo "   python run_volcano_qa.py --input-dir data/texts --output-dir data/volcano_output"
    echo ""
    echo "2. 大规模处理（4000文档，生成20000 QA）:"
    echo "   python run_volcano_qa.py \\"
    echo "     --input-dir data/texts \\"
    echo "     --output-dir data/volcano_output \\"
    echo "     --max-files 4000 \\"
    echo "     --target-qa 20000 \\"
    echo "     --workers 32"
    echo ""
    echo "3. 启用数据增强:"
    echo "   python run_volcano_qa.py \\"
    echo "     --input-dir data/texts \\"
    echo "     --output-dir data/volcano_output \\"
    echo "     --enable-enhancement"
    echo ""
    echo "4. 使用自定义配置:"
    echo "   python run_volcano_qa.py \\"
    echo "     --config config_volcano_deepseek.json \\"
    echo "     --input-dir data/texts \\"
    echo "     --output-dir data/volcano_output"
    echo ""
    echo "5. 监控进度:"
    echo "   tail -f volcano_qa_generation.log"
    echo ""
    echo "======================================"
}

# 主函数
main() {
    echo "开始设置..."
    echo ""
    
    # 检查环境变量
    if ! check_env_vars; then
        exit 1
    fi
    
    # 创建目录
    create_directories
    
    # 安装依赖
    install_dependencies
    
    # 验证配置
    if ! validate_config; then
        exit 1
    fi
    
    # 创建示例数据
    create_sample_data
    
    # 显示使用说明
    show_usage
    
    echo ""
    echo "✅ 设置完成！系统已准备就绪。"
    echo ""
}

# 运行主函数
main