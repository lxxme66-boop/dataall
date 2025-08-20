#!/bin/bash
# 安装火山引擎SDK和相关依赖

echo "==================================="
echo "火山引擎API环境配置脚本"
echo "==================================="

# 1. 安装火山引擎SDK
echo "步骤1: 安装火山引擎SDK..."
pip install volcengine-python-sdk[ark] -q

# 2. 更新其他依赖
echo "步骤2: 更新并行处理相关依赖..."
pip install aiohttp asyncio-throttle tqdm -q

# 3. 检查安装
echo "步骤3: 验证安装..."
python -c "
try:
    from volcenginesdkarkruntime import Ark, AsyncArk
    print('✓ 火山引擎SDK安装成功')
except ImportError:
    print('✗ 火山引擎SDK安装失败')
"

# 4. 设置环境变量提示
echo ""
echo "==================================="
echo "环境变量配置"
echo "==================================="
echo "请设置以下环境变量以使用火山API："
echo ""
echo "export VOLCANO_API_KEY='your_api_key_here'"
echo "export VOLCANO_ENDPOINT_ID='your_endpoint_id_here'"
echo ""
echo "或者在运行时指定："
echo "VOLCANO_API_KEY='xxx' python run_large_scale_qa.py --use-volcano"
echo ""
echo "==================================="
echo "配置完成！"
echo "===================================" 