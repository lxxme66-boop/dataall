#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用vLLM HTTP服务运行文本处理
简化版本，专门用于分离式架构
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_vllm_server(base_url="http://localhost:8000/v1"):
    """检查vLLM服务器是否运行"""
    try:
        import requests
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            logger.info(f"vLLM服务器运行正常，可用模型: {models}")
            return True
    except Exception as e:
        logger.error(f"无法连接到vLLM服务器: {e}")
    return False

def main():
    parser = argparse.ArgumentParser(description="使用vLLM HTTP服务处理文本")
    parser.add_argument("--input-dir", type=str, required=True, help="输入文本目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1", help="vLLM服务器地址")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    parser.add_argument("--max-workers", type=int, default=1, help="并发工作进程数")
    
    args = parser.parse_args()
    
    # 检查vLLM服务器
    if not check_vllm_server(args.vllm_url):
        logger.error("请先启动vLLM服务器：python start_vllm_server.py")
        sys.exit(1)
    
    # 设置环境变量
    os.environ['VLLM_SERVER_URL'] = args.vllm_url
    os.environ['USE_LOCAL_MODELS'] = 'true'
    
    # 创建vLLM HTTP配置
    config = {
        "api": {
            "use_local_models": True,
            "use_vllm_http": True,
            "vllm_server_url": args.vllm_url,
            "default_backend": "vllm_http"
        },
        "models": {
            "local_models": {
                "default_backend": "vllm_http",
                "vllm_http": {
                    "base_url": args.vllm_url,
                    "api_key": "dummy-key",
                    "model_name": "qwen-vllm",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "timeout": 300
                }
            }
        },
        "processing": {
            "batch_size": args.batch_size,
            "pool_size": args.max_workers
        }
    }
    
    # 保存临时配置文件
    config_path = "temp_vllm_http_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    try:
        # 运行主程序
        from run_semiconductor_qa import main as run_qa
        
        # 构建参数
        qa_args = [
            "--input-dir", args.input_dir,
            "--output-dir", args.output_dir,
            "--config", config_path,
            "--mode", "simple"  # 使用简化模式
        ]
        
        # 修改sys.argv
        original_argv = sys.argv
        sys.argv = ["run_semiconductor_qa.py"] + qa_args
        
        # 运行
        run_qa()
        
    finally:
        # 清理临时配置文件
        if os.path.exists(config_path):
            os.remove(config_path)
        
        # 恢复原始argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()