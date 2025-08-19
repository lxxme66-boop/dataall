# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# vLLM服务器启动脚本
# 用于独立启动vLLM模型服务，供其他进程通过HTTP API调用
# """

# import os
# import sys
# import argparse
# import logging
# from typing import Optional

# # 设置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def start_vllm_server(
#     model_path: str = "/mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain",
#     host: str = "0.0.0.0",
#     port: int = 8000,
#     gpu_memory_utilization: float = 0.95,
#     tensor_parallel_size: int = 2,
#     max_model_len: int = 32768,
#     api_key: Optional[str] = None
# ):
#     """
#     启动vLLM服务器
    
#     Args:
#         model_path: 模型路径
#         host: 服务器地址
#         port: 服务器端口
#         gpu_memory_utilization: GPU内存利用率
#         tensor_parallel_size: 张量并行大小
#         max_model_len: 最大模型长度
#         api_key: API密钥（可选）
#     """
#     try:
#         # 构建vLLM服务器命令
#         cmd = [
#             "python", "-m", "vllm.entrypoints.openai.api_server",
#             "--model", model_path,
#             "--host", host,
#             "--port", str(port),
#             "--gpu-memory-utilization", str(gpu_memory_utilization),
#             "--tensor-parallel-size", str(tensor_parallel_size),
#             "--max-model-len", str(max_model_len),
#             "--trust-remote-code",
#             "--served-model-name", "qwen-vllm"
#         ]
        
#         # 如果提供了API密钥，添加到命令中
#         if api_key:
#             cmd.extend(["--api-key", api_key])
        
#         # 打印启动信息
#         logger.info("="*60)
#         logger.info("启动vLLM服务器")
#         logger.info(f"模型路径: {model_path}")
#         logger.info(f"服务地址: http://{host}:{port}")
#         logger.info(f"GPU内存利用率: {gpu_memory_utilization}")
#         logger.info(f"张量并行大小: {tensor_parallel_size}")
#         logger.info(f"最大模型长度: {max_model_len}")
#         logger.info("="*60)
        
#         # 执行命令
#         import subprocess
#         subprocess.run(cmd)
        
#     except KeyboardInterrupt:
#         logger.info("\n服务器已停止")
#     except Exception as e:
#         logger.error(f"启动vLLM服务器失败: {e}")
#         sys.exit(1)

# def main():
#     """主函数"""
#     parser = argparse.ArgumentParser(description="启动vLLM模型服务器")
#     parser.add_argument(
#         "--model-path",
#         type=str,
#         default="/mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain",
#         help="模型路径"
#     )
#     parser.add_argument(
#         "--host",
#         type=str,
#         default="0.0.0.0",
#         help="服务器地址"
#     )
#     parser.add_argument(
#         "--port",
#         type=int,
#         default=8000,
#         help="服务器端口"
#     )
#     parser.add_argument(
#         "--gpu-memory-utilization",
#         type=float,
#         default=0.95,
#         help="GPU内存利用率"
#     )
#     parser.add_argument(
#         "--tensor-parallel-size",
#         type=int,
#         default=2,
#         help="张量并行大小"
#     )
#     parser.add_argument(
#         "--max-model-len",
#         type=int,
#         default=32768,
#         help="最大模型长度"
#     )
#     parser.add_argument(
#         "--api-key",
#         type=str,
#         default=None,
#         help="API密钥（可选）"
#     )
    
#     args = parser.parse_args()
    
#     # 检查vLLM是否安装
#     try:
#         import vllm
#         logger.info(f"vLLM版本: {vllm.__version__}")
#     except ImportError:
#         logger.error("vLLM未安装，请运行: pip install vllm")
#         sys.exit(1)
    
#     # 启动服务器
#     start_vllm_server(
#         model_path=args.model_path,
#         host=args.host,
#         port=args.port,
#         gpu_memory_utilization=args.gpu_memory_utilization,
#         tensor_parallel_size=args.tensor_parallel_size,
#         max_model_len=args.max_model_len,
#         api_key=args.api_key
#     )

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM服务器启动脚本
用于独立启动vLLM模型服务，供其他进程通过HTTP API调用
"""

import os
import sys
import argparse
import logging
from typing import Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_vllm_server(
    model_path: str = "/mnt/data/LLM/lhy/models/Qwen3/Qwen3-235B-A22B",
    host: str = "0.0.0.0",
    port: int = 8000,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 8,
    max_model_len: int = 40960,  #32768
    api_key: Optional[str] = None
):
    """
    启动vLLM服务器
    
    Args:
        model_path: 模型路径
        host: 服务器地址
        port: 服务器端口
        gpu_memory_utilization: GPU内存利用率
        tensor_parallel_size: 张量并行大小
        max_model_len: 最大模型长度
        api_key: API密钥（可选）
    """
    try:
        # 构建vLLM服务器命令
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", host,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-model-len", str(max_model_len),
            "--trust-remote-code",
            "--served-model-name", "qwen-vllm"
        ]

        # 如果提供了API密钥，添加到命令中
        if api_key:
            cmd.extend(["--api-key", api_key])
        
        # 打印启动信息
        logger.info("="*60)
        logger.info("启动vLLM服务器")
        logger.info(f"模型路径: {model_path}")
        logger.info(f"服务地址: http://{host}:{port}")
        logger.info(f"GPU内存利用率: {gpu_memory_utilization}")
        logger.info(f"张量并行大小: {tensor_parallel_size}")
        logger.info(f"最大模型长度: {max_model_len}")
        logger.info("="*60)
        
        # 执行命令
        import subprocess
        # 使用Popen保持进程运行
        process = subprocess.Popen(cmd)
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("\n服务器已停止")
    except Exception as e:
        logger.error(f"启动vLLM服务器失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动vLLM模型服务器")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/data/LLM/lhy/models/Qwen3/Qwen3-235B-A22B",
        help="模型路径"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU内存利用率"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="张量并行大小"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
        help="最大模型长度"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API密钥（可选）"
    )
    
    args = parser.parse_args()
    
    # 检查vLLM是否安装
    try:
        import vllm
        logger.info(f"vLLM版本: {vllm.__version__}")
    except ImportError:
        logger.error("vLLM未安装，请运行: pip install vllm")
        sys.exit(1)
    
    # 启动服务器
    start_vllm_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        api_key=args.api_key
    )

if __name__ == "__main__":
    main()


# python -m vllm.entrypoints.openai.api_server \
#   --model /mnt/data/LLM/lhy/models/Qwen3/Qwen3-235B-A22B \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --gpu-memory-utilization 0.95 \
#   --tensor-parallel-size 2 \
#   --max-model-len 32768 \
#   --trust-remote-code \
#   --served-model-name qwen-vllm