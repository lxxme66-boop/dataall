import asyncio
from run_semiconductor_qa import run_complete_pipeline  # 确保主流程函数名正确

import asyncio
import os
import sys

# 确保能导入 run_complete_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_semiconductor_qa import run_complete_pipeline

async def test_preprocessing():
    """测试文本预处理流程"""
    # 模拟配置
    config = {
        "paths": {
            "text_dir": "data/texts",
            "output_dir": "data/qa_results"
        },
        "processing": {
            "batch_size": 2,
            "chunk_size": 2000,
            "chunk_overlap": 200
        },
        "api": {
            "use_vllm_http": True,
            "vllm_server_url": "http://localhost:8000/v1"
        },
        "models": {
            "qa_generator_model": {
                "name": "qwen-vllm",
                "temperature": 0.7,
                "max_tokens": 4096
            }
        }
    }
    
    # 运行完整流程
    results = await run_complete_pipeline(
        config,
        input_dir=config['paths']['text_dir'],
        output_dir=config['paths']['output_dir'],
        model_name="qwen-vllm"
    )
    
    print(f"处理结果: {len(results)} 个文件")
    
    # 显示部分结果
    if results:
        for i, result in enumerate(results[:2]):
            print(f"\n文件 {i+1}: {result['source_file']}")
            print(f"内容摘要: {result['content'][:100]}...")

# 运行测试
if __name__ == "__main__":
    asyncio.run(test_preprocessing())