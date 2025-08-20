#!/usr/bin/env python3
"""
测试DeepSeek API连接的简单脚本
"""

import asyncio
import os
import sys
from deepseek_api_client import DeepSeekAPIClient

async def test_api():
    """测试API连接"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("错误: 请设置DEEPSEEK_API_KEY环境变量")
        print("export DEEPSEEK_API_KEY='your_api_key'")
        return False
    
    print(f"使用API密钥: {api_key[:10]}...")
    
    try:
        async with DeepSeekAPIClient(api_key=api_key) as client:
            print("测试简单生成...")
            response = await client.generate(
                prompt="什么是半导体？请用一句话回答。",
                temperature=0.5,
                max_tokens=100
            )
            
            print(f"API响应: {response}")
            
            print("\n测试批量生成...")
            prompts = [
                "什么是晶体管？",
                "什么是集成电路？",
                "什么是摩尔定律？"
            ]
            
            responses = await client.batch_generate(
                prompts=prompts,
                temperature=0.5,
                max_tokens=100
            )
            
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                print(f"\n问题{i+1}: {prompt}")
                print(f"答案{i+1}: {response[:100]}...")
            
            print("\n✅ API测试成功！")
            return True
            
    except Exception as e:
        print(f"\n❌ API测试失败: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_api())
    sys.exit(0 if success else 1)