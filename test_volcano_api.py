#!/usr/bin/env python3
"""
火山引擎API测试脚本
用于验证API连接和基本功能
"""

import asyncio
import json
import logging
from volcenginesdkarkruntime import Ark, AsyncArk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sync_api(api_key: str, endpoint_id: str):
    """测试同步API"""
    logger.info("测试同步API...")
    
    try:
        client = Ark(api_key=api_key)
        
        response = client.chat.completions.create(
            model=endpoint_id,
            messages=[
                {"role": "system", "content": "你是一个半导体技术专家"},
                {"role": "user", "content": "请简单介绍一下OLED显示技术"}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        logger.info("同步API测试成功！")
        logger.info(f"响应: {response.choices[0].message.content[:100]}...")
        return True
        
    except Exception as e:
        logger.error(f"同步API测试失败: {e}")
        return False

async def test_async_api(api_key: str, endpoint_id: str):
    """测试异步API"""
    logger.info("测试异步API...")
    
    try:
        client = AsyncArk(api_key=api_key)
        
        response = await client.chat.completions.create(
            model=endpoint_id,
            messages=[
                {"role": "system", "content": "你是一个半导体技术专家"},
                {"role": "user", "content": "TFT和OLED的主要区别是什么？"}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        logger.info("异步API测试成功！")
        logger.info(f"响应: {response.choices[0].message.content[:100]}...")
        return True
        
    except Exception as e:
        logger.error(f"异步API测试失败: {e}")
        return False

async def test_batch_processing(api_key: str, endpoint_id: str):
    """测试批处理"""
    logger.info("测试批处理...")
    
    client = AsyncArk(api_key=api_key)
    
    questions = [
        "什么是IGZO？",
        "AMOLED的工作原理是什么？",
        "柔性显示技术的挑战有哪些？"
    ]
    
    tasks = []
    for q in questions:
        task = client.chat.completions.create(
            model=endpoint_id,
            messages=[{"role": "user", "content": q}],
            max_tokens=150
        )
        tasks.append(task)
    
    try:
        responses = await asyncio.gather(*tasks)
        logger.info(f"批处理测试成功！处理了 {len(responses)} 个请求")
        for i, resp in enumerate(responses):
            logger.info(f"问题 {i+1}: {questions[i][:20]}... -> 回答长度: {len(resp.choices[0].message.content)}")
        return True
        
    except Exception as e:
        logger.error(f"批处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试火山引擎API")
    parser.add_argument("--api-key", required=True, help="火山引擎API密钥")
    parser.add_argument("--endpoint-id", required=True, help="火山引擎端点ID")
    parser.add_argument("--test-type", choices=['sync', 'async', 'batch', 'all'], 
                       default='all', help="测试类型")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("火山引擎API测试")
    logger.info("="*60)
    
    results = {}
    
    if args.test_type in ['sync', 'all']:
        results['sync'] = test_sync_api(args.api_key, args.endpoint_id)
    
    if args.test_type in ['async', 'all']:
        results['async'] = asyncio.run(test_async_api(args.api_key, args.endpoint_id))
    
    if args.test_type in ['batch', 'all']:
        results['batch'] = asyncio.run(test_batch_processing(args.api_key, args.endpoint_id))
    
    # 打印测试结果
    logger.info("="*60)
    logger.info("测试结果汇总:")
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
    logger.info("="*60)
    
    # 如果所有测试都通过
    if all(results.values()):
        logger.info("🎉 所有测试通过！API配置正确。")
    else:
        logger.error("❌ 部分测试失败，请检查API配置。")

if __name__ == "__main__":
    main()