#!/usr/bin/env python3
"""
测试火山引擎集成
验证DeepSeek-R1模型调用和多线程处理
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from LocalModels.volcano_client import VolcanoAPIClient, VolcanoModelManager


def test_basic_connection():
    """测试基础连接"""
    print("=" * 50)
    print("测试1: 基础连接测试")
    print("=" * 50)
    
    # 检查环境变量
    api_key = os.getenv('VOLCANO_API_KEY')
    endpoint_id = os.getenv('VOLCANO_ENDPOINT_ID')
    
    if not api_key:
        print("❌ 未设置 VOLCANO_API_KEY 环境变量")
        print("   请运行: export VOLCANO_API_KEY='your-api-key'")
        return False
    
    if not endpoint_id:
        print("⚠️  未设置 VOLCANO_ENDPOINT_ID 环境变量")
        print("   将使用默认端点")
    
    print("✅ 环境变量已设置")
    return True


def test_single_generation():
    """测试单个文本生成"""
    print("\n" + "=" * 50)
    print("测试2: 单个文本生成")
    print("=" * 50)
    
    try:
        # 加载配置
        config_path = "config_volcano_deepseek.json"
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 创建客户端
        client = VolcanoAPIClient(config)
        
        # 测试生成
        prompt = "什么是OLED显示技术？请用100字简要说明。"
        print(f"提示: {prompt}")
        print("生成中...")
        
        start_time = time.time()
        result = client.generate(prompt, temperature=0.7, max_tokens=200)
        elapsed = time.time() - start_time
        
        if result:
            print(f"✅ 生成成功 (耗时: {elapsed:.2f}秒)")
            print(f"回答: {result[:200]}...")
            return True
        else:
            print("❌ 生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()


def test_batch_generation():
    """测试批量生成"""
    print("\n" + "=" * 50)
    print("测试3: 批量生成（多线程）")
    print("=" * 50)
    
    try:
        # 加载配置
        with open("config_volcano_deepseek.json", 'r') as f:
            config = json.load(f)
        
        # 创建客户端
        client = VolcanoAPIClient(config)
        
        # 准备测试提示
        prompts = [
            "什么是LCD技术？",
            "OLED和LCD的主要区别是什么？",
            "什么是量子点显示技术？",
            "TFT的工作原理是什么？",
            "Micro-LED技术的优势有哪些？"
        ]
        
        print(f"批量生成 {len(prompts)} 个回答...")
        
        start_time = time.time()
        results = client.batch_generate(prompts, temperature=0.7, max_tokens=150)
        elapsed = time.time() - start_time
        
        # 检查结果
        success_count = sum(1 for r in results if r)
        print(f"✅ 完成: {success_count}/{len(prompts)} 成功")
        print(f"耗时: {elapsed:.2f}秒")
        print(f"平均速度: {len(prompts)/elapsed:.2f} 个/秒")
        
        # 显示部分结果
        for i, (prompt, result) in enumerate(zip(prompts[:2], results[:2])):
            if result:
                print(f"\n问题{i+1}: {prompt}")
                print(f"回答: {result[:100]}...")
        
        return success_count == len(prompts)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()


async def test_async_generation():
    """测试异步生成"""
    print("\n" + "=" * 50)
    print("测试4: 异步生成")
    print("=" * 50)
    
    try:
        # 加载配置
        with open("config_volcano_deepseek.json", 'r') as f:
            config = json.load(f)
        
        # 创建客户端
        client = VolcanoAPIClient(config)
        
        # 准备测试提示
        prompts = [
            "解释半导体的导电原理",
            "什么是PN结？",
            "晶体管的工作原理"
        ]
        
        print(f"异步生成 {len(prompts)} 个回答...")
        
        start_time = time.time()
        results = await client.abatch_generate(prompts, temperature=0.5, max_tokens=200)
        elapsed = time.time() - start_time
        
        success_count = sum(1 for r in results if r)
        print(f"✅ 完成: {success_count}/{len(prompts)} 成功")
        print(f"耗时: {elapsed:.2f}秒")
        
        return success_count == len(prompts)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()


def test_stream_generation():
    """测试流式生成"""
    print("\n" + "=" * 50)
    print("测试5: 流式生成（带进度）")
    print("=" * 50)
    
    try:
        # 加载配置
        with open("config_volcano_deepseek.json", 'r') as f:
            config = json.load(f)
        
        # 创建客户端
        client = VolcanoAPIClient(config)
        
        # 准备测试提示
        prompts = [
            f"问题{i}: 描述半导体技术的应用场景{i}"
            for i in range(1, 6)
        ]
        
        print(f"流式生成 {len(prompts)} 个回答...")
        
        # 进度回调
        def progress_callback(completed, total):
            percent = (completed / total) * 100
            print(f"进度: {completed}/{total} ({percent:.1f}%)")
        
        results = {}
        start_time = time.time()
        
        for index, result in client.stream_generate(prompts, callback=progress_callback, max_tokens=100):
            results[index] = result
        
        elapsed = time.time() - start_time
        
        success_count = sum(1 for r in results.values() if r)
        print(f"\n✅ 完成: {success_count}/{len(prompts)} 成功")
        print(f"总耗时: {elapsed:.2f}秒")
        
        return success_count == len(prompts)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()


def test_performance():
    """性能测试"""
    print("\n" + "=" * 50)
    print("测试6: 性能测试")
    print("=" * 50)
    
    try:
        # 加载配置
        with open("config_volcano_deepseek.json", 'r') as f:
            config = json.load(f)
        
        # 调整配置以进行性能测试
        config['parallel_processing']['max_workers'] = 16
        
        # 创建客户端
        client = VolcanoAPIClient(config)
        
        # 准备大批量提示
        num_prompts = 20
        prompts = [
            f"简要说明半导体技术要点{i}"
            for i in range(num_prompts)
        ]
        
        print(f"性能测试: 生成 {num_prompts} 个回答")
        print(f"并行线程数: {config['parallel_processing']['max_workers']}")
        
        start_time = time.time()
        results = client.batch_generate(prompts, temperature=0.5, max_tokens=50)
        elapsed = time.time() - start_time
        
        # 统计结果
        success_count = sum(1 for r in results if r)
        stats = client.get_stats()
        
        print(f"\n性能统计:")
        print(f"  成功率: {success_count}/{num_prompts} ({success_count/num_prompts*100:.1f}%)")
        print(f"  总耗时: {elapsed:.2f}秒")
        print(f"  平均速度: {num_prompts/elapsed:.2f} 个/秒")
        print(f"  API调用次数: {stats['total_requests']}")
        print(f"  API成功率: {stats['success_rate']*100:.1f}%")
        print(f"  总Token数: {stats['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()


def main():
    """运行所有测试"""
    print("=" * 50)
    print("火山引擎 DeepSeek-R1 集成测试")
    print("=" * 50)
    
    # 测试结果
    results = {}
    
    # 运行测试
    results['connection'] = test_basic_connection()
    
    if results['connection']:
        results['single'] = test_single_generation()
        results['batch'] = test_batch_generation()
        
        # 运行异步测试
        results['async'] = asyncio.run(test_async_generation())
        
        results['stream'] = test_stream_generation()
        results['performance'] = test_performance()
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    test_names = {
        'connection': '基础连接',
        'single': '单个生成',
        'batch': '批量生成',
        'async': '异步生成',
        'stream': '流式生成',
        'performance': '性能测试'
    }
    
    for key, name in test_names.items():
        if key in results:
            status = "✅ 通过" if results[key] else "❌ 失败"
            print(f"{name}: {status}")
    
    # 总体结果
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print("\n" + "=" * 50)
    if passed_tests == total_tests:
        print(f"✅ 所有测试通过 ({passed_tests}/{total_tests})")
    else:
        print(f"⚠️  部分测试失败 ({passed_tests}/{total_tests})")
    print("=" * 50)


if __name__ == "__main__":
    main()