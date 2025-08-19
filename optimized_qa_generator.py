#!/usr/bin/env python3
"""
优化版半导体QA生成系统
- 支持多线程/多进程并发处理
- 集成火山引擎API
- 优化大数据集处理性能
"""

import os
import sys
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
from functools import partial
from dataclasses import dataclass
import queue
import threading

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 火山引擎SDK
try:
    from volcenginesdkarkruntime import Ark, AsyncArk
    HAS_VOLC_SDK = True
except ImportError:
    HAS_VOLC_SDK = False
    logger.warning("火山引擎SDK未安装，请运行: pip install volcengine-python-sdk[ark]")

@dataclass
class VolcanoConfig:
    """火山引擎API配置"""
    api_key: str
    endpoint_id: str
    region: str = "cn-beijing"
    model: str = "doubao-pro-32k"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    timeout: int = 60

@dataclass
class OptimizationConfig:
    """性能优化配置"""
    # 并发配置
    max_workers: int = None  # None表示自动设置为CPU核心数
    use_multiprocess: bool = False  # 是否使用多进程（适合CPU密集型）
    use_async: bool = True  # 是否使用异步（适合IO密集型）
    
    # 批处理配置
    batch_size: int = 32
    queue_size: int = 1000
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = "cache"
    
    # 速率限制
    rate_limit: int = 100  # 每秒最大请求数
    rate_limit_window: float = 1.0  # 速率限制窗口（秒）

class VolcanoAPIClient:
    """火山引擎API客户端"""
    
    def __init__(self, config: VolcanoConfig):
        self.config = config
        if HAS_VOLC_SDK:
            self.client = Ark(api_key=config.api_key)
            self.async_client = AsyncArk(api_key=config.api_key)
        else:
            self.client = None
            self.async_client = None
        
        # 速率限制
        self.request_times = []
        self.rate_limit_lock = threading.Lock()
    
    def _check_rate_limit(self):
        """检查速率限制"""
        with self.rate_limit_lock:
            current_time = time.time()
            # 清理过期的请求时间
            self.request_times = [t for t in self.request_times 
                                if current_time - t < self.config.timeout]
            
            # 检查是否超过速率限制
            if len(self.request_times) >= 100:  # 假设限制为100请求/秒
                sleep_time = 1.0 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.request_times.append(current_time)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """同步生成文本"""
        if not self.client:
            raise RuntimeError("火山引擎SDK未安装")
        
        self._check_rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.endpoint_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"火山引擎API调用失败: {e}")
            raise
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """异步生成文本"""
        if not self.async_client:
            raise RuntimeError("火山引擎SDK未安装")
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.endpoint_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"火山引擎API异步调用失败: {e}")
            raise

class OptimizedQAGenerator:
    """优化的QA生成器"""
    
    def __init__(
        self,
        volcano_config: VolcanoConfig,
        optimization_config: OptimizationConfig = None
    ):
        self.volcano_config = volcano_config
        self.opt_config = optimization_config or OptimizationConfig()
        
        # 初始化API客户端
        self.api_client = VolcanoAPIClient(volcano_config)
        
        # 设置并发执行器
        max_workers = self.opt_config.max_workers or cpu_count()
        if self.opt_config.use_multiprocess:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 缓存
        self.cache = {} if self.opt_config.enable_cache else None
        if self.opt_config.enable_cache:
            os.makedirs(self.opt_config.cache_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0,
            'start_time': time.time()
        }
    
    def process_text_chunk(self, text: str, chunk_id: int) -> Dict[str, Any]:
        """处理单个文本块（同步）"""
        try:
            # 检查缓存
            if self.cache is not None:
                cache_key = f"{hash(text)}_{chunk_id}"
                if cache_key in self.cache:
                    self.stats['cached'] += 1
                    return self.cache[cache_key]
            
            # 生成问题
            question_prompt = f"""请为以下半导体显示技术文本生成高质量的问题：

文本内容：
{text}

要求：
1. 生成3-5个专业问题
2. 问题类型要多样化（事实型、比较型、推理型、开放型）
3. 问题要有深度和专业性
4. 输出JSON格式

输出格式：
{{
    "questions": [
        {{"type": "类型", "question": "问题内容", "difficulty": "难度"}}
    ]
}}"""
            
            questions_response = self.api_client.generate(question_prompt)
            
            # 解析问题
            try:
                questions_data = json.loads(questions_response)
            except:
                # 如果JSON解析失败，尝试提取
                questions_data = {"questions": []}
            
            # 为每个问题生成答案
            qa_pairs = []
            for q in questions_data.get('questions', []):
                answer_prompt = f"""请回答以下关于半导体显示技术的问题：

上下文：
{text}

问题：{q.get('question', '')}

要求：
1. 答案要准确、专业、详细
2. 使用Chain of Thought方式组织答案
3. 包含推理过程和结论

输出格式：
{{
    "reasoning": "推理过程",
    "answer": "最终答案",
    "confidence": "置信度(0-1)"
}}"""
                
                answer_response = self.api_client.generate(answer_prompt)
                
                try:
                    answer_data = json.loads(answer_response)
                except:
                    answer_data = {"answer": answer_response, "reasoning": "", "confidence": 0.8}
                
                qa_pairs.append({
                    "chunk_id": chunk_id,
                    "question": q.get('question', ''),
                    "type": q.get('type', 'unknown'),
                    "difficulty": q.get('difficulty', 'medium'),
                    "answer": answer_data.get('answer', ''),
                    "reasoning": answer_data.get('reasoning', ''),
                    "confidence": answer_data.get('confidence', 0.8),
                    "context": text[:500]  # 保存部分上下文
                })
            
            result = {
                "chunk_id": chunk_id,
                "qa_pairs": qa_pairs,
                "processed_at": time.time()
            }
            
            # 更新缓存
            if self.cache is not None:
                self.cache[cache_key] = result
            
            self.stats['successful'] += 1
            return result
            
        except Exception as e:
            logger.error(f"处理文本块 {chunk_id} 失败: {e}")
            self.stats['failed'] += 1
            return {"chunk_id": chunk_id, "error": str(e), "qa_pairs": []}
    
    async def aprocess_text_chunk(self, text: str, chunk_id: int) -> Dict[str, Any]:
        """处理单个文本块（异步）"""
        try:
            # 检查缓存
            if self.cache is not None:
                cache_key = f"{hash(text)}_{chunk_id}"
                if cache_key in self.cache:
                    self.stats['cached'] += 1
                    return self.cache[cache_key]
            
            # 生成问题
            question_prompt = f"""请为以下半导体显示技术文本生成高质量的问题：

文本内容：
{text}

要求：
1. 生成3-5个专业问题
2. 问题类型要多样化
3. 输出JSON格式"""
            
            questions_response = await self.api_client.agenerate(question_prompt)
            
            # 解析并生成答案（并发处理）
            try:
                questions_data = json.loads(questions_response)
            except:
                questions_data = {"questions": []}
            
            # 并发生成所有答案
            answer_tasks = []
            for q in questions_data.get('questions', []):
                answer_prompt = f"""回答问题：
上下文：{text[:1000]}
问题：{q.get('question', '')}"""
                answer_tasks.append(self.api_client.agenerate(answer_prompt))
            
            if answer_tasks:
                answers = await asyncio.gather(*answer_tasks, return_exceptions=True)
            else:
                answers = []
            
            # 组装QA对
            qa_pairs = []
            for i, q in enumerate(questions_data.get('questions', [])):
                answer = answers[i] if i < len(answers) else ""
                if isinstance(answer, Exception):
                    answer = f"生成失败: {answer}"
                
                qa_pairs.append({
                    "chunk_id": chunk_id,
                    "question": q.get('question', ''),
                    "type": q.get('type', 'unknown'),
                    "answer": answer,
                    "context": text[:500]
                })
            
            result = {
                "chunk_id": chunk_id,
                "qa_pairs": qa_pairs,
                "processed_at": time.time()
            }
            
            # 更新缓存
            if self.cache is not None:
                self.cache[cache_key] = result
            
            self.stats['successful'] += 1
            return result
            
        except Exception as e:
            logger.error(f"异步处理文本块 {chunk_id} 失败: {e}")
            self.stats['failed'] += 1
            return {"chunk_id": chunk_id, "error": str(e), "qa_pairs": []}
    
    def process_batch_sync(self, text_chunks: List[str]) -> List[Dict[str, Any]]:
        """同步批处理（使用线程池/进程池）"""
        logger.info(f"开始批处理 {len(text_chunks)} 个文本块（同步模式）")
        results = []
        
        # 提交所有任务
        futures = []
        for i, chunk in enumerate(text_chunks):
            future = self.executor.submit(self.process_text_chunk, chunk, i)
            futures.append(future)
        
        # 使用进度条收集结果
        from tqdm import tqdm
        with tqdm(total=len(futures), desc="处理进度") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.volcano_config.timeout)
                    results.append(result)
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
                    results.append({"error": str(e), "qa_pairs": []})
                finally:
                    pbar.update(1)
                    self.stats['total_processed'] += 1
        
        return results
    
    async def process_batch_async(self, text_chunks: List[str]) -> List[Dict[str, Any]]:
        """异步批处理"""
        logger.info(f"开始批处理 {len(text_chunks)} 个文本块（异步模式）")
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(self.opt_config.max_workers or cpu_count())
        
        async def process_with_semaphore(text, chunk_id):
            async with semaphore:
                return await self.aprocess_text_chunk(text, chunk_id)
        
        # 创建所有任务
        tasks = [
            process_with_semaphore(chunk, i) 
            for i, chunk in enumerate(text_chunks)
        ]
        
        # 使用进度条执行
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            self.stats['total_processed'] += 1
            
            # 定期打印进度
            if self.stats['total_processed'] % 10 == 0:
                logger.info(f"已处理: {self.stats['total_processed']}/{len(text_chunks)}")
        
        return results
    
    def process_large_dataset(
        self,
        input_files: List[str],
        output_dir: str,
        chunk_size: int = 2000
    ) -> Dict[str, Any]:
        """处理大规模数据集"""
        logger.info(f"开始处理 {len(input_files)} 个文件")
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        for file_path in input_files:
            logger.info(f"处理文件: {file_path}")
            
            # 读取并分块
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 文本分块
            chunks = [
                content[i:i+chunk_size] 
                for i in range(0, len(content), chunk_size-200)  # 有重叠
            ]
            
            # 批处理
            if self.opt_config.use_async:
                # 异步处理
                loop = asyncio.get_event_loop()
                file_results = loop.run_until_complete(
                    self.process_batch_async(chunks)
                )
            else:
                # 同步处理
                file_results = self.process_batch_sync(chunks)
            
            # 保存结果
            output_file = os.path.join(
                output_dir,
                f"{Path(file_path).stem}_qa.json"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(file_results, f, ensure_ascii=False, indent=2)
            
            all_results.extend(file_results)
            logger.info(f"文件 {file_path} 处理完成，生成 {len(file_results)} 个结果")
        
        # 打印统计信息
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"""
处理完成统计：
- 总处理数: {self.stats['total_processed']}
- 成功: {self.stats['successful']}
- 失败: {self.stats['failed']}
- 缓存命中: {self.stats['cached']}
- 总耗时: {elapsed_time:.2f}秒
- 平均速度: {self.stats['total_processed']/elapsed_time:.2f} 块/秒
""")
        
        return {
            "results": all_results,
            "stats": self.stats
        }
    
    def cleanup(self):
        """清理资源"""
        self.executor.shutdown(wait=True)
        if self.cache and self.opt_config.cache_dir:
            # 保存缓存到文件
            cache_file = os.path.join(self.opt_config.cache_dir, "cache.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f)
            logger.info(f"缓存已保存到 {cache_file}")


def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="优化的QA生成系统")
    parser.add_argument("--input-dir", required=True, help="输入文本目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--api-key", required=True, help="火山引擎API密钥")
    parser.add_argument("--endpoint-id", required=True, help="火山引擎端点ID")
    parser.add_argument("--max-workers", type=int, default=None, help="最大并发数")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    parser.add_argument("--use-async", action="store_true", help="使用异步模式")
    parser.add_argument("--use-multiprocess", action="store_true", help="使用多进程")
    parser.add_argument("--chunk-size", type=int, default=2000, help="文本分块大小")
    
    args = parser.parse_args()
    
    # 配置火山引擎API
    volcano_config = VolcanoConfig(
        api_key=args.api_key,
        endpoint_id=args.endpoint_id
    )
    
    # 配置优化参数
    opt_config = OptimizationConfig(
        max_workers=args.max_workers,
        use_multiprocess=args.use_multiprocess,
        use_async=args.use_async,
        batch_size=args.batch_size
    )
    
    # 创建生成器
    generator = OptimizedQAGenerator(volcano_config, opt_config)
    
    # 获取输入文件
    input_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.txt'):
                input_files.append(os.path.join(root, file))
    
    logger.info(f"找到 {len(input_files)} 个输入文件")
    
    # 处理数据集
    try:
        results = generator.process_large_dataset(
            input_files,
            args.output_dir,
            chunk_size=args.chunk_size
        )
        
        # 保存汇总结果
        summary_file = os.path.join(args.output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results['stats'], f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理完成，结果保存在 {args.output_dir}")
        
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()