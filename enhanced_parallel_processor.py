#!/usr/bin/env python3
"""
增强的并行处理器 - 支持大规模数据处理
支持处理2万+条数据和5000个文本的高性能并行处理器
"""

import asyncio
import aiohttp
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
import queue
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedParallelProcessor:
    """增强的并行处理器，支持大规模数据处理"""
    
    def __init__(
        self,
        max_workers: int = None,
        batch_size: int = 100,
        use_multiprocessing: bool = True,
        max_concurrent_api_calls: int = 50,
        api_timeout: int = 300
    ):
        """
        初始化并行处理器
        
        Args:
            max_workers: 最大工作进程/线程数（默认为CPU核心数）
            batch_size: 批处理大小
            use_multiprocessing: 是否使用多进程（True）或多线程（False）
            max_concurrent_api_calls: 最大并发API调用数
            api_timeout: API调用超时时间（秒）
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.batch_size = batch_size
        self.use_multiprocessing = use_multiprocessing
        self.max_concurrent_api_calls = max_concurrent_api_calls
        self.api_timeout = api_timeout
        
        # 创建信号量控制并发API调用
        self.api_semaphore = asyncio.Semaphore(max_concurrent_api_calls)
        
        logger.info(f"初始化并行处理器: workers={self.max_workers}, batch_size={self.batch_size}, "
                   f"mode={'multiprocessing' if use_multiprocessing else 'threading'}")
    
    def process_in_batches(
        self,
        items: List[Any],
        process_func: Callable,
        desc: str = "Processing"
    ) -> List[Any]:
        """
        批量处理数据（同步版本）
        
        Args:
            items: 待处理的数据列表
            process_func: 处理函数
            desc: 进度条描述
            
        Returns:
            处理结果列表
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        # 选择执行器
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # 创建批次
            batches = [items[i:i + self.batch_size] 
                      for i in range(0, len(items), self.batch_size)]
            
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self._process_batch, batch, process_func): idx
                for idx, batch in enumerate(batches)
            }
            
            # 使用tqdm显示进度
            with tqdm(total=total_batches, desc=desc) as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"批次 {batch_idx} 处理失败: {e}")
                        pbar.update(1)
        
        return results
    
    async def process_in_batches_async(
        self,
        items: List[Any],
        async_process_func: Callable,
        desc: str = "Processing"
    ) -> List[Any]:
        """
        批量处理数据（异步版本）
        
        Args:
            items: 待处理的数据列表
            async_process_func: 异步处理函数
            desc: 进度条描述
            
        Returns:
            处理结果列表
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        # 创建批次
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        # 创建进度条
        pbar = tqdm(total=total_batches, desc=desc)
        
        # 异步处理所有批次
        tasks = []
        for batch_idx, batch in enumerate(batches):
            task = self._process_batch_async(batch, async_process_func, batch_idx, pbar)
            tasks.append(task)
        
        # 等待所有任务完成
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果
        for batch_idx, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                logger.error(f"批次 {batch_idx} 处理失败: {batch_result}")
            else:
                results.extend(batch_result)
        
        pbar.close()
        return results
    
    def _process_batch(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """处理单个批次（同步）"""
        results = []
        for item in batch:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"处理项目失败: {e}")
                results.append(None)
        return results
    
    async def _process_batch_async(
        self,
        batch: List[Any],
        async_process_func: Callable,
        batch_idx: int,
        pbar: tqdm
    ) -> List[Any]:
        """处理单个批次（异步）"""
        results = []
        
        # 创建批次内的并发任务
        tasks = []
        for item in batch:
            # 使用信号量限制并发
            task = self._process_item_with_semaphore(item, async_process_func)
            tasks.append(task)
        
        # 等待批次内所有任务完成
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for item_idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"批次 {batch_idx} 项目 {item_idx} 处理失败: {result}")
                results.append(None)
            else:
                results.append(result)
        
        pbar.update(1)
        return results
    
    async def _process_item_with_semaphore(
        self,
        item: Any,
        async_process_func: Callable
    ) -> Any:
        """使用信号量限制并发处理单个项目"""
        async with self.api_semaphore:
            try:
                return await async_process_func(item)
            except Exception as e:
                logger.error(f"处理项目失败: {e}")
                raise


class VolcanoAPIClient:
    """火山引擎API客户端 - 支持Qwen3-235B-A22B-Instruct模型"""
    
    def __init__(
        self,
        api_key: str = None,
        endpoint_id: str = None,
        region: str = "cn-beijing",
        timeout: int = 300
    ):
        """
        初始化火山API客户端
        
        Args:
            api_key: 火山引擎API密钥
            endpoint_id: 模型端点ID
            region: 区域
            timeout: 超时时间
        """
        self.api_key = api_key or os.environ.get("VOLCANO_API_KEY")
        self.endpoint_id = endpoint_id or os.environ.get("VOLCANO_ENDPOINT_ID")
        self.region = region
        self.timeout = timeout
        self.base_url = f"https://ark.cn-beijing.volces.com/api/v3"
        
        if not self.api_key:
            logger.warning("未设置火山API密钥，将使用本地vLLM服务")
            self.use_local = True
            self.local_url = "http://localhost:8000/v1"
        else:
            self.use_local = False
            
        self.session = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        调用模型生成文本
        
        Args:
            prompt: 输入提示词
            temperature: 采样温度
            max_tokens: 最大生成令牌数
            top_p: Top-p采样参数
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        if self.use_local:
            return await self._call_local_vllm(prompt, temperature, max_tokens, top_p, **kwargs)
        else:
            return await self._call_volcano_api(prompt, temperature, max_tokens, top_p, **kwargs)
    
    async def _call_local_vllm(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> str:
        """调用本地vLLM服务"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "model": "qwen-vllm",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        try:
            async with self.session.post(
                f"{self.local_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"调用本地vLLM失败: {e}")
            raise
    
    async def _call_volcano_api(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> str:
        """调用火山引擎API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Qwen3-235B-A22B-Instruct",  # 指定模型
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"调用火山API失败: {e}")
            raise
    
    async def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 提示词列表
            temperature: 采样温度
            max_tokens: 最大生成令牌数
            **kwargs: 其他参数
            
        Returns:
            生成的文本列表
        """
        tasks = []
        for prompt in prompts:
            task = self.generate(prompt, temperature, max_tokens, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"生成失败 (prompt {idx}): {result}")
                processed_results.append("")
            else:
                processed_results.append(result)
        
        return processed_results


class LargeScaleQAProcessor:
    """大规模QA处理器 - 整合并行处理和API调用"""
    
    def __init__(
        self,
        parallel_processor: EnhancedParallelProcessor,
        api_client: VolcanoAPIClient,
        output_dir: str = "data/output"
    ):
        """
        初始化大规模QA处理器
        
        Args:
            parallel_processor: 并行处理器实例
            api_client: API客户端实例
            output_dir: 输出目录
        """
        self.parallel_processor = parallel_processor
        self.api_client = api_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_texts(
        self,
        text_files: List[str],
        target_qa_count: int = 20000
    ) -> Dict[str, Any]:
        """
        处理文本文件生成QA数据
        
        Args:
            text_files: 文本文件路径列表
            target_qa_count: 目标QA数量
            
        Returns:
            处理结果统计
        """
        start_time = time.time()
        
        logger.info(f"开始处理 {len(text_files)} 个文本文件，目标生成 {target_qa_count} 条QA")
        
        # 第一步：读取和预处理文本
        logger.info("步骤1: 读取和预处理文本...")
        all_chunks = self._preprocess_texts(text_files)
        logger.info(f"生成了 {len(all_chunks)} 个文本块")
        
        # 第二步：并行生成问题
        logger.info("步骤2: 并行生成问题...")
        questions = await self._generate_questions_parallel(all_chunks, target_qa_count)
        logger.info(f"生成了 {len(questions)} 个问题")
        
        # 第三步：并行生成答案
        logger.info("步骤3: 并行生成答案...")
        qa_pairs = await self._generate_answers_parallel(questions)
        logger.info(f"生成了 {len(qa_pairs)} 个完整的QA对")
        
        # 第四步：保存结果
        output_file = self.output_dir / f"qa_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        elapsed_time = time.time() - start_time
        
        stats = {
            "total_files": len(text_files),
            "total_chunks": len(all_chunks),
            "total_questions": len(questions),
            "total_qa_pairs": len(qa_pairs),
            "elapsed_time": elapsed_time,
            "output_file": str(output_file)
        }
        
        logger.info(f"处理完成！耗时: {elapsed_time:.2f}秒")
        logger.info(f"结果保存到: {output_file}")
        
        return stats
    
    def _preprocess_texts(self, text_files: List[str]) -> List[Dict[str, Any]]:
        """预处理文本文件"""
        all_chunks = []
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 分块处理（每2000字符一块）
                chunk_size = 2000
                chunks = [content[i:i+chunk_size] 
                         for i in range(0, len(content), chunk_size)]
                
                for idx, chunk in enumerate(chunks):
                    all_chunks.append({
                        "file": file_path,
                        "chunk_id": idx,
                        "content": chunk
                    })
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {e}")
        
        return all_chunks
    
    async def _generate_questions_parallel(
        self,
        chunks: List[Dict[str, Any]],
        target_count: int
    ) -> List[Dict[str, Any]]:
        """并行生成问题"""
        questions = []
        
        # 计算每个块需要生成的问题数
        questions_per_chunk = max(1, target_count // len(chunks))
        
        # 定义异步处理函数
        async def generate_questions_for_chunk(chunk):
            prompt = f"""基于以下文本内容，生成{questions_per_chunk}个高质量的问题。
要求：
1. 问题要涵盖事实型、比较型、推理型和开放型
2. 问题要有深度，能够考察对内容的理解
3. 返回JSON格式：[{{"question": "...", "type": "..."}}]

文本内容：
{chunk['content'][:1000]}  # 限制长度避免超出token限制

请生成问题："""
            
            try:
                response = await self.api_client.generate(prompt)
                # 解析响应
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    questions_data = json.loads(json_match.group())
                    for q in questions_data:
                        q['source_chunk'] = chunk['chunk_id']
                        q['source_file'] = chunk['file']
                        q['context'] = chunk['content']
                    return questions_data
            except Exception as e:
                logger.error(f"生成问题失败: {e}")
                return []
            
            return []
        
        # 使用并行处理器处理所有块
        chunk_questions = await self.parallel_processor.process_in_batches_async(
            chunks,
            generate_questions_for_chunk,
            desc="生成问题"
        )
        
        # 展平结果
        for chunk_q in chunk_questions:
            if chunk_q:
                questions.extend(chunk_q)
        
        # 限制到目标数量
        return questions[:target_count]
    
    async def _generate_answers_parallel(
        self,
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """并行生成答案"""
        
        async def generate_answer_for_question(question):
            prompt = f"""请为以下问题提供详细、准确的答案。

问题：{question['question']}
问题类型：{question.get('type', '未知')}

相关上下文：
{question.get('context', '')[:1500]}

要求：
1. 答案要准确、专业、有深度
2. 使用Chain of Thought方式，先分析再回答
3. 返回JSON格式：{{"reasoning": "...", "answer": "..."}}

请生成答案："""
            
            try:
                response = await self.api_client.generate(prompt)
                # 解析响应
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    answer_data = json.loads(json_match.group())
                    return {
                        **question,
                        "reasoning": answer_data.get("reasoning", ""),
                        "answer": answer_data.get("answer", "")
                    }
            except Exception as e:
                logger.error(f"生成答案失败: {e}")
                return {**question, "answer": "", "reasoning": ""}
            
            return {**question, "answer": "", "reasoning": ""}
        
        # 使用并行处理器处理所有问题
        qa_pairs = await self.parallel_processor.process_in_batches_async(
            questions,
            generate_answer_for_question,
            desc="生成答案"
        )
        
        # 过滤掉空答案
        return [qa for qa in qa_pairs if qa and qa.get("answer")]


async def main():
    """主函数示例"""
    
    # 配置参数
    config = {
        "max_workers": mp.cpu_count(),  # 使用所有CPU核心
        "batch_size": 100,  # 增大批处理大小
        "use_multiprocessing": True,  # 使用多进程
        "max_concurrent_api_calls": 50,  # 并发API调用数
        "api_timeout": 300
    }
    
    # 创建并行处理器
    parallel_processor = EnhancedParallelProcessor(**config)
    
    # 创建API客户端（支持火山API或本地vLLM）
    async with VolcanoAPIClient() as api_client:
        # 创建大规模处理器
        processor = LargeScaleQAProcessor(
            parallel_processor=parallel_processor,
            api_client=api_client,
            output_dir="data/output"
        )
        
        # 获取所有文本文件
        text_dir = Path("data/texts")
        text_files = list(text_dir.glob("*.txt"))[:5000]  # 处理5000个文本文件
        
        if not text_files:
            logger.error("未找到文本文件")
            return
        
        # 处理文本生成QA
        stats = await processor.process_texts(
            text_files=[str(f) for f in text_files],
            target_qa_count=20000  # 目标生成2万条QA
        )
        
        # 打印统计信息
        print("\n=== 处理统计 ===")
        print(f"处理文件数: {stats['total_files']}")
        print(f"文本块数: {stats['total_chunks']}")
        print(f"生成问题数: {stats['total_questions']}")
        print(f"完整QA对数: {stats['total_qa_pairs']}")
        print(f"总耗时: {stats['elapsed_time']:.2f}秒")
        print(f"平均速度: {stats['total_qa_pairs'] / stats['elapsed_time']:.2f} QA/秒")
        print(f"结果文件: {stats['output_file']}")


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行主函数
    asyncio.run(main())