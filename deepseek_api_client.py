#!/usr/bin/env python3
"""
DeepSeek API客户端 - 支持DeepSeek-R1模型
集成并行处理和批量推理功能
"""

import asyncio
import aiohttp
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional
from asyncio import Semaphore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekAPIClient:
    """DeepSeek API客户端 - 支持R1模型"""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-r1",
        timeout: int = 300,
        max_concurrent_calls: int = 10
    ):
        """
        初始化DeepSeek API客户端
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
            model: 使用的模型名称
            timeout: 超时时间（秒）
            max_concurrent_calls: 最大并发调用数
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.session = None
        self.semaphore = Semaphore(max_concurrent_calls)
        
        if not self.api_key:
            raise ValueError("DeepSeek API密钥未设置。请设置DEEPSEEK_API_KEY环境变量或传入api_key参数")
    
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
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        调用DeepSeek模型生成文本
        
        Args:
            prompt: 输入提示词
            temperature: 采样温度
            max_tokens: 最大生成令牌数
            top_p: Top-p采样参数
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        async with self.semaphore:
            return await self._call_api(prompt, temperature, max_tokens, top_p, stream, **kwargs)
    
    async def _call_api(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stream: bool,
        **kwargs
    ) -> str:
        """内部API调用方法"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        # 添加额外参数
        payload.update(kwargs)
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 检查响应格式
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"]
                        else:
                            logger.error(f"API响应格式错误: {result}")
                            raise ValueError(f"API响应缺少'choices'字段: {result}")
                    
                    elif response.status == 429:  # Rate limit
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"达到速率限制，等待{wait_time}秒后重试...")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"API调用失败，状态码: {response.status}, 错误: {error_text}")
                        raise Exception(f"API调用失败: {error_text}")
                        
            except asyncio.TimeoutError:
                logger.error(f"API调用超时（尝试 {attempt + 1}/{max_retries}）")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                raise
            
            except Exception as e:
                logger.error(f"调用DeepSeek API失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                raise
        
        raise Exception(f"API调用失败，已重试{max_retries}次")
    
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
        
        # 处理结果，将异常转换为空字符串
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量生成第{i}个提示词失败: {result}")
                processed_results.append("")
            else:
                processed_results.append(result)
        
        return processed_results


class DeepSeekQAProcessor:
    """使用DeepSeek API的QA处理器"""
    
    def __init__(
        self,
        api_client: DeepSeekAPIClient,
        max_workers: int = 16,
        batch_size: int = 10
    ):
        """
        初始化QA处理器
        
        Args:
            api_client: DeepSeek API客户端
            max_workers: 最大并发工作数
            batch_size: 批处理大小
        """
        self.api_client = api_client
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    async def generate_questions_from_text(
        self,
        text: str,
        num_questions: int = 10
    ) -> List[Dict[str, str]]:
        """
        从文本生成问题
        
        Args:
            text: 输入文本
            num_questions: 生成问题数量
            
        Returns:
            问题列表
        """
        prompt = f"""请基于以下文本内容生成{num_questions}个高质量的问答对。

要求：
1. 问题要有深度，涵盖文本的关键信息
2. 问题类型要多样化（事实型、推理型、分析型等）
3. 避免过于简单或重复的问题
4. 每个问题都要能从文本中找到答案

文本内容：
{text[:3000]}  # 限制文本长度避免超出token限制

请以JSON格式输出，格式如下：
[
    {{"question": "问题1", "context": "相关文本片段"}},
    {{"question": "问题2", "context": "相关文本片段"}},
    ...
]
"""
        
        try:
            response = await self.api_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # 解析JSON响应
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                return questions[:num_questions]
            else:
                logger.error(f"无法从响应中提取JSON: {response[:200]}")
                return []
                
        except Exception as e:
            logger.error(f"生成问题失败: {e}")
            return []
    
    async def generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """
        基于上下文生成答案
        
        Args:
            question: 问题
            context: 上下文
            
        Returns:
            答案
        """
        prompt = f"""请基于以下上下文回答问题。

上下文：
{context[:2000]}

问题：{question}

请提供准确、详细的答案。如果上下文中没有足够信息，请说明。

答案："""
        
        try:
            answer = await self.api_client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000
            )
            return answer.strip()
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return ""
    
    async def process_text_batch(
        self,
        texts: List[str],
        questions_per_text: int = 10
    ) -> List[Dict[str, Any]]:
        """
        批量处理文本生成QA对
        
        Args:
            texts: 文本列表
            questions_per_text: 每个文本生成的问题数
            
        Returns:
            QA对列表
        """
        all_qa_pairs = []
        
        # 分批处理文本
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.info(f"处理批次 {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            # 并行生成问题
            question_tasks = []
            for text in batch:
                task = self.generate_questions_from_text(text, questions_per_text)
                question_tasks.append(task)
            
            batch_questions = await asyncio.gather(*question_tasks)
            
            # 并行生成答案
            answer_tasks = []
            for text, questions in zip(batch, batch_questions):
                for q in questions:
                    if "question" in q and "context" in q:
                        task = self.generate_answer(q["question"], q.get("context", text[:1000]))
                        answer_tasks.append((q["question"], task))
            
            # 收集答案
            for question, answer_task in answer_tasks:
                try:
                    answer = await answer_task
                    if answer:
                        all_qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "metadata": {
                                "source": "deepseek-r1",
                                "timestamp": time.time()
                            }
                        })
                except Exception as e:
                    logger.error(f"处理QA对失败: {e}")
        
        return all_qa_pairs