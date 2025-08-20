#!/usr/bin/env python3
"""
Volcano API客户端 - 支持火山引擎API调用
支持deepseek-r1等模型的调用
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class VolcanoAPIClient:
    """火山引擎API客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Volcano API客户端
        
        Args:
            config: 配置字典，包含API密钥、端点等信息
        """
        self.config = config
        self.api_config = config.get('api', {})
        self.model_config = config.get('models', {}).get('volcano', {})
        self.parallel_config = config.get('parallel_processing', {})
        
        # API配置
        self.api_key = self.api_config.get('volcano_api_key', os.getenv('VOLCANO_API_KEY'))
        self.endpoint_id = self.api_config.get('volcano_endpoint_id', os.getenv('VOLCANO_ENDPOINT_ID'))
        self.region = self.api_config.get('volcano_region', 'cn-beijing')
        
        # 模型配置
        self.model_name = self.model_config.get('model_name', 'deepseek-r1')
        self.temperature = self.model_config.get('temperature', 0.7)
        self.max_tokens = self.model_config.get('max_tokens', 4096)
        self.top_p = self.model_config.get('top_p', 0.95)
        
        # 并行处理配置
        self.max_workers = self.parallel_config.get('max_workers', 16)
        self.max_concurrent_api_calls = self.parallel_config.get('max_concurrent_api_calls', 50)
        self.api_timeout = self.parallel_config.get('api_timeout', 300)
        self.retry_attempts = self.parallel_config.get('retry_attempts', 3)
        self.retry_delay = self.parallel_config.get('retry_delay', 2)
        
        # 创建HTTP会话
        self.session = self._create_session()
        
        # 线程池执行器
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # API调用速率限制
        self.semaphore = threading.Semaphore(self.max_concurrent_api_calls)
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'start_time': time.time()
        }
        
        logger.info(f"Volcano API客户端初始化完成 - Model: {self.model_name}, Workers: {self.max_workers}")
    
    def _create_session(self) -> requests.Session:
        """创建HTTP会话，配置重试策略"""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _build_api_url(self) -> str:
        """构建API URL"""
        # 火山引擎API端点格式
        if self.endpoint_id:
            return f"https://ark.cn-beijing.volces.com/api/v3/endpoints/{self.endpoint_id}/chat/completions"
        else:
            # 默认使用模型端点
            return f"https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    
    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _build_request_body(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """构建请求体"""
        messages = kwargs.get('messages')
        if not messages:
            # 如果没有提供messages，将prompt转换为messages格式
            messages = [{"role": "user", "content": prompt}]
        
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "top_p": kwargs.get('top_p', self.top_p),
            "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
            "presence_penalty": kwargs.get('presence_penalty', 0.0),
            "stream": False
        }
        
        # 添加其他可选参数
        if 'stop' in kwargs:
            body['stop'] = kwargs['stop']
        if 'n' in kwargs:
            body['n'] = kwargs['n']
            
        return body
    
    def _make_api_call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        执行单个API调用
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        with self.semaphore:
            try:
                url = self._build_api_url()
                headers = self._build_headers()
                body = self._build_request_body(prompt, **kwargs)
                
                response = self.session.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.api_timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                self.stats['successful_requests'] += 1
                if 'usage' in result:
                    self.stats['total_tokens'] += result['usage'].get('total_tokens', 0)
                
                return result
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API调用失败: {e}")
                self.stats['failed_requests'] += 1
                raise
            finally:
                self.stats['total_requests'] += 1
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本（同步接口）
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        try:
            result = self._make_api_call(prompt, **kwargs)
            if result and 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            return ""
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return ""
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """
        异步生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate, prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        批量生成文本（多线程并行）
        
        Args:
            prompts: 输入提示列表
            **kwargs: 其他参数
            
        Returns:
            生成的文本列表
        """
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 创建任务
            future_to_index = {
                executor.submit(self.generate, prompt, **kwargs): i 
                for i, prompt in enumerate(prompts)
            }
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"批量生成索引 {index} 失败: {e}")
                    results[index] = ""
        
        return results
    
    async def abatch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        异步批量生成文本
        
        Args:
            prompts: 输入提示列表
            **kwargs: 其他参数
            
        Returns:
            生成的文本列表
        """
        tasks = [self.agenerate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def stream_generate(self, prompts: List[str], callback=None, **kwargs):
        """
        流式批量生成（带进度回调）
        
        Args:
            prompts: 输入提示列表
            callback: 进度回调函数
            **kwargs: 其他参数
            
        Yields:
            (index, result) 元组
        """
        total = len(prompts)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 创建任务
            futures = {
                executor.submit(self.generate, prompt, **kwargs): i 
                for i, prompt in enumerate(prompts)
            }
            
            # 流式返回结果
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    
                    if callback:
                        callback(completed, total)
                    
                    yield index, result
                    
                except Exception as e:
                    logger.error(f"流式生成索引 {index} 失败: {e}")
                    completed += 1
                    
                    if callback:
                        callback(completed, total)
                    
                    yield index, ""
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        对话接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            生成的回复
        """
        return self.generate("", messages=messages, **kwargs)
    
    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        异步对话接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            生成的回复
        """
        return await self.agenerate("", messages=messages, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        elapsed_time = time.time() - self.stats['start_time']
        return {
            **self.stats,
            'elapsed_time': elapsed_time,
            'requests_per_second': self.stats['total_requests'] / elapsed_time if elapsed_time > 0 else 0,
            'success_rate': self.stats['successful_requests'] / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'start_time': time.time()
        }
    
    def close(self):
        """关闭客户端"""
        self.executor.shutdown(wait=True)
        self.session.close()
        logger.info("Volcano API客户端已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VolcanoModelManager:
    """Volcano模型管理器 - 兼容现有接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化管理器"""
        self.config = config
        self.client = VolcanoAPIClient(config)
        self.model_name = self.client.model_name
        logger.info(f"Volcano模型管理器初始化完成 - Model: {self.model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        return self.client.generate(prompt, **kwargs)
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """异步生成文本"""
        return await self.client.agenerate(prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成文本"""
        return self.client.batch_generate(prompts, **kwargs)
    
    async def abatch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """异步批量生成文本"""
        return await self.client.abatch_generate(prompts, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """对话接口"""
        return self.client.chat(messages, **kwargs)
    
    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """异步对话接口"""
        return await self.client.achat(messages, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.client.get_stats()
    
    def close(self):
        """关闭管理器"""
        self.client.close()


# 测试代码
if __name__ == "__main__":
    import asyncio
    
    # 加载配置
    with open('config_volcano.json', 'r') as f:
        config = json.load(f)
    
    # 创建客户端
    client = VolcanoAPIClient(config)
    
    # 测试单个生成
    print("测试单个生成...")
    result = client.generate("什么是半导体？请简要回答。")
    print(f"结果: {result[:200]}...")
    
    # 测试批量生成
    print("\n测试批量生成...")
    prompts = [
        "什么是OLED？",
        "LCD和OLED的区别是什么？",
        "TFT技术的原理是什么？"
    ]
    results = client.batch_generate(prompts)
    for i, result in enumerate(results):
        print(f"问题 {i+1}: {result[:100]}...")
    
    # 打印统计信息
    print("\n统计信息:")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 关闭客户端
    client.close()