# # -*- coding: utf-8 -*-
# """
# vLLM HTTP客户端实现
# 用于连接到独立运行的vLLM服务器
# """
# import os
# import logging
# from typing import List, Union, Optional, Dict, Any, AsyncGenerator
# import asyncio
# import aiohttp
# import json
# from openai import OpenAI, AsyncOpenAI

# logger = logging.getLogger(__name__)


# class VLLMHTTPClient:
#     """vLLM HTTP客户端，连接到独立的vLLM服务器"""
    
#     def __init__(self, config: Dict[str, Any]):
#         """
#         初始化vLLM HTTP客户端
        
#         Args:
#             config: 配置字典，包含以下键：
#                 - base_url: vLLM服务器地址（默认: http://localhost:8000/v1）
#                 - api_key: API密钥（可选）
#                 - model_name: 模型名称（默认: qwen-vllm）
#                 - temperature: 采样温度
#                 - max_tokens: 最大生成令牌数
#                 - timeout: 请求超时时间
#         """
#         self.config = config
#         self.base_url = config.get('base_url', 'http://localhost:8000/v1')
#         self.api_key = config.get('api_key', 'fake-key')
#         self.model_name = config.get('model_name', 'qwen-vllm')
#         self.timeout = config.get('timeout', 300)
        
#         # 初始化OpenAI客户端（vLLM兼容OpenAI API）
#         self.client = OpenAI(
#             base_url=self.base_url,
#             api_key=self.api_key
#         )
        
#         self.async_client = AsyncOpenAI(
#             base_url=self.base_url,
#             api_key="fake-key"

#         )
        
#         logger.info(f"vLLM HTTP客户端初始化完成，服务器地址: {self.base_url}")
    
#     def check_connection(self) -> bool:
#         """
#         检查与vLLM服务器的连接
        
#         Returns:
#             连接是否成功
#         """
#         try:
#             # 尝试获取模型列表
#             models = self.client.models.list()
#             logger.info(f"成功连接到vLLM服务器，可用模型: {[m.id for m in models.data]}")
#             return True
#         except Exception as e:
#             logger.error(f"无法连接到vLLM服务器: {e}")
#             return False
    
#     async def acheck_connection(self) -> bool:
#         """
#         异步检查与vLLM服务器的连接
        
#         Returns:
#             连接是否成功
#         """
#         try:
#             # 尝试获取模型列表
#             models = await self.async_client.models.list()
#             logger.info(f"成功连接到vLLM服务器，可用模型: {[m.id for m in models.data]}")
#             return True
#         except Exception as e:
#             logger.error(f"无法连接到vLLM服务器: {e}")
#             return False
    
#     def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
#         """
#         生成文本
        
#         Args:
#             prompts: 输入提示词（字符串或字符串列表）
#             **kwargs: 额外的生成参数
            
#         Returns:
#             生成的文本（字符串或字符串列表）
#         """
#         # 确保prompts是列表
#         is_single = isinstance(prompts, str)
#         if is_single:
#             prompts = [prompts]
        
#         results = []
#         for prompt in prompts:
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=kwargs.get('temperature', self.config.get('temperature', 0.7)),
#                     max_tokens=kwargs.get('max_tokens', self.config.get('max_tokens', 2048)),
#                     top_p=kwargs.get('top_p', self.config.get('top_p', 0.95)),
#                     frequency_penalty=kwargs.get('frequency_penalty', 0.0),
#                     presence_penalty=kwargs.get('presence_penalty', 0.0)
#                 )
#                 results.append(response.choices[0].message.content)
#             except Exception as e:
#                 logger.error(f"生成失败: {e}")
#                 results.append("")
        
#         return results[0] if is_single else results
    
#     async def agenerate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
#         """
#         异步生成文本
        
#         Args:
#             prompts: 输入提示词（字符串或字符串列表）
#             **kwargs: 额外的生成参数
            
#         Returns:
#             生成的文本（字符串或字符串列表）
#         """
#         # 确保prompts是列表
#         is_single = isinstance(prompts, str)
#         if is_single:
#             prompts = [prompts]
        
#         results = []
#         for prompt in prompts:
#             try:
#                 response = await self.async_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=kwargs.get('temperature', self.config.get('temperature', 0.7)),
#                     max_tokens=kwargs.get('max_tokens', self.config.get('max_tokens', 2048)),
#                     top_p=kwargs.get('top_p', self.config.get('top_p', 0.95)),
#                     frequency_penalty=kwargs.get('frequency_penalty', 0.0),
#                     presence_penalty=kwargs.get('presence_penalty', 0.0)
#                 )
#                 results.append(response.choices[0].message.content)
#             except Exception as e:
#                 logger.error(f"生成失败: {e}")
#                 results.append("")
        
#         return results[0] if is_single else results
    
#     def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
#         """
#         应用聊天模板
        
#         Args:
#             messages: 消息列表
#             **kwargs: 额外参数
            
#         Returns:
#             格式化后的提示词
#         """
#         # 对于HTTP客户端，直接返回最后一条用户消息
#         # 因为聊天模板会在服务器端处理
#         for msg in reversed(messages):
#             if msg.get('role') == 'user':
#                 return msg.get('content', '')
#         return ""
    
#     def batch_generate(self, prompts: List[str], batch_size: int = 32, **kwargs) -> List[str]:
#         """
#         批量生成文本
        
#         Args:
#             prompts: 提示词列表
#             batch_size: 批次大小
#             **kwargs: 额外的生成参数
            
#         Returns:
#             生成的文本列表
#         """
#         results = []
        
#         # 分批处理
#         for i in range(0, len(prompts), batch_size):
#             batch = prompts[i:i + batch_size]
#             batch_results = self.generate(batch, **kwargs)
#             if isinstance(batch_results, str):
#                 batch_results = [batch_results]
#             results.extend(batch_results)
        
#         return results
    
#     async def abatch_generate(self, prompts: List[str], batch_size: int = 32, **kwargs) -> List[str]:
#         """
#         异步批量生成文本
        
#         Args:
#             prompts: 提示词列表
#             batch_size: 批次大小
#             **kwargs: 额外的生成参数
            
#         Returns:
#             生成的文本列表
#         """
#         results = []
        
#         # 分批处理
#         for i in range(0, len(prompts), batch_size):
#             batch = prompts[i:i + batch_size]
#             # 并发处理批次中的所有请求
#             tasks = [self.agenerate(prompt, **kwargs) for prompt in batch]
#             batch_results = await asyncio.gather(*tasks)
#             results.extend(batch_results)
        
#         return results


# def create_vllm_http_client(config: Optional[Dict[str, Any]] = None) -> VLLMHTTPClient:
#     """
#     创建vLLM HTTP客户端
    
#     Args:
#         config: 配置字典
        
#     Returns:
#         VLLMHTTPClient实例
#     """
#     if config is None:
#         # 默认配置
#         config = {
#             'base_url': os.environ.get('VLLM_SERVER_URL', 'http://localhost:8000/v1'),
#             'api_key': os.environ.get('VLLM_API_KEY', 'dummy-key'),
#             'model_name': 'qwen-vllm',
#             'temperature': 0.7,
#             'max_tokens': 2048,
#             'top_p': 0.95,
#             'timeout': 300
#         }
    
#     return VLLMHTTPClient(config)


# # 测试代码
# if __name__ == "__main__":
#     import asyncio
    
#     # 测试vLLM HTTP客户端
#     client = create_vllm_http_client()
    
#     # 检查连接
#     if client.check_connection():
#         print("成功连接到vLLM服务器")
        
#         # 测试单个生成
#         prompt = "什么是半导体？"
#         response = client.generate(prompt)
#         print(f"Question: {prompt}")
#         print(f"Answer: {response}")
        
#         # 测试异步生成
#         async def test_async():
#             response = await client.agenerate("解释一下OLED的工作原理")
#             print(f"\n异步生成结果: {response[:200]}...")
        
#         asyncio.run(test_async())
#     else:
#         print("无法连接到vLLM服务器，请确保服务器已启动")


# -*- coding: utf-8 -*-
"""
vLLM HTTP客户端实现
用于连接到独立运行的vLLM服务器
"""
import os
import logging
from typing import List, Union, Optional, Dict, Any
import time
import requests
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


class VLLMHTTPClient:
    """vLLM HTTP客户端，连接到独立的vLLM服务器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化vLLM HTTP客户端
        
        Args:
            config: 配置字典，包含以下键：
                - base_url: vLLM服务器地址（默认: http://localhost:8000/v1）
                - api_key: API密钥（可选）
                - model_name: 模型名称（默认: qwen-vllm）
                - temperature: 采样温度
                - max_tokens: 最大生成令牌数
                - timeout: 请求超时时间
        """
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:8000/v1')
        self.api_key = config.get('api_key', 'fake-key')
        self.model_name = config.get('model_name', 'qwen-vllm')
        self.timeout = config.get('timeout', 300)
        
        # 初始化OpenAI客户端（vLLM兼容OpenAI API）
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        logger.info(f"vLLM HTTP客户端初始化完成，服务器地址: {self.base_url}")
    
    def check_connection(self) -> bool:
        """
        检查与vLLM服务器的连接
        
        Returns:
            连接是否成功
        """
        try:
            # 尝试获取模型列表
            models = self.client.models.list()
            logger.info(f"成功连接到vLLM服务器，可用模型: {[m.id for m in models.data]}")
            return True
        except Exception as e:
            logger.error(f"无法连接到vLLM服务器: {e}")
            return False
    
    async def acheck_connection(self) -> bool:
        """
        异步检查与vLLM服务器的连接
        
        Returns:
            连接是否成功
        """
        try:
            # 尝试获取模型列表
            models = await self.async_client.models.list()
            logger.info(f"成功连接到vLLM服务器，可用模型: {[m.id for m in models.data]}")
            return True
        except Exception as e:
            logger.error(f"无法连接到vLLM服务器: {e}")
            return False
    
    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        生成文本（改用 requests，带超时）
        
        Args:
            prompts: 输入提示词（字符串或字符串列表）
            **kwargs: 额外的生成参数
            
        Returns:
            生成的文本（字符串或字符串列表）
        """
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        results = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = self.timeout
        
        for prompt in prompts:
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get('temperature', self.config.get('temperature', 0.7)),
                    "max_tokens": kwargs.get('max_tokens', self.config.get('max_tokens', 512)),
                    "top_p": kwargs.get('top_p', self.config.get('top_p', 0.95)),
                    "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
                    "presence_penalty": kwargs.get('presence_penalty', 0.0)
                }
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                resp.raise_for_status()
                data = resp.json()
                results.append(data["choices"][0]["message"]["content"])
            except Exception as e:
                logger.error(f"生成失败: {e}")
                results.append("")
        
        return results[0] if is_single else results
    
    def batch_generate(self, prompts: List[str], batch_size: int = 4, delay: float = 0.5, **kwargs) -> List[str]:
        """
        批量生成文本（带批次间延时）
        
        Args:
            prompts: 提示词列表
            batch_size: 批次大小
            delay: 每批次间隔秒数，默认0.5秒
            **kwargs: 额外的生成参数
            
        Returns:
            生成的文本列表
        """
        results = []
        batch_size=1
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self.generate(batch, **kwargs)
            if isinstance(batch_results, str):
                batch_results = [batch_results]
            results.extend(batch_results)
            time.sleep(delay)  # 批次间隔，防止请求压力过大
        
        return results
    
    async def agenerate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        异步生成文本（保留原调用OpenAI AsyncOpenAI接口）
        
        Args:
            prompts: 输入提示词（字符串或字符串列表）
            **kwargs: 额外的生成参数
            
        Returns:
            生成的文本（字符串或字符串列表）
        """
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        results = []
        for prompt in prompts:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get('temperature', self.config.get('temperature', 0.7)),
                    max_tokens=kwargs.get('max_tokens', self.config.get('max_tokens', 2048)),
                    top_p=kwargs.get('top_p', self.config.get('top_p', 0.95)),
                    frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                    presence_penalty=kwargs.get('presence_penalty', 0.0)
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"生成失败: {e}")
                results.append("")
        
        return results[0] if is_single else results
    
    # 其它函数保持不变...


def create_vllm_http_client(config: Optional[Dict[str, Any]] = None) -> VLLMHTTPClient:
    """
    创建vLLM HTTP客户端
    
    Args:
        config: 配置字典
        
    Returns:
        VLLMHTTPClient实例
    """
    if config is None:
        # 默认配置
        config = {
            'base_url': os.environ.get('VLLM_SERVER_URL', 'http://localhost:8000/v1'),
            'api_key': os.environ.get('VLLM_API_KEY', 'fake-key'),
            'model_name': 'qwen-vllm',
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 0.95,
            'timeout': 300
        }
    
    return VLLMHTTPClient(config)


# 测试代码
if __name__ == "__main__":
    import asyncio
    
    client = create_vllm_http_client()
    
    if client.check_connection():
        print("成功连接到vLLM服务器")
        
        prompt = "什么是半导体？"
        response = client.generate(prompt)
        print(f"Question: {prompt}")
        print(f"Answer: {response}")
        
        async def test_async():
            response = await client.agenerate("解释一下OLED的工作原理")
            print(f"\n异步生成结果: {response[:200]}...")
        
        asyncio.run(test_async())
    else:
        print("无法连接到vLLM服务器，请确保服务器已启动")
