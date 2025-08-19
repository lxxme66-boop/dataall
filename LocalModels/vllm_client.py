# -*- coding: utf-8 -*-
"""
VLLM客户端实现
支持本地VLLM服务的调用
"""
import os
import logging
from typing import List, Union, Optional, Dict, Any, AsyncGenerator
import asyncio
import aiohttp
import json
from tqdm import tqdm

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入vllm
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not installed. Install with: pip install vllm")
    VLLM_AVAILABLE = False
    
    # 定义mock类以避免导入错误
    class LLM:
        """Mock LLM class"""
        def __init__(self, **kwargs):
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
    
    class SamplingParams:
        """Mock SamplingParams class"""
        def __init__(self, **kwargs):
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")

# 尝试导入transformers
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed")


class VLLMClient:
    """vLLM客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化vLLM客户端
        
        Args:
            config: 配置字典，包含以下键：
                - model_path: 模型路径
                - gpu_memory_utilization: GPU内存利用率
                - tensor_parallel_size: 张量并行大小
                - max_model_len: 最大模型长度
                - temperature: 采样温度
                - repetition_penalty: 重复惩罚
                - top_p: Top-p采样
                - top_k: Top-k采样
                - max_tokens: 最大生成令牌数
        """
        self.config = config
        self.model_path = config.get('model_path', '/mnt/data/LLM/lhy/models/Qwen3/Qwen3-235B-A22B')
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        
        # 初始化模型
        if VLLM_AVAILABLE:
            self._initialize_vllm()
        else:
            logger.error("vLLM not available. Please install vllm package.")
    
    def _initialize_vllm(self):
        """初始化vLLM模型"""
        try:
            # 初始化LLM
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                gpu_memory_utilization=self.config.get('gpu_memory_utilization', 0.95),
                tensor_parallel_size=self.config.get('tensor_parallel_size', 1),
                max_model_len=self.config.get('max_model_len', 40960)
            )
            
            # 初始化tokenizer
            if TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            
            # 设置采样参数
            stop_token_ids = self.config.get('stop_token_ids', [151329, 151336, 151338])
            self.sampling_params = SamplingParams(
                temperature=self.config.get('temperature', 0.6),
                repetition_penalty=self.config.get('repetition_penalty', 1.1),
                min_p=self.config.get('min_p', 0),
                top_p=self.config.get('top_p', 0.95),
                top_k=self.config.get('top_k', 40),
                max_tokens=self.config.get('max_tokens', 4096),
                stop_token_ids=stop_token_ids
            )
            
            logger.info(f"vLLM initialized with model: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
    
    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        生成文本
        
        Args:
            prompts: 输入提示词（字符串或字符串列表）
            **kwargs: 额外的采样参数
            
        Returns:
            生成的文本（字符串或字符串列表）
        """
        if not self.llm:
            raise RuntimeError("vLLM not initialized")
        
        # 确保prompts是列表
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        # 更新采样参数
        sampling_params = self._update_sampling_params(**kwargs)
        
        # 生成
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        
        # 提取结果
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text)
            else:
                results.append("")
        
        return results[0] if is_single else results
    
    async def agenerate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        异步生成文本
        
        Args:
            prompts: 输入提示词（字符串或字符串列表）
            **kwargs: 额外的采样参数
            
        Returns:
            生成的文本（字符串或字符串列表）
        """
        # vLLM的generate方法是同步的，这里用线程池包装
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompts, **kwargs)
    
    def _update_sampling_params(self, **kwargs) -> SamplingParams:
        """更新采样参数"""
        if not self.sampling_params:
            return SamplingParams(**kwargs)
        
        # 创建新的采样参数
        params = {
            'temperature': kwargs.get('temperature', self.sampling_params.temperature),
            'repetition_penalty': kwargs.get('repetition_penalty', self.sampling_params.repetition_penalty),
            'top_p': kwargs.get('top_p', self.sampling_params.top_p),
            'top_k': kwargs.get('top_k', self.sampling_params.top_k),
            'max_tokens': kwargs.get('max_tokens', self.sampling_params.max_tokens),
            'stop_token_ids': kwargs.get('stop_token_ids', self.sampling_params.stop_token_ids)
        }
        
        # 添加其他参数
        if 'min_p' in kwargs:
            params['min_p'] = kwargs['min_p']
        
        return SamplingParams(**params)
    
    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        应用聊天模板
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            格式化后的提示词
        """
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs
            )
        else:
            # 简单的模板回退
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt += f"System: {content}\n\n"
                elif role == 'user':
                    prompt += f"User: {content}\n\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "
            return prompt
    
    def batch_generate(self, prompts: List[str], batch_size: int = 32, **kwargs) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 提示词列表
            batch_size: 批次大小
            **kwargs: 额外的采样参数
            
        Returns:
            生成的文本列表
        """
        if not self.llm:
            raise RuntimeError("vLLM not initialized")
        
        results = []
        
        # 分批处理
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self.generate(batch, **kwargs)
            results.extend(batch_results)
        
        return results
    
    async def abatch_generate(self, prompts: List[str], batch_size: int = 32, **kwargs) -> List[str]:
        """
        异步批量生成文本
        
        Args:
            prompts: 提示词列表
            batch_size: 批次大小
            **kwargs: 额外的采样参数
            
        Returns:
            生成的文本列表
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.batch_generate, prompts, batch_size, **kwargs)


def create_vllm_client(config: Optional[Dict[str, Any]] = None) -> VLLMClient:
    """
    创建vLLM客户端
    
    Args:
        config: 配置字典
        
    Returns:
        VLLMClient实例
    """
    if config is None:
        # 默认配置
        config = {
            'model_path': os.environ.get('VLLM_MODEL_PATH', '/mnt/storage/models/Skywork/Skywork-R1V3-38B'),
            'gpu_memory_utilization': 0.95,
            'tensor_parallel_size': 2,
            'max_model_len': 40960,
            'temperature': 0.6,
            'repetition_penalty': 1.1,
            'top_p': 0.95,
            'top_k': 40,
            'max_tokens': 4096,
            'stop_token_ids': [151329, 151336, 151338]
        }
    
    return VLLMClient(config)


# 测试代码
if __name__ == "__main__":
    # 测试vLLM客户端
    if VLLM_AVAILABLE:
        client = create_vllm_client()
        
        # 测试单个生成
        prompt = "什么是半导体？"
        response = client.generate(prompt)
        print(f"Question: {prompt}")
        print(f"Answer: {response}")
        
        # 测试批量生成
        prompts = [
            "什么是OLED？",
            "解释一下TFT的工作原理",
            "IGZO材料有什么特点？"
        ]
        responses = client.batch_generate(prompts, batch_size=2)
        for p, r in zip(prompts, responses):
            print(f"\nQ: {p}")
            print(f"A: {r[:200]}...")
    else:
        print("vLLM not available. Please install vllm package.")