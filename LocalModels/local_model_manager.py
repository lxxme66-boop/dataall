"""
本地模型管理器
用于管理和选择不同的本地模型后端
"""
import logging
from typing import Dict, Any, Optional, Union
import asyncio
import logging
from .ollama_client import OllamaClient, create_ollama_client

# 尝试导入vLLM
try:
    from .vllm_client import VLLMClient, create_vllm_client
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# 导入vLLM HTTP客户端
try:
    from .vllm_http_client import VLLMHTTPClient, create_vllm_http_client
    VLLM_HTTP_AVAILABLE = True
except ImportError:
    VLLM_HTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocalModelManager:

    # def __init__(self, config: Dict[str, Any]):
    #     """
    #     初始化模型管理器
        
    #     Args:
    #         config: 配置字典
    #     """
    #     self.config = config
    #     self.models = {}
    #     self.default_backend = config.get('default_backend', 'ollama')
        
    # def get_client(self, backend: Optional[str] = None):
    #     """
    #     获取指定后端的客户端
        
    #     Args:
    #         backend: 后端名称 ('ollama', 'vllm')
            
    #     Returns:
    #         对应的客户端实例
    #     """
    #     if backend is None:
    #         backend = self.default_backend
            
    #     if backend in self.models:
    #         return self.models[backend]
            
    #     if backend == 'ollama':
    #         client = create_ollama_client(self.config.get('ollama', {}))
    #         self.models[backend] = client
    #         return client
            
    #     elif backend == 'vllm' and VLLM_AVAILABLE:
    #         client = create_vllm_client(self.config.get('vllm', {}))
    #         self.models[backend] = client
    #         return client
            
    #     elif backend == 'vllm_http' and VLLM_HTTP_AVAILABLE:
    #         client = create_vllm_http_client(self.config.get('vllm_http', {}))
    #         self.models[backend] = client
    #         return client
    def __init__(self, config):
        self.config = config
        self.backend = config['models']['local_models']['default_backend']
        self.clients = {}
        
    def get_client(self,backend:Optional[str]=None):
        if backend is None:
            backend=self.backend

        if backend == "vllm_http":
            if "vllm_http" not in self.clients:
                from .vllm_http_client import VLLMHTTPClient
                vllm_config = self.config['models']['local_models']['vllm_http']
                self.clients["vllm_http"] = VLLMHTTPClient(vllm_config)
            return self.clients["vllm_http"]
        elif backend == "vllm":
            if "vllm" not in self.clients:
                from .vllm_client import VLLMClient
                vllm_config = self.config['models']['local_models']['vllm']
                self.clients["vllm"] = VLLMClient(vllm_config)
            return self.clients["vllm"]
            
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    
    # async def generate(self, prompt: str, backend: Optional[str] = None, **kwargs):
    #     """
    #     使用指定后端生成文本
        
    #     Args:
    #         prompt: 输入提示词
    #         backend: 后端名称
    #         **kwargs: 额外参数
            
    #     Returns:
    #         生成的文本
    #     """
    #     client = self.get_client(backend)
        
    #     if hasattr(client, 'agenerate'):
    #         return await client.agenerate(prompt, **kwargs)
    #     else:
    #         # 同步方法的异步包装
    #         import asyncio
    #         loop = asyncio.get_event_loop()
    #         return await loop.run_in_executor(None, client.generate, prompt, **kwargs)
    async def generate(self, prompt: str, system_prompt: str = "", backend: Optional[str] = None, **kwargs):
        """
        使用指定后端生成文本
        
        Args:
            prompt: 输入提示词
            system_prompt: 系统提示词（可选）
            backend: 后端名称（可选）
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        client = self.get_client(backend)
        
        # 根据后端类型处理输入
        if hasattr(client, 'apply_chat_template'):
            # 对于支持聊天模板的客户端（vllm, vllm_http）
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = client.apply_chat_template(messages)
            
            if hasattr(client, 'agenerate'):
                return await client.agenerate(formatted_prompt, **kwargs)
            else:
                return await client.generate(formatted_prompt, **kwargs)
        else:
            # 对于其他后端（如 ollama）
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            if hasattr(client, 'agenerate'):
                return await client.agenerate(full_prompt, **kwargs)
            else:
                # 同步方法的异步包装
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, client.generate, full_prompt, **kwargs)
    
    def list_available_backends(self):
        """列出可用的后端"""
        backends = ['ollama']
        if VLLM_AVAILABLE:
            backends.append('vllm')
        if VLLM_HTTP_AVAILABLE:
            backends.append('vllm_http')
        return backends
    
    def is_available(self):
        """
        检查当前配置的后端是否可用
        
        Returns:
            bool: 如果当前后端可用则返回True，否则返回False
        """
        try:
            if self.backend == "vllm_http":
                return VLLM_HTTP_AVAILABLE
            elif self.backend == "vllm":
                return VLLM_AVAILABLE
            elif self.backend == "ollama":
                # Ollama默认认为是可用的，因为它不需要额外依赖
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error checking backend availability: {e}")
            return False
    
    def get_backend_name(self):
        """
        获取当前使用的后端名称
        
        Returns:
            str: 后端名称
        """
        return self.backend



def create_local_model_manager(config: Optional[Dict[str, Any]] = None) -> LocalModelManager:
    """
    创建本地模型管理器
    
    Args:
        config: 配置字典
        
    Returns:
        LocalModelManager实例
    """
    if config is None:
        config = {
            'default_backend': 'ollama',
            'ollama': {
                'base_url': 'http://localhost:11434',
                'model': 'qwen2.5:32b'
            }
        }
        
        if VLLM_AVAILABLE:
            config['vllm'] = {
                'model_path': '/mnt/workspace/models/Qwen/QwQ-32B/',
                'gpu_memory_utilization': 0.95,
                'tensor_parallel_size': 1
            }
    
    return LocalModelManager(config)