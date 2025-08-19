import requests
import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama本地模型客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", 
                 model_name: str = "llama2", 
                 timeout: int = 300):
        """
        初始化Ollama客户端
        
        Args:
            base_url: Ollama服务地址
            model_name: 模型名称
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.session = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def check_connection(self) -> bool:
        """检查Ollama服务连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            else:
                logger.error(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """拉取模型"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=1800  # 30分钟超时
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', '')
                else:
                    logger.error(f"Generation failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    yield data['response']
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"Stream generation failed: {response.status}")
        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        聊天模式
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            **kwargs: 其他参数
            
        Returns:
            模型回复
        """
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('message', {}).get('content', '')
                else:
                    logger.error(f"Chat failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return None
    
    async def embeddings(self, text: str) -> Optional[List[float]]:
        """
        获取文本嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/embeddings", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('embedding', [])
                else:
                    logger.error(f"Embeddings failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return None
    
    def get_model_info(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称，默认使用当前模型
            
        Returns:
            模型信息
        """
        model = model_name or self.model_name
        
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get model info: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None


def create_ollama_client(base_url: str = "http://localhost:11434",
                        model_name: str = "llama2",
                        timeout: int = 300) -> OllamaClient:
    """
    创建Ollama客户端的便捷函数
    
    Args:
        base_url: Ollama服务地址
        model_name: 模型名称
        timeout: 超时时间
        
    Returns:
        OllamaClient实例
    """
    client = OllamaClient(base_url, model_name, timeout)
    
    # 检查连接
    if not client.check_connection():
        logger.warning(f"Cannot connect to Ollama at {base_url}")
    
    # 检查模型是否存在
    available_models = client.get_available_models()
    if available_models and model_name not in available_models:
        logger.warning(f"Model {model_name} not found. Available models: {available_models}")
        logger.info(f"Attempting to pull model {model_name}...")
        if client.pull_model(model_name):
            logger.info(f"Successfully pulled model {model_name}")
        else:
            logger.error(f"Failed to pull model {model_name}")
    
    return client


async def test_ollama_client():
    """测试Ollama客户端"""
    async with create_ollama_client() as client:
        # 测试生成
        response = await client.generate("Hello, how are you?")
        print(f"Response: {response}")
        
        # 测试聊天
        messages = [{"role": "user", "content": "What is AI?"}]
        chat_response = await client.chat(messages)
        print(f"Chat response: {chat_response}")


if __name__ == "__main__":
    asyncio.run(test_ollama_client())