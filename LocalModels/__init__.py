"""
本地模型支持包

支持的模型类型：
- Ollama models
- vLLM model serving
- 其他本地模型接口
"""

from .ollama_client import OllamaClient, create_ollama_client

# 尝试导入vLLM客户端
try:
    from .vllm_client import VLLMClient, create_vllm_client
    VLLM_SUPPORT = True
except ImportError:
    VLLM_SUPPORT = False
    VLLMClient = None
    create_vllm_client = None

__all__ = [
    'OllamaClient',
    'create_ollama_client',
]

# 如果vLLM可用，添加到导出列表
if VLLM_SUPPORT:
    __all__.extend([
        'VLLMClient',
        'create_vllm_client',
    ])

# 尝试导入vLLM HTTP客户端
try:
    from .vllm_http_client import VLLMHTTPClient, create_vllm_http_client
    VLLM_HTTP_SUPPORT = True
    __all__.extend([
        'VLLMHTTPClient',
        'create_vllm_http_client',
    ])
except ImportError:
    VLLM_HTTP_SUPPORT = False
    VLLMHTTPClient = None
    create_vllm_http_client = None

__version__ = '0.1.0'