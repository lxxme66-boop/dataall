# -*- coding: utf-8 -*-
"""
半导体显示技术领域智能QA生成系统 - 核心业务逻辑模块
包含完整的三步评估流程：文本质量评估、问题生成、问题质量评估
"""
import os
import re
import json
import asyncio
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入vLLM，如果失败则使用模拟类
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not available, using mock implementation")
    VLLM_AVAILABLE = False
    
    class LLM:
        """Mock LLM class for when vLLM is not available"""
        def __init__(self, **kwargs):
            self.model_path = kwargs.get('model')
            logger.warning(f"Mock LLM initialized with model: {self.model_path}")
            
        def generate(self, prompts, sampling_params, use_tqdm=True):
            """Mock generate method"""
            results = []
            for prompt in prompts:
                # 模拟输出
                mock_output = type('Output', (), {
                    'outputs': [type('GeneratedOutput', (), {
                        'text': '【是】' if '评分' in prompt else '[[1]] 示例问题1\n[[2]] 示例问题2\n[[3]] 示例问题3'
                    })()]
                })()
                results.append(mock_output)
            return results
    
    class SamplingParams:
        """Mock SamplingParams class"""
        def __init__(self, **kwargs):
            self.temperature = kwargs.get('temperature', 0.6)
            self.repetition_penalty = kwargs.get('repetition_penalty', 1.1)

# 尝试导入LocalModelManager用于vLLM HTTP模式
try:
    from LocalModels.local_model_manager import LocalModelManager
    LOCAL_MODEL_MANAGER_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_MANAGER_AVAILABLE = False
    logger.warning("LocalModelManager not available, vLLM HTTP mode will not work")


@dataclass
class ModelConfig:
    """模型配置类"""
    name: str
    path: str
    stop_tokens: List[Any]
    max_model_len: int = 96 * 1024
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int = 2
    temperature: float = 0.6
    repetition_penalty: float = 1.1
    min_p: float = 0
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 4096


# 预定义的模型配置
MODEL_CONFIGS = {
    "qwq_32": ModelConfig(
        name="qwq_32",
        path="/mnt/data/LLM/lhy/models/Qwen3/Qwen3-235B-A22B",
        stop_tokens=[151329, 151336, 151338]
    ),
    "qw2_72": ModelConfig(
        name="qw2_72",
        path="/data/lc/openmodels/qw2_72b_instruct",
        stop_tokens=["<|im_end|>"]
    ),
    "qw2_72_awq": ModelConfig(
        name="qw2_72_awq",
        path="/data/lc/openmodels/qw2_72b_instruct_awq",
        stop_tokens=["<|im_end|>"]
    ),
    "qw2.5_32": ModelConfig(
        name="qw2.5_32",
        path="/data/lc/openmodels/qw2.5_32b_instruct",
        stop_tokens=["<|im_end|>"]
    ),
    "qw2.5_72": ModelConfig(
        name="qw2.5_72",
        path="/data/lc/openmodels/qw2.5_72b_instruct",
        stop_tokens=["<|im_end|>"]
    ),
    "llama3.1_70": ModelConfig(
        name="llama3.1_70",
        path="/data/lc/openmodels/llama3.1_70b_instruct",
        stop_tokens=["<|eot_id|>"]
    )
}


class SemiconductorQAGenerator:
    def __init__(self, batch_size=4, gpu_devices=None):
        """初始化QA生成器 - 修复model_name参数问题"""
        self.batch_size = batch_size
        self.gpu_devices = gpu_devices or "0"
        self.model_name = "qwq_32"  # 默认模型名称
        
        # 检查是否使用vLLM HTTP模式
        self.use_vllm_http = os.environ.get('USE_VLLM_HTTP', 'false').lower() == 'true'
        self.vllm_server_url = os.environ.get('VLLM_SERVER_URL', 'http://localhost:8000/v1')
        logger.info(f"DEBUG: USE_VLLM_HTTP={os.environ.get('USE_VLLM_HTTP')}, self.use_vllm_http={self.use_vllm_http}")
        
        # 加载模型配置
        self.config = MODEL_CONFIGS.get(self.model_name)
        if not self.config:
            logger.warning(f"未找到模型 {self.model_name} 的配置，使用默认配置")
            self.config = ModelConfig(
                name=self.model_name,
                path=self.model_name,
                stop_tokens=[],
                max_model_len=96 * 1024
            )
        self.stats = {
            "generated_questions": 0,
            "total_questions": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }
        # 原有的初始化代码保持不变
        self.tokenizer = None
        self.model = None
        self.device_map = self._setup_device_map()
        
        # 加载模板
        self._load_templates()
        
        # 初始化vLLM相关
        self.llm = None
        self.sampling_params = None
        self.local_model_manager = None
        
        # 自动初始化LLM
        self.setup_llm_and_tokenizer()
        
        logger.info(f"初始化QA生成器完成: batch_size={batch_size}, gpu_devices={gpu_devices}, use_vllm_http={self.use_vllm_http}")
    
    def setup_llm_and_tokenizer(self):
        """初始化vLLM和tokenizer"""
        logger.info(f"DEBUG setup_llm_and_tokenizer: self.use_vllm_http={self.use_vllm_http}, USE_VLLM_HTTP={os.environ.get('USE_VLLM_HTTP')}")
            # 初始化 tokenizer
        if self.use_vllm_http:
            print("使用vLLM HTTP模式，跳过本地tokenizer初始化")
            
            # 创建一个更完整的假tokenizer对象
            class MockTokenizer:
                def __init__(self):
                    self.vocab_size = 151936  # Qwen的词汇表大小
                    self.pad_token_id = 151643
                    self.eos_token_id = 151645
                    
                def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, truncation=True):
                    # 手动格式化
                    formatted = ""
                    for msg in messages:
                        role = msg['role']
                        content = msg['content']
                        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    if add_generation_prompt:
                        formatted += "<|im_start|>assistant\n"
                    return formatted
                
                def encode(self, text, add_special_tokens=True, max_length=None, truncation=True):
                    # 简单的长度估算：中文字符约1个token，英文单词约0.75个token
                    estimated_length = len(text.replace(' ', '')) * 1.2
                    return list(range(int(estimated_length)))  # 返回假的token列表
                
                def decode(self, token_ids, skip_special_tokens=True):
                    # 简单返回空字符串或占位符
                    return f"decoded_text_{len(token_ids)}_tokens"
                
                def __len__(self):
                    return self.vocab_size
            
            self.tokenizer = MockTokenizer()
            #return
        else:
            # 只有非HTTP模式才初始化本地tokenizer
            try:
                model_path = self.config.get('models', {}).get('qa_generator_model', {}).get('path')
                if model_path and os.path.exists(model_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                else:
                    self.tokenizer = None
            except Exception as e:
                print(f"初始化tokenizer失败: {e}")
                self.tokenizer = None
            
        logger.info(f"初始化LLM模型: use_vllm_http={self.use_vllm_http}")
        
        try:
            # 如果使用vLLM HTTP模式
            if self.use_vllm_http and LOCAL_MODEL_MANAGER_AVAILABLE:
                logger.info("使用vLLM HTTP模式")
                # 创建配置
                config = {
                    'api': {
                        'use_local_models': True,
                        'use_vllm_http': True,
                        'vllm_server_url': self.vllm_server_url,
                        'default_backend': 'vllm_http'
                    },
                    'models': {
                        'local_models': {
                            'default_backend': 'vllm_http',
                            'vllm_http': {
                                'base_url': self.vllm_server_url,
                                'api_key': 'dummy-key',
                                'model_name': 'qwen-vllm',
                                'temperature': 0.6,
                                'repetition_penalty': 1.1,
                                'max_tokens': 8192,
                                'stop_token_ids': self.config.stop_tokens if self.config.stop_tokens and isinstance(self.config.stop_tokens[0], int) else None
                            }
                        }
                    }
                }
                
                # 初始化LocalModelManager
                self.local_model_manager = LocalModelManager(config)
                
                # 设置为HTTP模式的标志
                self.llm = "vllm_http"  # 使用字符串标记
                
                # 初始化tokenizer（如果有本地模型路径）
                if os.path.exists(self.config.path):
                    self.tokenizer = AutoTokenizer.from_pretrained("/mnt/data/LLM/lhy/models/Qwen3/Qwen3-235B-A22B", trust_remote_code=True)
                else:
                    # 使用默认的tokenizer
                    logger.warning(f"模型路径 {self.config.path} 不存在，尝试使用Qwen tokenizer")
                    try:
                        # 尝试使用Qwen的tokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
                    except:
                        logger.warning("无法加载Qwen tokenizer，使用模拟tokenizer")
                        self.tokenizer = None
                
                # 设置sampling参数（虽然在HTTP模式下可能不直接使用）
                # 处理stop_tokens，确保它们是字符串列表
                stop_tokens = None
                if self.config.stop_tokens:
                    if isinstance(self.config.stop_tokens, list):
                        # 过滤掉整数类型的token，只保留字符串
                        stop_tokens = [str(token) for token in self.config.stop_tokens if isinstance(token, str)]
                        if not stop_tokens:  # 如果过滤后为空，设为None
                            stop_tokens = None
                
                self.sampling_params = SamplingParams(
                    temperature=0.6,
                    repetition_penalty=1.1,
                    max_tokens=8192,
                    stop=stop_tokens
                )
                
                logger.info("vLLM HTTP模式初始化成功")
                
            else:
                # 原有的vLLM初始化逻辑
                logger.info(f"初始化vLLM模型: {self.config.path}")
                
                # 设置环境变量
                os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
                
                # 初始化vLLM
                if VLLM_AVAILABLE:
                    self.llm = LLM(
                        model=self.config.path,
                        trust_remote_code=True,
                        gpu_memory_utilization=0.95,
                        tensor_parallel_size=len(self.gpu_devices.split(',')) if ',' in self.gpu_devices else 1,
                        max_model_len=self.config.max_model_len
                    )
                    
                    # 初始化tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.path, trust_remote_code=True)
                    
                    # 设置sampling参数
                    # 处理stop_tokens，确保它们是字符串列表
                    stop_tokens = None
                    if self.config.stop_tokens:
                        if isinstance(self.config.stop_tokens, list):
                            # 过滤掉整数类型的token，只保留字符串
                            stop_tokens = [str(token) for token in self.config.stop_tokens if isinstance(token, str)]
                            if not stop_tokens:  # 如果过滤后为空，设为None
                                stop_tokens = None
                    
                    self.sampling_params = SamplingParams(
                        temperature=0.6,
                        repetition_penalty=1.1,
                        max_tokens=8192,
                        stop=stop_tokens
                    )
                    
                    logger.info("vLLM和tokenizer初始化成功")
                else:
                    logger.warning("vLLM不可用，使用模拟实现")
                    self.llm = LLM(model=self.config.path)
                    self.tokenizer = None  # Mock模式下不需要真实tokenizer
                    self.sampling_params = SamplingParams(
                        temperature=0.6,
                        repetition_penalty=1.1,
                        max_tokens=8192
                    )
                
        except Exception as e:
            logger.error(f"初始化LLM失败: {str(e)}")
            import traceback
            logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise RuntimeError(f"无法初始化LLM模型") from e
    
    def _format_messages(self, messages, add_generation_prompt=True, truncation=True):
        """格式化消息，处理tokenizer可能为None的情况"""
        if self.tokenizer is not None:
            # return self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=add_generation_prompt,
            #     truncation=truncation
            # )
            prompt = self._format_messages(messages, add_generation_prompt=True)
        else:
            # Mock模式或tokenizer未初始化时，直接构建提示
            formatted = ""
            for msg in messages:
                role = msg['role'].capitalize()
                content = msg['content']
                formatted += f"{role}: {content}\n\n"
            if add_generation_prompt:
                formatted += "Assistant:"
            return formatted
    # def setup_llm_and_tokenizer(self):
    #     """带调试信息的LLM和tokenizer初始化方法"""
    #     try:
    #         import json
    #         import os
    #         import traceback
            
    #         print("=== 开始初始化LLM ===")
            
    #         # 确定模型路径
    #         if hasattr(self, 'model_configs') and hasattr(self, 'model_name') and self.model_name in self.model_configs:
    #             model_path = self.model_configs[self.model_name]["model_path"]
    #             print(f"使用配置中的模型路径: {model_path}")
    #         else:
    #             model_path = "/mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain"
    #             print(f"使用默认模型路径: {model_path}")
            
    #         # 检查路径是否存在
    #         if not os.path.exists(model_path):
    #             raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
    #         # 检查配置文件
    #         config_path = os.path.join(model_path, "config.json")
    #         print(f"检查配置文件: {config_path}")
            
    #         max_model_len = 32768  # 安全的默认值
            
    #         if os.path.exists(config_path):
    #             try:
    #                 with open(config_path, 'r', encoding='utf-8') as f:
    #                     model_config = json.load(f)
    #                     print(f"成功读取模型配置")
                        
    #                     max_position_embeddings = model_config.get('max_position_embeddings')
    #                     model_max_length = model_config.get('model_max_length')
                        
    #                     print(f"max_position_embeddings: {max_position_embeddings}")
    #                     print(f"model_max_length: {model_max_length}")
                        
    #                     if max_position_embeddings:
    #                         max_model_len = min(max_model_len, max_position_embeddings)
    #                     if model_max_length:
    #                         max_model_len = min(max_model_len, model_max_length)
                            
    #             except Exception as e:
    #                 print(f"读取配置文件失败: {e}")
    #         else:
    #             print(f"配置文件不存在，使用默认值")
            
    #         print(f"最终使用的max_model_len: {max_model_len}")
            
    #         # 检查设备映射
    #         if hasattr(self, 'device_map'):
    #             tensor_parallel_size = 2
    #             print(f"tensor_parallel_size: {tensor_parallel_size}")
    #         else:
    #             tensor_parallel_size = 2  # 默认值
    #             print(f"使用默认tensor_parallel_size: {tensor_parallel_size}")
            
    #         print("开始创建LLM实例...")
            
    #         # 创建LLM实例
    #         self.llm = LLM(
    #             model=model_path,
    #             tensor_parallel_size=tensor_parallel_size,
    #             gpu_memory_utilization=0.9,
    #             trust_remote_code=True,
    #             dtype="auto",
    #             enforce_eager=False,
    #             max_model_len=max_model_len,
    #             tokenizer_mode="auto",
    #             disable_custom_all_reduce=False,
    #             quantization=None,
    #         )
            
    #         print("LLM创建成功，开始初始化tokenizer...")
            
    #         # 初始化tokenizer
    #         self.tokenizer = AutoTokenizer.from_pretrained(
    #             model_path,
    #             trust_remote_code=True,
    #             padding_side="left"
    #         )
            
    #         if self.tokenizer.pad_token is None:
    #             self.tokenizer.pad_token = self.tokenizer.eos_token
            
    #         print("=== LLM和tokenizer初始化完成 ===")
            
    #     except Exception as e:
    #         print(f"初始化失败: {e}")
    #         print(f"堆栈跟踪:\n{traceback.format_exc()}")
    #         raise RuntimeError(f"无法初始化LLM模型") from e
    def _format_messages(self, messages, add_generation_prompt=True, truncation=True):
        """格式化消息，处理tokenizer可能为None的情况"""
        # 直接使用Qwen格式，不依赖tokenizer
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    async def _generate_async(self, prompts, sampling_params=None, use_tqdm=False):
        """异步生成方法，支持vLLM HTTP模式"""
        logger.debug(f"DEBUG _generate_async: self.llm={self.llm}, self.local_model_manager={self.local_model_manager}, self.use_vllm_http={self.use_vllm_http}")
        if self.llm == "vllm_http" and self.local_model_manager:
            # vLLM HTTP模式
            results = []
            for prompt in prompts:
                try:
                    # 使用LocalModelManager生成
                    response = await self.local_model_manager.generate(prompt)
                    # 创建与vLLM兼容的输出格式
                    output = type('Output', (), {
                        'outputs': [type('GeneratedOutput', (), {
                            'text': response
                        })()]
                    })()
                    results.append(output)
                except Exception as e:
                    logger.error(f"vLLM HTTP生成失败: {e}")
                    # 创建错误输出
                    output = type('Output', (), {
                        'outputs': [type('GeneratedOutput', (), {
                            'text': '【否】生成过程中出现错误'
                        })()]
                    })()
                    results.append(output)
            return results
        elif self.llm is not None:
            # 原有的vLLM模式 - 在异步上下文中调用同步方法
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.llm.generate, prompts, sampling_params or self.sampling_params, use_tqdm)
        else:
            # self.llm 为 None，返回错误
            logger.error("LLM未初始化，无法生成")
            return [type('Output', (), {
                'outputs': [type('GeneratedOutput', (), {
                    'text': '【否】模型未初始化'
                })()]
            })() for _ in prompts]
    
    def _generate(self, prompts, sampling_params=None, use_tqdm=False):
        """统一的生成方法，支持vLLM和vLLM HTTP模式"""
        logger.debug(f"DEBUG _generate: self.llm={self.llm}, self.local_model_manager={self.local_model_manager}, self.use_vllm_http={self.use_vllm_http}")
        if self.llm == "vllm_http" and self.local_model_manager:
            # vLLM HTTP模式 - 需要使用同步包装器
            import asyncio
            
            async def async_generate_batch():
                results = []
                for prompt in prompts:
                    try:
                        # 使用LocalModelManager生成
                        response = await self.local_model_manager.generate(prompt)
                        # 创建与vLLM兼容的输出格式
                        output = type('Output', (), {
                            'outputs': [type('GeneratedOutput', (), {
                                'text': response
                            })()]
                        })()
                        results.append(output)
                    except Exception as e:
                        logger.error(f"vLLM HTTP生成失败: {e}")
                        # 创建错误输出
                        output = type('Output', (), {
                            'outputs': [type('GeneratedOutput', (), {
                                'text': '【否】生成过程中出现错误'
                            })()]
                        })()
                        results.append(output)
                return results
            
            # 在同步上下文中运行异步函数
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果已经在异步上下文中，创建任务
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, async_generate_batch())
                        return future.result()
                else:
                    # 如果不在异步上下文中，直接运行
                    return asyncio.run(async_generate_batch())
            except Exception as e:
                logger.error(f"运行异步生成时出错: {e}")
                # 返回错误结果
                return [type('Output', (), {
                    'outputs': [type('GeneratedOutput', (), {
                        'text': '【否】生成过程中出现错误'
                    })()]
                })() for _ in prompts]
        elif self.llm is not None:
            # 原有的vLLM模式
            return self.llm.generate(prompts, sampling_params or self.sampling_params, use_tqdm=use_tqdm)
        else:
            # self.llm 为 None，返回错误
            logger.error("LLM未初始化，无法生成")
            return [type('Output', (), {
                'outputs': [type('GeneratedOutput', (), {
                    'text': '【否】模型未初始化'
                })()]
            })() for _ in prompts]
    
    def _encode_length(self, text):
        """获取文本编码长度，处理tokenizer可能为None的情况"""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            # 粗略估计：每个字符算1个token
            return len(text)
    
    def set_model_name(self, model_name: str):
        """设置模型名称 - 新增方法"""
        self.model_name = model_name
        logger.info(f"设置模型名称为: {model_name}")
        
        # 更新模型配置
        self.config = MODEL_CONFIGS.get(self.model_name)
        if not self.config:
            logger.warning(f"未找到模型 {self.model_name} 的配置，使用默认配置")
            self.config = ModelConfig(
                name=self.model_name,
                path=self.model_name,
                stop_tokens=[],
                max_model_len=96 * 1024
            )
        
        # 如果模型已经加载，重新加载新模型
        if self.model:
            logger.info("模型已加载，重新加载新模型...")
            self._load_model()
            
        # 重新初始化vLLM
        if self.llm:
            logger.info("重新初始化vLLM...")
            self.llm = None
            self.tokenizer = None
            self.sampling_params = None
            self.setup_llm_and_tokenizer()
    
    def load_model(self):
        """加载模型（公有方法）"""
        if self.model is None:
            self._load_model()
        else:
            logger.info("模型已加载，跳过加载")
    
    def _load_model(self):
        """加载模型和分词器"""
        logger.info(f"加载模型: {self.model_name}")
        
        try:
            # 实际加载模型的代码
            # 这里应该是加载tokenizer和模型的代码
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16
            )
            logger.info(f"成功加载模型: {self.model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            # 添加详细错误信息
            import traceback
            logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise RuntimeError(f"无法加载模型 {self.model_name}") from e
    
    def _setup_device_map(self):
        """设置GPU设备映射"""
        if not torch.cuda.is_available():
            logger.warning("未检测到GPU，将使用CPU")
            return {"": "cpu"}
        
        # 处理可能的空值或无效值
        if not self.gpu_devices or not isinstance(self.gpu_devices, str):
            logger.warning(f"无效的GPU设备配置: {self.gpu_devices}，使用默认GPU 0")
            return {"": "cuda:0"}
        
        try:
            device_ids = [int(id) for id in self.gpu_devices.split(",") if id.isdigit()]
            if not device_ids:
                logger.warning("未指定有效的GPU设备，使用默认GPU 0")
                return {"": "cuda:0"}
            
            # 创建设备映射
            device_map = {f"cuda:{i}": f"cuda:{id}" for i, id in enumerate(device_ids)}
            logger.info(f"创建设备映射: {device_map}")
            return device_map
        except Exception as e:
            logger.error(f"创建设备映射失败: {str(e)}，使用默认GPU 0")
            return {"": "cuda:0"}
    
    # 其他方法保持不变...
    
    def _load_templates(self):
        """加载评估模板"""
        # 文本质量评分模板
        self.score_template = """你的任务是依据以下评分规则对文本质量进行打分，并输出最终得分。评分流程如下：
1.依照每个标准依次评估文本。对每个子问题如实作答。若对某子问题答案为明确 "是"，则按标准相应加分或减分;
2.记录每个标准的累计分数，得出总分;
3.依据以下说明，将最终评估结果整理为有效的 JSON 对象。

## 评分标准：
1.标准 1：问题完整性
(1) 内容无清晰主要问题，或缺乏足够线索得出正确答案，得 0 分。
(2) 内容包含一个主要问题，且有足够线索得出正确答案，得 + 1 分。
(3) 文本体现多位作者间互动与讨论，如提出答案、评估反思答案、回应批评、修订编辑答案，得 + 1 分。
2.标准 2：问题复杂性和技术深度
(1) 内容难度为大学水平或以下，得 0 分。
(2) 内容难度为研究生水平或以上，仅领域专家能理解，得 + 1 分。
(3) 所讨论问题极具挑战性，高技能非专家花费 30 分钟上网搜索或阅读文献后，仍无法完全理解问题或给出正确答案，得 + 1 分。
3.标准 3：技术正确性和准确性
(1) 文本含显著技术错误或不准确，得 -1 分。
(2) 文本有一定技术正确性，但存在明显缺陷或遗漏（如单位错误、推导不完整），得 0 分。
(3) 文本技术正确，但有小缺陷或遗漏（如小代数错误、解释不完整），得 + 0.5 分。
(4) 文本技术高度正确，解释清晰准确（如精确定义、完整推导），得 + 0.5 分。
(5) 文本技术卓越正确，解释严格精确（如形式化证明、精确计算），得 + 1 分。
4.标准 4：思维和推理
(1) 文本无任何思维或推理迹象，得 -1 分。
(2) 文本展现一些基本思维和推理能力（如直接应用已知技术、简单分析问题），得 + 0.5 分。
(3) 文本展现一定思维和推理能力（如考虑多种解决方法、讨论不同方案权衡），得 + 0.5 分。
(4) 文本展现显著思维和推理能力（如通过多步推理链解决复杂问题、运用专业科学领域高级推理模式），得 + 1 分。
(5) 文本展现卓越思维和推理能力（如以高度创新方式解决专业领域复杂问题、结合多种推理技术对问题进行新抽象），得 + 1 分。

最终评判标准：若各项标准得分均大于零，且标准 4 得分大于等于 1 分，则该文本内容适合生成逻辑推理问题。

[文本内容的开始]
{academic_paper}
[文本内容的结束]

格式要求：只输出文本内容是否适合生成复杂推理问题，不输出任何别的内容。并且是否适合严格按照以下格式进行输出：
【是】或者【否】。不要输出为空，不要输出其他内容，输出是或否时，要带上【】符号进行输出。
"""

        # 问题生成模板 - 使用用户提供的专业版本
        self.prompt_template = """你是一位半导体显示技术领域的资深专家，擅长从技术文献中提炼核心知识点。你的职责是从下面论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容，生成3个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：

[学术论文的开始]
{academic_paper}
[学术论文的结束]

<think>
首先，我需要仔细阅读这篇学术论文，理解其核心内容和技术要点。

让我分析论文的主要内容：
1. 识别论文讨论的核心技术问题
2. 找出涉及逻辑推理的技术原理和机制
3. 确定可以设计深度问题的关键点

接下来，我将基于以下原则设计问题：
- 问题必须基于论文内容，答案可以从论文中找到
- 问题需要逻辑推理才能解答，不能是简单的事实性问题
- 问题描述要清晰、完整、专业、具有通用性
- 避免使用"本文"、"论文"等自指表述
- 确保问题具有技术深度和挑战性
</think>

##【核心要求】
### 问题设计准则：
(1) 仔细通读全文，找出涉及逻辑推理的文本部分，据此设计相关问题。
(2) 问题要基于论文里的技术原理，描述务必清晰、全面、明确。问题中主语或名词的表述要精准、全面且具有通用性。
(3) 避免在问题中引用文献或文章自定义的专有名词。利用自身半导体显示领域知识，生成通用问题，确保不读论文也能理解问题含义。
(4) 问题中的名词描述必须和论文一致，不能缩写。比如论文是 "OLED 材料"，问题不能写成 "材料"；论文是 "LTPS 器件"，问题不能写成 "器件"。
(5) 提问不能针对论文里的某个特定示例。要让顶尖科学家不读论文也能理解并回答问题。问题里不能有 "书本""论文""本文""本实验" 等类似表述。
(6) 确保生成问题完整，与论文彻底解耦，不依赖论文内容。若问题有背景信息，必须阐述清楚。
(7) 问题简洁：生成的问题要凝练、简洁。
(8) 非常关键：生成的问题能基于上传的论文回答
    错误示例：原文阐述一个现象A,大模型生成的问题为什么会出现A？，分析：原文没有提到答案
(9) 只生成一个完整的问题（需要补充要求则整合成一个），不要一个问题有几个小问题这种，
    例如：1.在背沟道刻蚀型IGZO TFT中，采用单接触孔结构与传统双接触孔刻蚀阻挡层结构相比，如何影响器件的寄生电容和工艺复杂度？从材料特性和工艺集成角度分析其性能差异的技术机理。?"
        分析：这一个问题看似有2个小问题，其实是一个问题，这种则需要整合成一个即可。
        2."在柔性电子纸显示器驱动应用中，金属氧化物薄膜晶体管（MOTFT）相较于有机薄膜晶体管（OTFT）和非晶硅薄膜晶体管（a-Si TFT），其迁移率优势（10-100 cm²·V⁻¹·s⁻¹）如何通过载流子传输机制差异解释？并分析该优势对电子纸双稳态特性和像素响应时间的具体影响。?"
        分析：这是两个问题，只需要生成一个问题（单一性）。

### 问题设计的科学严谨性：
(1) 因果链：问题需呈现完整技术逻辑链，比如 "机制 A 怎样影响参数 B，进而引发现象 C"。
(2) 周密性：思考过程要科学严谨，逐步推进。保证问题及对应答案源自论文内容，且答案在论文中有详细阐述。
(3) 可追溯性：问题基于原文生成，在原文中有详细的描述。其对应的答案也必须能基于原文回答，答案在原文中有详细的描述。
(4) 单一性：每个问题只包含一个问题，要不包含一个以上的问题。（严格执行！）
例如：
    正确（只包含一个问题）：在柔性电子纸显示器驱动电路中，金属氧化物薄膜晶体管（MOTFT）的载流子迁移率典型范围是多少？
    错误（一个问题包含有两个问题）："在柔性电子纸显示器驱动应用中，金属氧化物薄膜晶体管（MOTFT）相较于有机薄膜晶体管（OTFT）和非晶硅薄膜晶体管（a-Si TFT），其迁移率优势（10-100 cm²·V⁻¹·s⁻¹）如何通过载流子传输机制差异解释？并分析该优势对电子纸双稳态特性和像素响应时间的具体影响。?"
    
## 【禁止事项】
× 禁止使用"本文/本研究/本实验"等论文自指表述
× 禁止提问孤立概念（如：XX技术的定义是什么）
× 禁止超出论文技术范围的假设性问题
×  禁止一个问题中有多个问题。保证每个问题单一。例如：在柔性电子纸显示器驱动电路中，金属氧化物薄膜晶体管（MOTFT）的载流子迁移率典型范围是多少？

##【格式要求】
用中文输出。当前阶段只设计问题，不输出答案。输出问题前必须用 </think> 结束思考后在输出问题。严格按照以下格式输出你设计的问题：
[[1]] 第1个问题
[[2]] 第2个问题
[[3]] 第3个问题
[[4]] 第4个问题
[[5]] 第5个问题
[[6]] 第6个问题
...



"""

        # 问题质量评估模板
        self.evaluator_template = """您是一位专家评估员，负责决定问题是否符合推理问题标准。您的评估必须结合给定文章内容和给定问题判断。

## 【评估标准】
### 因果性：
(1) 问题应展现出完整的技术逻辑链。比如，类似 "机制 A 怎样影响参数 B，最终致使现象 C 出现" 这种形式。

### 周密性：
(1) 思维过程要科学且严谨，需逐步思考。

### 追溯性：
(1) 问题必须源于论文内容。
(2) 针对问题的答案也在论文中要有详细描述。

### 通用性：
(1) 问题不是太绝对。问题具有通用性，不只是局限于文本中。

### 完整性：
(1) 问题是否全面涵盖文章相关内容的各个方面？
(2) 问题描述应简洁凝练，语义完整。
(3) 问题要与文章内容完全独立，不依赖文章也能被清晰理解，即问题需完整、自足。
(4) 问题能基于文章内容回答

###单一性
(1) 每个问题只包含一个问题，要不包含一个以上的问题。
例如：
    正确（只包含一个问题）：在柔性电子纸显示器驱动电路中，金属氧化物薄膜晶体管（MOTFT）的载流子迁移率典型范围是多少？
    错误（一个问题包含有两个问题）："在柔性电子纸显示器驱动应用中，金属氧化物薄膜晶体管（MOTFT）相较于有机薄膜晶体管（OTFT）和非晶硅薄膜晶体管（a-Si TFT），其迁移率优势（10-100 cm²·V⁻¹·s⁻¹）如何通过载流子传输机制差异解释？并分析该优势对电子纸双稳态特性和像素响应时间的具体影响。?"
    
[文章内容的开始]
{academic_paper}
[文章内容的结束]

[问题内容]
{academic_question}

格式要求：仅输出文本内容生成的问题是否符合标准，严格按以下格式，有且仅输出【是】或者【否】，不输出任何别的内容，不能输出为空，输出是或否时，要带上【】符号进行输出。用中文输出，严格按照以下格式进行输出：【是】或者【否】
"""

    def is_to_drop(self, text: str) -> bool:
        """
        判断文本是否需要被过滤掉
        
        Args:
            text: 待检查的文本
            
        Returns:
            bool: True表示需要过滤，False表示保留
        """
        text = text.strip()[:10]    
        patterns = ["", "#"]
        for pattern in patterns:
            if text == pattern:
                return True 
                
        patterns = [
            r'http://www.cnki.net', r'https://www.cnki.net', r'^\[\d{1,4}\]', 
            r'^\*\s+\[\d{1,4}\]', r'^\*\s+\(\d{1,4}\)', 
            r'^致谢.*[0-9]$', r'.*致\s*谢.*', r'.*目\s*录.*', 
            r'\.\.\.\.\.\.\.\.', r'\…\…\…', r"(http://www|doi:|DOI:|please contact)",
            r"(work was supported by|study was supported by|China|Republic of Korea|Authorized licensed use limited to)",
            r"\s[1-9]\d{5}(?!\d)",  # 邮编
            r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*", 
            r"(received in revised form|All rights reserved|©)", r"[a-zA-z]+://[^\s]*",
            r"(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}", 
            r"\d{3}-\d{8}|\d{4}-\d{7}",
            r'^分\s*类\s*号', r'^学\s*科\s*专\s*业', r'^签\s*字\s*日\s*期', 
            r'^申\s*请\s*人\s*员\s*姓\s*名',
            r'^日\s*期', r'^指\s*定\s*教\s*师', r'学\s*位\s*论\s*文', 
            r'^工\s*作\s*单\s*位', r'^电\s*话', r'^通讯地址', r'^邮\s*编',
            r'^\[?\d+\]?', r'^\s*\[?\d+\]?', r'^\［?\d+\］?', r'^\s*\［?\d+\］?'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        keyword_patterns = [
            '申请号|专利号|已录用|学报|研究生|已收录|攻读|第一作者|第二作者|参考文献|专业名称|863项目|导师',
            '教授|感谢|致谢|谢谢|指导|朋友|家人|亲友|师弟|师妹|老师|同学|父母|充实|答辩|祝愿|独创性声明|作者签名',
            '发表文章|论文使用授权声明|本人|知网|论文使用权|发表的论文|申请的专利|申请专利|发表的文章|发表学术论文|发表论文',
            '参与科研项目|作者简介|三年的学习|大学硕士学位论文|大学博士学位论文|涉密论文|学校代码|论文提交日期|委员：|中图分类号',
            '原创性声明|顺利完成学业|All rights reserved|参 考 文 献|参考文献|所在学院|国家自然科学基金|教育部重点学科建设',
            '时间飞梭|时光飞梭|光阴似箭|白驹过隙|论文版权|本学位论文|使用授权书|References|Acknowledgements',
            '论文著作权|保密的学位论文|中国第一所现代大学|参加科研情况|独 创 性 声 明|论文使用授权|获得的专利|家庭的爱|文献标识码|文章编号'
        ]
        
        for pattern in keyword_patterns:
            if re.findall(pattern, text):
                return True   
        
        # 判断是否不包含中文字符
        num = 0
        for t in text:
            if '\u4e00' <= t <= '\u9fa5':
                num += 1    
        if len(text) > 0 and num / len(text) < 0.01:
            return True
                    
        return False

    def drop(self, texts: str, concatenation: str = "\n") -> str:
        """
        过滤文本中的无关内容
        
        Args:
            texts: 输入文本
            concatenation: 连接符
            
        Returns:
            str: 过滤后的文本
        """
        new_texts = []
        texts = texts.split("\n")
        for i, text in enumerate(texts):
            if not self.is_to_drop(text):
                new_texts.append(text)
        return concatenation.join(new_texts)

    def load_paper(self, file_path: str) -> str:
        """
        加载论文并进行过滤处理
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 过滤后的论文内容
        """
        with open(file_path, "r", encoding="utf8") as f:
            content = f.read()
        deal_content = self.drop(content)
        return deal_content

    def to_batch(self, lst: List[Any], groupsize: int) -> List[List[Any]]:
        """
        将列表分批处理
        
        Args:
            lst: 输入列表
            groupsize: 每批大小
            
        Returns:
            list: 分批后的列表
        """
        return [lst[i:i+groupsize] for i in range(0, len(lst), groupsize)]

    async def judge_md_data(self, raw_folders: List[str], save_paths: List[str], 
                           jsonl_file_path: str) -> Dict[str, Any]:
        """
        评估文档数据质量，判断是否适合生成逻辑推理问题
        
        Args:
            raw_folders: 原始文件夹列表
            save_paths: 保存路径列表
            jsonl_file_path: JSONL输出文件路径
            
        Returns:
            dict: 评估结果统计
        """
        start_time = time.time()
        total_stats = {
            "total_files": 0,
            "passed_files": 0,
            "failed_files": 0,
            "error_files": 0
        }
        
        for raw_folder, save_path in zip(raw_folders, save_paths):
            logger.info(f"Processing folder: {raw_folder}")
            
            # 获取所有txt文件
            files = []
            if os.path.exists(raw_folder):
                files = [f for f in os.listdir(raw_folder) if f.endswith('.txt')]
            else:
                logger.error(f"Folder not found: {raw_folder}")
                continue
                
            files.sort()
            total_stats["total_files"] += len(files)
            
            # 检查已处理的文件
            already_ids = []
            results = []
            if os.path.exists(save_path):
                try:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    already_ids = [r["id"] for r in results if "id" in r]
                except Exception as e:
                    logger.error(f"Error loading existing results: {e}")
            
            logger.info(f"Already processed: {len(already_ids)} files")
            
            # 待处理文件
            to_do = [f for f in files if f not in already_ids]
            if not to_do:
                logger.info("No new files to process")
                continue
                
            # 批处理
            batches = self.to_batch(to_do, self.batch_size)
            comply_stand_datas = []
            
            for batch_idx, batch in enumerate(tqdm(batches, desc=f"Judging {raw_folder}")):
                score_inputs = []
                batch_metadata = []
                
                # 准备批处理输入
                for paper in batch:
                    try:
                        paper_name = paper.split("_part")[0]
                        paper_content = self.load_paper(os.path.join(raw_folder, paper))
                        
                        if len(paper_content.strip()) < 100:
                            logger.warning(f"Paper {paper} too short, skipping")
                            continue
                        
                        # 构建评分提示
                        score_prompt = self.score_template.replace("{academic_paper}", paper_content)
                        score_messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": score_prompt}
                        ]
                        
                        # 应用聊天模板
                        data_li = self._format_messages(score_messages)
                        
                        # 检查长度并截断
                        if self._encode_length(data_li) > self.config.max_model_len - 1024:
                            logger.warning(f"Prompt too long for {paper}, truncating")
                            # 截断内容
                            max_content_len = self.config.max_model_len - 2048
                            truncated_content = paper_content[:max_content_len]
                            score_prompt = self.score_template.replace("{academic_paper}", truncated_content)
                            score_messages = [
                                {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                                {"role": "user", "content": score_prompt}
                            ]
                            data_li = self._format_messages(score_messages)
                        
                        score_inputs.append(data_li)
                        batch_metadata.append({
                            "paper": paper,
                            "paper_name": paper_name,
                            "paper_content": paper_content
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {paper}: {e}")
                        total_stats["error_files"] += 1
                        continue
                
                if not score_inputs:
                    continue
                
                # 批量推理
                try:
                    score_outputs = self._generate(score_inputs, use_tqdm=False)
                    
                    # 处理输出
                    for idx, output in enumerate(score_outputs):
                        metadata = batch_metadata[idx]
                        score_text = output.outputs[0].text.strip()
                        
                        # 提取最后一行作为判断结果
                        lines = score_text.split('\n')
                        final_judgment = lines[-1] if lines else ""
                        
                        # 判断是否通过
                        is_suitable = '【是】' in final_judgment
                        
                        result = {
                            "stats": 1 if is_suitable else 0,
                            "paper_name": metadata["paper_name"],
                            "paper_content": metadata["paper_content"],
                            "score_text": score_text,
                            "judgment": final_judgment,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        comply_stand_datas.append(result)
                        
                        if is_suitable:
                            total_stats["passed_files"] += 1
                        else:
                            total_stats["failed_files"] += 1
                            
                except Exception as e:
                    logger.error(f"Error in batch inference: {e}")
                    total_stats["error_files"] += len(batch_metadata)
                
                # 定期保存结果
                if (batch_idx + 1) % self.save_steps == 0:
                    self._save_intermediate_results(comply_stand_datas, jsonl_file_path)
            
            # 保存最终结果
            if comply_stand_datas:
                self._save_intermediate_results(comply_stand_datas, jsonl_file_path)
                
                # 更新总结果文件
                all_results = results + [
                    {"id": d["paper_name"], "paper_content": d["paper_content"], 
                     "stats": d["stats"], "score_text": d["score_text"]}
                    for d in comply_stand_datas if d["stats"] == 1
                ]
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                
                logger.info(f"Results saved to {save_path}")
        
        # 统计处理时间
        total_stats["processing_time"] = time.time() - start_time
        self.stats.update(total_stats)
        
        logger.info(f"Text quality assessment completed: {total_stats}")
        return total_stats

    async def judge_processed_texts(self, processed_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估已处理的文本数据质量，判断是否适合生成逻辑推理问题
        
        Args:
            processed_texts: 已处理的文本列表，每个元素包含:
                - paper_name: 文档名称
                - md_content: Markdown格式的内容
                - source_info: 原始信息
                
        Returns:
            list: 评估结果列表，每个元素包含原始数据和判断结果
        """
        start_time = time.time()
        logger.info(f"Starting text quality evaluation for {len(processed_texts)} texts")
        
        # 创建批次
        batch_size = self.batch_size
        batches = [processed_texts[i:i + batch_size] for i in range(0, len(processed_texts), batch_size)]
        
        all_results = []
        total_passed = 0
        total_failed = 0
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating text quality")):
            score_inputs = []
            batch_metadata = []
            
            # 准备批处理输入
            for item in batch:
                try:
                    paper_content = item.get('md_content', '')
                    
                    if len(paper_content.strip()) < 100:
                        logger.warning(f"Text {item.get('paper_name', 'unknown')} too short, marking as failed")
                        all_results.append({
                            **item,
                            'judgment': {
                                'suitable_for_qa': False,
                                'reason': '文本内容过短',
                                'score_text': '【否】文本内容过短，不足以生成有意义的问题'
                            }
                        })
                        total_failed += 1
                        continue
                    
                    # 构建评分提示
                    score_prompt = self.score_template.replace("{academic_paper}", paper_content)
                    score_messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": score_prompt}
                    ]
                    
                    # 应用聊天模板
                    data_li = self._format_messages(score_messages)
                    
                    # 检查长度并截断
                    if self._encode_length(data_li) > self.config.max_model_len - 1024:
                        logger.warning(f"Prompt too long for {item.get('paper_name', 'unknown')}, truncating")
                        # 截断内容
                        max_content_len = self.config.max_model_len - 2048
                        truncated_content = paper_content[:max_content_len]
                        score_prompt = self.score_template.replace("{academic_paper}", truncated_content)
                        score_messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": score_prompt}
                        ]
                        data_li = self._format_messages(score_messages)
                    
                    score_inputs.append(data_li)
                    batch_metadata.append(item)
                    
                except Exception as e:
                    logger.error(f"Error processing text {item.get('paper_name', 'unknown')}: {e}")
                    all_results.append({
                        **item,
                        'judgment': {
                            'suitable_for_qa': False,
                            'reason': f'处理错误: {str(e)}',
                            'score_text': f'【否】处理过程中出现错误: {str(e)}'
                        }
                    })
                    total_failed += 1
                    continue
            
            # 批量推理
            if score_inputs:
                try:
                    score_outputs = await self._generate_async(score_inputs, use_tqdm=False)
                    
                    # 处理输出
                    for output, metadata in zip(score_outputs, batch_metadata):
                        score_text = output.outputs[0].text.strip()
                        
                        # 判断是否通过评估
                        suitable = "【是】" in score_text or "是】" in score_text
                        
                        result = {
                            **metadata,
                            'judgment': {
                                'suitable_for_qa': suitable,
                                'reason': '通过质量评估' if suitable else '未通过质量评估',
                                'score_text': score_text
                            }
                        }
                        
                        all_results.append(result)
                        
                        if suitable:
                            total_passed += 1
                        else:
                            total_failed += 1
                            
                except Exception as e:
                    logger.error(f"Error in batch inference: {e}")
                    # 批量推理失败，将整批标记为失败
                    for metadata in batch_metadata:
                        all_results.append({
                            **metadata,
                            'judgment': {
                                'suitable_for_qa': False,
                                'reason': f'批量推理错误: {str(e)}',
                                'score_text': f'【否】批量推理过程中出现错误: {str(e)}'
                            }
                        })
                        total_failed += 1
        
        # 记录统计信息
        processing_time = time.time() - start_time
        logger.info(f"Text quality evaluation completed in {processing_time:.2f}s")
        logger.info(f"Total: {len(processed_texts)}, Passed: {total_passed}, Failed: {total_failed}")
        
        return all_results

    def _save_intermediate_results(self, results: List[Dict], jsonl_path: str):
        """保存中间结果到JSONL文件"""
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def generate_question_data(self, jsonl_file_input: str, jsonl_file_output: str, add_prompt: str = "") -> Dict[str, Any]:
        """
        根据通过质量评估的文本生成问题
        
        Args:
            jsonl_file_input: 输入JSONL文件路径
            jsonl_file_output: 输出JSONL文件路径
            add_prompt: 额外的提示信息，用于指定问题类型和示例
            
        Returns:
            dict: 生成结果统计
        """
        #start_time = time.time()
        # 确保 stats 存在
        if not hasattr(self, 'stats'):
            self.stats = {}
        question_list = []
        start_time = time.time()
        stats = {
            "total_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_questions": 0
        }
        # 添加调试 1：检查额外提示
        if add_prompt:
            logger.info(f"=== 使用额外提示 ===")
            logger.info(f"额外提示内容: {add_prompt[:200]}...")
        # 读取输入数据
        suitable_papers = []
        with open(jsonl_file_input, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("stats") == 1:
                        suitable_papers.append(data)
                    
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    continue
        
        logger.info(f"Found {len(suitable_papers)} suitable papers for question generation")
        stats["total_processed"] = len(suitable_papers)
        
        # 批处理生成问题
        batches = self.to_batch(suitable_papers, self.batch_size)
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Generating questions")):
            inputs = []
            batch_metadata = []
            
            # 准备输入
            for paper_data in batch:
                try:
                    paper_name = paper_data["paper_name"]
                    paper_content = paper_data["paper_content"]
                    # 添加调试 2：检查输入内容
                    logger.info(f"处理论文: {paper_name}")
                    logger.info(f"内容长度: {len(paper_content)} 字符")
                    # 构建生成提示 - 如果有额外提示，则添加到模板中
                    if add_prompt:
                        # 在模板的要求部分后添加额外提示
                        # enhanced_template = self.prompt_template.replace(
                        #     "### 输出格式要求：",
                        #     f"{add_prompt}\n\n### 输出格式要求："
                        # )
                        # enhanced_template = self.prompt_template.replace(
                        #     "##【格式要求】",  # 使用实际存在的字符串
                        #     f"\n### 额外要求：\n{add_prompt}\n\n##【格式要求】"
                        # )
                        # 确保add_prompt插入在核心要求之前
                        if "##【核心要求】" in self.prompt_template:
                            enhanced_template = self.prompt_template.replace(
                                "##【核心要求】",
                                f"{add_prompt}\n\n##【核心要求】"
                            )
                        else:
                            # 如果找不到标记，直接添加在开头
                            enhanced_template = f"{add_prompt}\n\n{self.prompt_template}"
                        generate_prompt = enhanced_template.replace("{academic_paper}", paper_content)
                    else:
                        generate_prompt = self.prompt_template.replace("{academic_paper}", paper_content)
                    # print("paper_content:----------------------------------")
                    # print(paper_content)
                    # print("----------------------------------")
                    # 添加调试 3：检查最终提示
                    # print("++++++++++++++++++++generate_prompt+++++++++++++++++++")
                    # logger.info(f"生成提示长度: {len(generate_prompt)}")
                    # logger.info(f"生成提示预览: {generate_prompt[:1000]}...")
                    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": generate_prompt}
                    ]
                    
                    # 应用聊天模板
                    prompt = self._format_messages(messages)
                    
                    # 检查长度
                    if self._encode_length(prompt) > self.config.max_model_len - 1024:
                        logger.warning(f"Prompt too long for {paper_name}, truncating")
                        # 截断内容
                        max_content_len = self.config.max_model_len - 3000
                        truncated_content = paper_content[:max_content_len]
                        generate_prompt = self.prompt_template.replace("{academic_paper}", truncated_content)
                        messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": generate_prompt}
                        ]
                        prompt = self._format_messages(messages)
                    
                    inputs.append(prompt)
                    batch_metadata.append(paper_data)
                    
                except Exception as e:
                    logger.error(f"Error preparing input for {paper_data.get('paper_name', 'unknown')}: {e}")
                    stats["failed_generations"] += 1
                    continue
            
            if not inputs:
                continue
            
            # 批量生成
#             try:
#                 # 添加调试 4：生成前的信息
#                 logger.info(f"批次 {batch_idx + 1}: 准备生成 {len(inputs)} 个问题集")

#                 outputs = self._generate(inputs, use_tqdm=False)
                
#                 # 添加调试 5：生成后的原始输出
#                 logger.info(f"批次 {batch_idx + 1}: 收到 {len(outputs)} 个输出")
#                 # 处理输出
#                 # 在 generate_question_data 方法中，找到处理输出的部分，修改如下：

#                 # 处理输出
#                 for idx, output in enumerate(outputs):
#                     paper_data = batch_metadata[idx]
#                     question_text = output.outputs[0].text
#                     # 添加调试 6：查看原始生成文本
#                     logger.info(f"\n=== 论文: {paper_data['paper_name']} ===")
#                     logger.info(f"原始生成文本长度: {len(question_text)} 字符")
#                     logger.info(f"原始生成文本预览:\n{question_text[:1000]}")

#                     # 【修复：在这里初始化 question_list】
#                     question_list = []

#                     # 添加调试 7：解析结果
#                     # 解析问题列表
#                     question_list = self._parse_questions(question_text)
#                     logger.info(f"解析出 {len(question_list)} 个问题")

#                     # 保存结果
#                     result = paper_data.copy()
#                     result["question_list"] = question_list
#                     result["question_generation_time"] = datetime.now().isoformat()
#                     result["model_used"] = self.model_name

#                     # 写入输出文件
#                     with open(jsonl_file_output, "a", encoding="utf-8") as f:
#                         f.write(json.dumps(result, ensure_ascii=False) + "\n")

#                     if question_list:
#                         stats["successful_generations"] += 1
#                         stats["total_questions"] += len(question_list)
#                     else:
#                         stats["failed_generations"] += 1
#                         logger.warning(f"No questions generated for {paper_data['paper_name']}")
#             except Exception as e:
#                 logger.error(f"Error in batch generation: {e}")
#                 stats["failed_generations"] += len(batch_metadata)

#             try:
#                 outputs = self._generate(inputs, use_tqdm=False)

#                 # 处理输出
#                 for idx, output in enumerate(outputs):
#                     paper_data = batch_metadata[idx]
#                     question_text = output.outputs[0].text

#                     # ================= 修改：这里调用_parse_questions =================
#                     question_list = self._parse_questions(question_text)  # 这里返回解析后的问题列表
#                     print("question_list=========================")
#                     print(question_list)
#                     print("====================================")
#                     # 保存结果
#                     result = paper_data.copy()
#                     result["question_list"] = question_list  # 这里使用解析出的问题列表
#                     result["question_generation_time"] = datetime.now().isoformat()
#                     result["model_used"] = self.model_name

#                     # 写入输出文件
#                     with open(jsonl_file_output, "a", encoding="utf-8") as f:
#                         f.write(json.dumps(result, ensure_ascii=False) + "\n")

#                     if question_list:
#                         stats["successful_generations"] += 1
#                         stats["total_questions"] += len(question_list)
#                     else:
#                         stats["failed_generations"] += 1
#                         logger.warning(f"No questions generated for {paper_data['paper_name']}")
            try:
                outputs = self._generate(inputs, use_tqdm=False)

                # 处理输出
                for idx, output in enumerate(outputs):
                    paper_data = batch_metadata[idx]
                    question_text = output.outputs[0].text
                    print("********************question_text******************")
                    print(question_text)
                    print("***************************************************")
                    # 解析问题列表
                    # ================= 修改：这里调用_parse_questions =================
                    question_list = self._parse_questions(question_text)  # 这里返回解析后的问题列表
                    print("question_list=========================")
                    print(question_list)
                    print("====================================")

                    # 添加调试信息
                    logger.info(f"解析出 {len(question_list)} 个问题: {question_list}")

                    # 保存结果
                    result = paper_data.copy()
                    result["question_list"] = question_list
                    result["question_generation_time"] = datetime.now().isoformat()
                    result["model_used"] = self.model_name

                    # 写入输出文件
                    with open(jsonl_file_output, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                    if question_list:
                        stats["successful_generations"] += 1
                        stats["total_questions"] += len(question_list)
                    else:
                        stats["failed_generations"] += 1
                        logger.warning(f"No questions generated for {paper_data['paper_name']}")

            except Exception as e:
                logger.error(f"Error in batch generation: {e}")
                stats["failed_generations"] += len(batch_metadata)        
        # 更新统计
        stats["processing_time"] = time.time() - start_time
        #self.stats["generated_questions"] = stats["total_questions"]
        if hasattr(self, 'stats'):
            self.stats["generated_questions"] = stats["total_questions"]
        logger.info(f"Question generation completed: {stats}")
            # 添加调试 8：最终统计
        logger.info(f"\n=== 问题生成完成统计 ===")
        logger.info(f"总处理数: {stats['total_processed']}")
        logger.info(f"成功生成: {stats['successful_generations']}")
        logger.info(f"失败生成: {stats['failed_generations']}")
        logger.info(f"总问题数: {stats['total_questions']}")
        logger.info(f"处理时间: {stats['processing_time']:.2f}秒")
        return stats

    # def _parse_questions(self, text: str) -> List[str]:
    #     """
    #     解析生成的问题文本
        
    #     Args:
    #         text: 生成的文本
            
    #     Returns:
    #         list: 问题列表
    #     """
    #     questions = []
        
    #     # 首先尝试从</think>标签后提取
    #     if "</think>" in text:
    #         text = text.split("</think>")[-1].strip()
        
    #     # 查找问题模式
    #     lines = text.split('\n')
    #     for line in lines:
    #         line = line.strip()
    #         # 匹配 [[1]], [[2]], [[3]] 格式
    #         match = re.match(r'\[\[(\d+)\]\]\s*(.+)', line)
    #         if match:
    #             question = match.group(2).strip()
    #             if question:
    #                 questions.append(question)
        
    #     # 如果没有找到标准格式，尝试其他格式
    #     if not questions:
    #         for i, line in enumerate(lines):
    #             line = line.strip()
    #             # 匹配 1. 2. 3. 格式
    #             match = re.match(r'^(\d+)\.\s*(.+)', line)
    #             if match:
    #                 question = match.group(2).strip()
    #                 if question and len(question) > 10:
    #                     questions.append(question)
        
    #     return questions
#     def _clean_question_text(text: str) -> str:
#         """清理问题文本，避免重复问号"""
#         # 移除HTML标签
#         clean_text = re.sub(r"</?think>", "", text)
#         # 移除多余换行
#         clean_text = re.sub(r"\n+", " ", clean_text).strip()

#         # 确保以问号结尾但避免重复
#         if clean_text.endswith(('?', '？')):
#             # 如果已经是问号结尾，不再添加
#             return clean_text
#         elif clean_text.endswith('.'):
#             # 如果以句号结尾，替换为问号
#             return clean_text[:-1] + '?'
#         else:
#             # 否则添加问号
#             return clean_text + '?'
    def _clean_question_text(self, text: str) -> str:
        """清理问题文本，避免重复问号"""
        # 移除HTML标签
        clean_text = re.sub(r"</?think>", "", text)
        # 移除多余换行
        clean_text = re.sub(r"\n+", " ", clean_text).strip()
        
        # 确保以问号结尾但避免重复
        if clean_text.endswith(('?', '？')):
            # 如果已经是问号结尾，不再添加
            return clean_text
        elif clean_text.endswith('.'):
            # 如果以句号结尾，替换为问号
            return clean_text[:-1] + '?'
        else:
            # 否则添加问号
            return clean_text + '?'
    
    # 2. 添加新的方法：处理数据列表而非文件路径
    # 在 SemiconductorQAGenerator 类中添加以下两个方法：

    # def convert_questionlist_li_data_from_list(self, question_data_list: list) -> list:
    #     """
    #     从问题数据列表转换为单独的问题记录列表

    #     Args:
    #         question_data_list: 包含分类问题的数据列表

    #     Returns:
    #         list: 转换后的单独问题列表
    #     """
    #     logger.info(f"Converting {len(question_data_list)} question data items to individual questions")

    #     converted_data = []
    #     converted_count = 0

    #     for item in question_data_list:
    #         try:
    #             paper_name = item.get("paper_name", "")
    #             source_content = item.get("source_content", "")
    #             questions_dict = item.get("questions", {})
    #             source_info = item.get("source_info", {})

    #             # 遍历所有问题类型
    #             for question_type, questions in questions_dict.items():
    #                 for question_data in questions:
    #                     converted_item = {
    #                         "paper_name": paper_name,
    #                         "paper_content": source_content,  # 保持与原方法一致的字段名
    #                         "question": question_data.get("question", ""),
    #                         "question_type": question_type,
    #                         "source_info": source_info,
    #                         "stats": 1,  # 标记为已通过质量评估
    #                         "timestamp": datetime.now().isoformat()
    #                     }
    #                     converted_data.append(converted_item)
    #                     converted_count += 1

    #         except Exception as e:
    #             logger.error(f"Error converting item {item.get('paper_name', 'unknown')}: {e}")
    #             continue

    #     logger.info(f"Conversion completed! Generated {converted_count} individual question records")
    #     return converted_data
    def convert_questionlist_li_data_from_list(self, question_data_list: list) -> list:
        """
        从问题数据列表转换为单独的问题记录列表

        Args:
            question_data_list: 包含分类问题的数据列表

        Returns:
            list: 转换后的单独问题列表
        """
        logger.info(f"Converting {len(question_data_list)} question data items to individual questions")

        converted_data = []
        converted_count = 0
        skipped_count = 0  # 跟踪跳过的无效项

        for item in question_data_list:
            # 添加类型检查：确保item是字典
            if not isinstance(item, dict):
                logger.error(f"Skipping invalid item (type: {type(item)}): {str(item)[:100]}...")
                skipped_count += 1
                continue
                
            try:
                paper_name = item.get("paper_name", "")
                source_content = item.get("source_content", "")
                questions_dict = item.get("questions", {})
                source_info = item.get("source_info", {})
                
                # 添加questions_dict的类型检查
                if not isinstance(questions_dict, dict):
                    logger.error(f"Skipping item {paper_name}: 'questions' field is not a dict (type: {type(questions_dict)})")
                    skipped_count += 1
                    continue

                # 遍历所有问题类型
                for question_type, questions in questions_dict.items():
                    # 确保questions是列表
                    if not isinstance(questions, list):
                        logger.error(f"Skipping {question_type} in {paper_name}: questions not a list (type: {type(questions)})")
                        skipped_count += 1
                        continue
                        
                    for question_data in questions:
                        # 确保question_data是字符串或字典
                        if isinstance(question_data, str):
                            # 如果是字符串，直接使用
                            question_text = question_data
                        elif isinstance(question_data, dict):
                            # 如果是字典，从中提取问题文本
                            question_text = question_data.get("question", "")
                        else:
                            logger.error(f"Skipping question in {paper_name}/{question_type}: invalid type {type(question_data)}")
                            skipped_count += 1
                            continue
                            
                        converted_item = {
                            "paper_name": paper_name,
                            "paper_content": source_content,
                            "question": question_text,  # 使用提取的问题文本
                            "question_type": question_type,
                            "source_info": source_info,
                            "stats": 1,
                            "timestamp": datetime.now().isoformat()
                        }
                        converted_data.append(converted_item)
                        converted_count += 1

            except Exception as e:
                logger.error(f"Error converting item {item.get('paper_name', 'unknown')}: {e}")
                skipped_count += 1
                continue

        # 添加转换统计信息
        logger.info(f"Conversion completed! Generated {converted_count} individual question records")
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid items during conversion")
        
        return converted_data


    async def judge_question_data_from_list(self, converted_data_list: list) -> list:
        """
        从数据列表评估问题质量

        Args:
            converted_data_list: 转换后的问题数据列表

        Returns:
            list: 评估后的问题数据列表
        """
        start_time = time.time()
        logger.info(f"Starting question quality evaluation for {len(converted_data_list)} questions")

        # 创建批次
        batches = [converted_data_list[i:i + self.batch_size] for i in range(0, len(converted_data_list), self.batch_size)]

        all_results = []
        total_passed = 0
        total_failed = 0

        for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating question quality")):
            evaluator_inputs = []
            batch_metadata = []

            # 准备批处理输入
            for item in batch:
                try:
                    paper_content = item.get('paper_content', '')
                    question = item.get('question', '')

                    if not paper_content or not question:
                        logger.warning(f"Missing paper_content or question for {item.get('paper_name', 'unknown')}")
                        all_results.append({
                            **item,
                            'quality_score': 0,
                            'evaluation_result': '【否】缺少必要信息',
                            'passed': False
                        })
                        total_failed += 1
                        continue

                    # 构建评估提示
                    evaluator_prompt = self.evaluator_template.replace(
                        "{academic_paper}", paper_content
                    ).replace(
                        "{academic_question}", question
                    )
                    
                    print("++++++++++++++++paper_content-----------")
                    print(paper_content)
                    print("++++++++++++++++++++++++++++++++++++++++")
                    print("----------------evaluator_prompt-----------")
                    print(evaluator_prompt)
                    print("-------------------------------------------")

                    evaluator_messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": evaluator_prompt}
                    ]

                    # 应用聊天模板
                    prompt = self._format_messages(evaluator_messages)

                    # 检查长度
                    if self._encode_length(prompt) > self.config.max_model_len - 1024:
                        logger.warning(f"Prompt too long for {item.get('paper_name', 'unknown')}, truncating")
                        # 截断内容
                        max_content_len = self.config.max_model_len - 2048
                        truncated_content = paper_content[:max_content_len]
                        evaluator_prompt = self.evaluator_template.replace(
                            "{academic_paper}", truncated_content
                        ).replace(
                            "{academic_question}", question
                        )
                        evaluator_messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": evaluator_prompt}
                        ]
                        prompt = self._format_messages(evaluator_messages)

                    evaluator_inputs.append(prompt)
                    batch_metadata.append(item)

                except Exception as e:
                    logger.error(f"Error processing question {item.get('paper_name', 'unknown')}: {e}")
                    all_results.append({
                        **item,
                        'quality_score': 0,
                        'evaluation_result': f'【否】处理错误: {str(e)}',
                        'passed': False
                    })
                    total_failed += 1
                    continue

            # 批量推理
            if evaluator_inputs:
                try:
                    outputs = await self._generate_async(evaluator_inputs, use_tqdm=False)

                    # 处理输出
                    for output, metadata in zip(outputs, batch_metadata):
                        evaluator_text = output.outputs[0].text.strip()

                        # 判断是否通过评估
                        suitable = "【是】" in evaluator_text or "是】" in evaluator_text
                        quality_score = 1 if suitable else 0

                        result = {
                            **metadata,
                            'quality_score': quality_score,
                            'evaluation_result': evaluator_text,
                            'passed': suitable,
                            'evaluation_time': datetime.now().isoformat()
                        }

                        all_results.append(result)

                        if suitable:
                            total_passed += 1
                        else:
                            total_failed += 1

                except Exception as e:
                    logger.error(f"Error in batch evaluation: {e}")
                    # 批量推理失败，将整批标记为失败
                    for metadata in batch_metadata:
                        all_results.append({
                            **metadata,
                            'quality_score': 0,
                            'evaluation_result': f'【否】批量推理错误: {str(e)}',
                            'passed': False
                        })
                        total_failed += 1

        # 记录统计信息
        processing_time = time.time() - start_time
        logger.info(f"Question quality evaluation completed in {processing_time:.2f}s")
        logger.info(f"Total: {len(converted_data_list)}, Passed: {total_passed}, Failed: {total_failed}")

        return all_results
    
#     def _parse_questions(self, text: str) -> List[str]:
#         """解析模型生成的响应，提取问题列表"""
#         # 先记录完整响应以便调试
#         logger.debug(f"完整模型响应:\n{text}")

#         # 使用传入的 text 参数
#         response_text = text

#         questions = []

#         # 方法1: 匹配[[1]]格式的问题
#         pattern1 = r"\[\[(\d+)\]\]\s(.+?)(?=\n\[\[\d+\]\]|\Z)"
#         matches1 = re.findall(pattern1, response_text, re.DOTALL)
#         if matches1:
#             for idx, question_text in matches1:
#                 #clean_question = re.sub(r"</?think>|[\n]{2,}", " ", question_text)
#                 clean_question = self._clean_question_text(question_text)
#                 clean_question = clean_question.strip()
#                 if not clean_question.endswith(('?', '？')):
#                     if clean_question.endswith('.'):
#                         clean_question = clean_question[:-1] + '?'
#                     else:
#                         clean_question += '?'
#                 questions.append(clean_question)
#             return questions

#         # 方法2: 匹配1. 问题内容 格式
#         pattern2 = r"^\s*(\d+\.)\s(.+?)(?=\n\d+\.|\Z)"
#         matches2 = re.findall(pattern2, response_text, re.MULTILINE | re.DOTALL)
#         if matches2:
#             for num, question_text in matches2:
#                 clean_question = question_text.strip()
#                 if not clean_question.endswith(('?', '？')):
#                     clean_question += '?'
#                 questions.append(clean_question)
#             return questions

#         # 方法3: 匹配Q1: 问题内容 格式
#         pattern3 = r"^\s*(Q\d+:|问题\d+:)\s(.+?)(?=\nQ\d+:|问题\d+:|\Z)"
#         matches3 = re.findall(pattern3, response_text, re.MULTILINE | re.DOTALL)
#         if matches3:
#             for prefix, question_text in matches3:
#                 clean_question = question_text.strip()
#                 if not clean_question.endswith(('?', '？')):
#                     clean_question += '?'
#                 questions.append(clean_question)
#             return questions

#         # 方法4: 提取所有包含问号的行
#         lines = response_text.split('\n')
#         for line in lines:
#             if '?' in line or '？' in line:
#                 # 移除标签和前缀
#                 clean_line = re.sub(r"</?think>|^[\s\-*]+", "", line).strip()
#                 if clean_line:
#                     # 确保以问号结尾
#                     if not clean_line.endswith(('?', '？')):
#                         clean_line += '?'
#                     questions.append(clean_line)

#         # 如果没有找到问题，记录警告
#         if not questions:
#             logger.warning(f"无法从响应中解析出问题:\n{response_text}")

#         return questions
    # def _parse_questions(self, response_text: str) -> List[str]:
    #     """增强版问题解析函数 - 处理多种格式"""
    #     # 先记录完整响应以便调试
    #     logger.debug(f"完整模型响应:\n{response_text}")

    #     questions = []

    #     # 方法1: 匹配[[1]]格式的问题
    #     pattern1 = r"\[\[(\d+)\]\]\s(.+?)(?=\n\[\[\d+\]\]|\Z)"
    #     matches1 = re.findall(pattern1, response_text, re.DOTALL)
    #     if matches1:
    #         for idx, question_text in matches1:
    #             clean_question = self._clean_question_text(question_text)
    #             questions.append(clean_question)
    #         return questions

    #     # 方法2: 匹配1. 问题内容 格式
    #     pattern2 = r"^\s*(\d+\.)\s(.+?)(?=\n\d+\.|\Z)"
    #     matches2 = re.findall(pattern2, response_text, re.MULTILINE | re.DOTALL)
    #     if matches2:
    #         for num, question_text in matches2:
    #             clean_question = self._clean_question_text(question_text)
    #             questions.append(clean_question)
    #         return questions

    #     # 方法3: 匹配Q1: 问题内容 格式
    #     pattern3 = r"^\s*(Q\d+:|问题\d+:)\s(.+?)(?=\nQ\d+:|问题\d+:|\Z)"
    #     matches3 = re.findall(pattern3, response_text, re.MULTILINE | re.DOTALL)
    #     if matches3:
    #         for prefix, question_text in matches3:
    #             clean_question = self._clean_question_text(question_text)
    #             questions.append(clean_question)
    #         return questions

    #     # 方法4: 提取所有包含问号的行
    #     lines = response_text.split('\n')
    #     for line in lines:
    #         if '?' in line or '？' in line:
    #             # 移除标签和前缀
    #             clean_line = re.sub(r"</?think>|^[\s\-*]+", "", line).strip()
    #             if clean_line:
    #                 # 使用清理函数确保正确格式
    #                 clean_question = self._clean_question_text(clean_line)
    #                 questions.append(clean_question)

    #     # 如果没有找到问题，记录警告
    #     if not questions:
    #         logger.warning(f"无法从响应中解析出问题:\n{response_text}")

    #     return questions
    def _parse_questions(self, response_text: str) -> List[str]:
        """增强版问题解析函数 - 确保每个问题独立"""
        logger.debug(f"完整模型响应:\n{response_text}")
        questions = []

        # 预处理：移除XML标签
        cleaned_text = re.sub(r"</?think>", "", response_text)
        
        # 方法1: 匹配[[1]]格式的问题
        pattern1 = r"\[\[(\d+)\]\]\s*([^[]+?)(?=\n\[\[\d+\]\]|\Z)"
        matches1 = re.findall(pattern1, cleaned_text, re.DOTALL)
        if matches1:
            for _, question_text in matches1:
                # 处理多问题分割
                self._split_and_add_questions(question_text, questions)
            return questions

        # 方法2: 匹配1. 问题内容 格式
        pattern2 = r"^\s*(\d+\.)\s*([^\n]+?)(?=\n\d+\.|\Z)"
        matches2 = re.findall(pattern2, cleaned_text, re.MULTILINE | re.DOTALL)
        if matches2:
            for _, question_text in matches2:
                self._split_and_add_questions(question_text, questions)
            return questions

        # 方法3: 匹配Q1: 问题内容 格式
        pattern3 = r"^\s*(?:Q\d+|问题\d+)[:：]\s*([^\n]+?)(?=\n(?:Q\d+|问题\d+)|$)"
        matches3 = re.findall(pattern3, cleaned_text, re.MULTILINE | re.DOTALL)
        if matches3:
            for question_text in matches3:
                self._split_and_add_questions(question_text, questions)
            return questions

        # 方法4: 提取所有问题句子（核心改进）
        # 匹配以问号结尾的独立句子，支持中文/英文问号
        question_sentences = re.findall(
            r'([^。！？.!?\n]*[？?][。！.!]?)', 
            cleaned_text,
            re.MULTILINE
        )
        
        for q in question_sentences:
            clean_q = self._clean_question_text(q.strip())
            if clean_q:
                questions.append(clean_q)
        
        # 兜底策略：按行分割
        if not questions:
            lines = cleaned_text.split('\n')
            for line in lines:
                if '?' in line or '？' in line:
                    self._split_and_add_questions(line, questions)
        
        # 最终检查
        if not questions:
            logger.warning(f"无法从响应中解析出问题:\n{response_text}")
        
        return questions

    def _split_and_add_questions(self, text: str, questions_list: list):
        """将包含多个问题的文本拆分为独立问题"""
        # 分割规则：按问号分割，同时保留问号
        sub_questions = re.split(r'(?<=[?？])', text)
        for q in sub_questions:
            q = q.strip()
            if not q:
                continue
            
            # 确保问题以问号结尾
            if not q.endswith(('?', '？')):
                q += '？' if any(char in q for char in ['？', '?', '吗', '何', '怎', '为什么']) else ''
            
            clean_q = self._clean_question_text(q)
            if clean_q:
                questions_list.append(clean_q)

    def _clean_question_text(self, text: str) -> str:
        """清理问题文本"""
        # 移除开头编号/符号
        text = re.sub(r'^[\s\d.、:：]*', '', text)
        # 合并连续问号
        text = re.sub(r'[？?]{2,}', '？', text)
        # 确保以问号结尾
        if not text.endswith(('?', '？')):
            text += '？'
        return text.strip()
    def generate_answers(self, qa_data_file: str, output_file: str, use_cot: bool = True) -> Dict[str, Any]:
        """
        为问题生成答案
        
        Args:
            qa_data_file: 包含问题的JSON文件路径
            output_file: 输出文件路径
            use_cot: 是否使用Chain of Thought方式
            
        Returns:
            dict: 生成结果统计
        """
        start_time = time.time()
        stats = {
            "total_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "no_answer_count": 0
        }
        
        # 读取QA数据
        qa_items = []
        with open(qa_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                qa_items = data
            else:
                logger.error("Invalid QA data format")
                return stats
        
        logger.info(f"Found {len(qa_items)} QA items for answer generation")
        stats["total_processed"] = len(qa_items)
        
        # 准备答案生成提示模板
        if use_cot:
            answer_template = """你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，可能含有与问题相近的干扰信息，请仔细阅读每个切片后再作答，不得出现错误,你需要准确且完整的回答问题：{question}

请注意：如果问题的难度超过你的能力或者问题不能基于下面的论文原文回答（严格执行！）或者你无法确定回答准确性，请输出'无法作答!'。

论文Context: {context}

<think>
首先，需要仔细阅读Context这篇学术论文，理解其核心内容和技术要点。

分析论文的主要内容：
1. 识别论文讨论的核心技术问题
2. 找出涉及逻辑推理的技术原理和机制

接下来，基于以下原则回答问题：
- 答案必须基于论文内容，答案可以从论文中找到出处（严格执行！）
- 答案描述要清晰、完整、专业
- 避免使用"本文"、"论文"等自指表述
- 确保基于论文内容完整回答了问题(严格执行！)
</think>

##【核心要求】
### 回答准则：
(1) 仔细通读全文，找出涉及逻辑推理的文本部分，根据文本内容逻辑回答相关问题。（必须严格执行！）
(2) 答案要基于论文里的技术原理，描述务必清晰、全面、明确。答案中主语或名词的表述要精准、全面且通用。
(3) 避免在答案中引用文献或文章自定义的专有名词。
(4) 答案中的名词描述必须和论文一致，不能缩写。比如论文是 "OLED 材料"，问题不能写成 "材料"；论文是 "LTPS 器件"，问题不能写成 "器件"。
(7) 答案简洁：根据文本生成凝练、简洁的答案（不得冗余）。
(8) 非常关键：生成的答案完全是基于论文生成的，不能脱离论文，答案必须在原文中有出处（严格执行）。

### 思考的科学严谨性：思考过程要科学严谨，逐步推进。保证问题及对应答案源自论文内容，且答案在论文中有详细阐述。

## 【禁止事项】
× 禁止使用"本文/本研究/本实验"等论文自指表述
× 禁止脱离文本回答(严格执行！)
× 禁止只回答部分问题(严格执行！)
× 禁止答案不完整，回答必须完整不得出现答案不完整（仅引导句）的情况
    - **覆盖全面性**（对应问题完整性要求）：
    - 检查答案是否覆盖问题中所有技术要点（如：机制、数据、比较项等）
    - 验证是否响应了问题中的每个子问题（如："如何...""为何..."等）
    - 示例：若问题含「迁移率变化机制+物相影响」，答案需同时包含二者

    -**表述完整性**（对应描述完整性要求）：
    - 答案必须为完整陈述句，禁止未完成的引导句（如"机制如下："后无内容）
    - 语义自闭环：主语+谓语+技术对象+结论明确
    - 示例：❌"通过以下机制实现：" → ✅"通过X机制实现Y效果，因为..."

    -**独立可读性**（对应问题独立性要求）：
    - 脱离原文仍可理解：答案需自带关键参数（如迁移率数值）、机制名称（如氧空位调控）
    - 禁止依赖上下文：如不出现"如背景所述""根据上文"等
    - 示例：❌"如前所述提升迁移率" → ✅"迁移率从X提升至Y"


##【开始回答问题】：
- 首先，请提供有关如何回答问题的详细 reasoning。
- 在 reasoning 中，如果需要复制上下文中的某些句子，请将其包含在 ##begin_quote## 和 ##end_quote## 中。这意味着 ##begin_quote## 和 ##end_quote## 之外的内容不是直接从上下文中复制的。
- 结束你的回答，以 final answer 的形式 <ANSWER>: $answer，答案应该基于论文简洁准确完整的回答问题。
你必须以<Reasoning>: 开头，包含 reasoning 相关的内容；以 <ANSWER>: 开头，包含答案。

##【分析答案出处】（以【答案出处】为标识且放在<ANSWER>: 整体后面，必须严格执行）
- 仔细阅读文本和生成的答案，分析生成的答案在文本中的详细来源和出处。
- 如果答案有出处请将其包含在【答案出处】标识下面。
- 如果答案找不到出处请在【答案出处】标识下面写：没有出处。

"""
        else:
            answer_template = """{
    "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，请仔细阅读每个切片后再作答，不得出现错误。",
    "input": {
        "context": "{context}",
        "question": "{question}"
    },
    "output": {
        "answer": "根据切片中提供的有效信息对问题进行详尽的回答，推荐分点回答格式。"
    },
    "requirements": {
        "criteria": "根据提供的切片信息提取有效信息进行回答",
        "format": "输出内容必须用中文作答。",
        "reasoning" : "在系统内部的think推理过程中，请将参考用到的上下文内容包含在 ##begin_quote## 和 ##end_quote## 中。"
    }
}"""
        
        # 批处理生成答案
        batches = self.to_batch(qa_items, self.batch_size)
        results = []
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Generating answers")):
            inputs = []
            batch_metadata = []
            
            for qa_item in batch:
                try:
                    question = qa_item.get("question", "")
                    context = qa_item.get("context", qa_item.get("paper_content", ""))
                    
                    if not question or not context:
                        logger.warning(f"Missing question or context for item: {qa_item.get('paper_name', 'unknown')}")
                        continue
                    
                    # 构建答案生成提示
                    prompt_content = answer_template.format(
                        question=question,
                        context=context
                    )
                    
                    messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": prompt_content}
                    ]
                    
                    # 应用聊天模板
                    prompt = self._format_messages(messages)
                    #print("----------prompt----------")
                    #print(prompt)
                    #print("---------------------------")
                    # 检查长度
                    if self._encode_length(prompt) > self.config.max_model_len - 1024:
                        logger.warning(f"Prompt too long, truncating context")
                        # 截断上下文
                        max_context_len = self.config.max_model_len - 3000
                        truncated_context = context[:max_context_len]
                        prompt_content = answer_template.format(
                            question=question,
                            context=truncated_context
                        )
                        messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": prompt_content}
                        ]
                        prompt = self._format_messages(messages)
                    
                    inputs.append(prompt)
                    batch_metadata.append(qa_item)
                    
                except Exception as e:
                    logger.error(f"Error preparing input: {e}")
                    stats["failed_generations"] += 1
                    continue
            
            if not inputs:
                continue
            
            # 批量生成答案
            try:
                outputs = self._generate(inputs, use_tqdm=False)
                
                # 处理输出
                for idx, output in enumerate(outputs):
                    qa_item = batch_metadata[idx]
                    answer_text = output.outputs[0].text
                    print(use_cot)
                    # 解析答案
                    if use_cot:
                        # 提取COT格式的答案
                        answer = self._parse_cot_answer(answer_text)
                        print("+++++++++++answer+++++++++++++")
                        print(answer)
                        print("++++++++++++++++++++++++++++++")
                    else:
                        # 直接使用生成的答案
                        answer = answer_text.strip()
                    
                    # 检查是否无法作答
                    if "无法作答" in answer:
                        stats["no_answer_count"] += 1
                    
                    # 构建结果
                    result = qa_item.copy()
                    result["answer"] = answer
                    result["answer_generation_time"] = datetime.now().isoformat()
                    result["answer_model"] = self.model_name
                    result["use_cot"] = use_cot
                    
                    results.append(result)
                    stats["successful_generations"] += 1
                    
            except Exception as e:
                logger.error(f"Error in batch answer generation: {e}")
                stats["failed_generations"] += len(batch_metadata)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 更新统计
        stats["processing_time"] = time.time() - start_time
        logger.info(f"Answer generation completed: {stats}")
        
        return stats
    
    # def _parse_cot_answer(self, text: str) -> str:
    #     """
    #     解析COT格式的答案
        
    #     Args:
    #         text: 生成的文本
            
    #     Returns:
    #         str: 提取的答案
    #     """
    #     # 查找<ANSWER>:标记
    #     answer_match = re.search(r'<ANSWER>:\s*(.+?)(?:\n|$)', text, re.DOTALL)
    #     if answer_match:
    #         return answer_match.group(1).strip()
        
    #     # 如果没有找到标准格式，尝试其他格式
    #     # 查找最后一段作为答案
    #     paragraphs = text.strip().split('\n\n')
    #     if paragraphs:
    #         last_paragraph = paragraphs[-1].strip()
    #         if last_paragraph and not last_paragraph.startswith('<Reasoning>'):
    #             return last_paragraph
        
    #     # 如果还是没有找到，返回整个文本
    #     return text.strip()
    def _parse_cot_answer(self, text: str) -> str:
        """
        解析COT格式的答案，保留<ANSWER>:后的所有内容
        
        Args:
            text: 生成的文本
                
        Returns:
            str: 提取的答案（包含所有内容）
        """
        print(f"=== 原始生成文本 ===\n{text}\n=== 原始文本结束 ===")
        
        # 提取<ANSWER>:后的所有内容
        answer_text = ""
        
        # 查找<ANSWER>:标记，提取后面的所有内容
        answer_match = re.search(r'<ANSWER>:\s*(.+)', text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            print(f"找到<ANSWER>标记，提取后续所有内容")
        else:
            # 如果没有找到<ANSWER>标记，使用备用策略
            print("未找到<ANSWER>标记，使用备用策略")
            
            # 尝试其他答案格式
            backup_patterns = [
                r'答案[：:]\s*(.+)',  # 中文格式
                r'Final Answer[：:]\s*(.+)',  # 英文格式
            ]
            
            for pattern in backup_patterns:
                backup_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if backup_match:
                    answer_text = backup_match.group(1).strip()
                    print(f"使用备用格式: {pattern}")
                    break
            
            # 如果还是没找到，使用整个文本
            if not answer_text:
                print("使用整个文本作为答案")
                answer_text = text.strip()
        
        print(f"=== 最终答案 ===\n{answer_text}\n=== 最终答案结束 ===")
        return answer_text

#     async def judge_question_data(self, jsonl_file_input: str, save_paths: List[str]) -> Dict[str, Any]:
#         """
#         评估生成的问题质量
        
#         Args:
#             jsonl_file_input: 输入JSONL文件路径
#             save_paths: 保存路径列表
            
#         Returns:
#             dict: 评估结果统计
#         """
#         start_time = time.time()
#         stats = {
#             "total_questions": 0,
#             "passed_questions": 0,
#             "failed_questions": 0,
#             "error_questions": 0
#         }
        
#         # 读取输入数据
#         questions_to_evaluate = []
#         with open(jsonl_file_input, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     data = json.loads(line)
#                     if data.get("stats") == 1 and "question_li" in data:
#                         questions_to_evaluate.append(data)
#                 except Exception as e:
#                     logger.error(f"Error parsing line: {e}")
#                     continue
        
#         logger.info(f"Found {len(questions_to_evaluate)} questions to evaluate")
#         stats["total_questions"] = len(questions_to_evaluate)
        
#         # 批处理评估
#         batches = self.to_batch(questions_to_evaluate, self.batch_size)
#         final_results = []
        
#         for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating questions")):
#             evaluator_inputs = []
#             batch_metadata = []
            
#             # 准备输入
#             for question_data in batch:
#                 try:
#                     paper_content = question_data["paper_content"]
#                     question_li = question_data["question_li"]
                    
#                     # 构建评估提示
#                     evaluator_prompt = self.evaluator_template.replace(
#                         "{academic_paper}", paper_content
#                     ).replace(
#                         "{academic_question}", question_li
#                     )
                    
#                     evaluator_messages = [
#                         {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
#                         {"role": "user", "content": evaluator_prompt}
#                     ]
                    
#                     # 应用聊天模板
#                     prompt = self._format_messages(evaluator_messages)
                    
#                     evaluator_inputs.append(prompt)
#                     batch_metadata.append(question_data)
                    
#                 except Exception as e:
#                     logger.error(f"Error preparing evaluation input: {e}")
#                     stats["error_questions"] += 1
#                     continue
            
#             if not evaluator_inputs:
#                 continue
            
#             # 批量评估
#             try:
#                 outputs = self._generate(evaluator_inputs, use_tqdm=False)
                
#                 # 处理输出
#                 for idx, output in enumerate(outputs):
#                     question_data = batch_metadata[idx]
#                     evaluator_text = output.outputs[0].text.strip()
                    
#                     # 提取评估结果
#                     lines = evaluator_text.split('\n')
#                     final_judgment = lines[-1] if lines else ""
                    
#                     # 判断是否通过
#                     is_passed = '【是】' in final_judgment
                    
#                     if is_passed:
#                         # 构建最终结果
#                         result = {
#                             "id": question_data["paper_name"],
#                             "paper_content": question_data["paper_content"],
#                             "question_li": question_data["question_li"],
#                             "evaluation_result": evaluator_text,
#                             "passed": True,
#                             "timestamp": datetime.now().isoformat()
#                         }
#                         final_results.append(result)
#                         stats["passed_questions"] += 1
#                     else:
#                         stats["failed_questions"] += 1
                        
#             except Exception as e:
#                 logger.error(f"Error in batch evaluation: {e}")
#                 stats["error_questions"] += len(batch_metadata)
        
#         # 保存结果
#         for save_path in save_paths:
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
#             # 如果文件已存在，追加结果
#             existing_results = []
#             if os.path.exists(save_path):
#                 try:
#                     with open(save_path, 'r', encoding='utf-8') as f:
#                         existing_results = json.load(f)
#                 except Exception as e:
#                     logger.error(f"Error loading existing results: {e}")
            
#             # 合并结果
#             all_results = existing_results + final_results
            
#             # 保存
#             with open(save_path, 'w', encoding='utf-8') as f:
#                 json.dump(all_results, f, ensure_ascii=False, indent=4)
            
#             logger.info(f"Evaluation results saved to {save_path}")
        
#         # 更新统计
#         stats["processing_time"] = time.time() - start_time
#         self.stats["passed_question_quality"] = stats["passed_questions"]
        
#         logger.info(f"Question evaluation completed: {stats}")
#         return stats
    # 在 SemiconductorQAGenerator 类中重新实现 judge_question_data 方法：
    async def judge_question_data(self, converted_data: list) -> list:
        """
        评估生成的问题质量

        Args:
            converted_data: 转换后的问题数据列表

        Returns:
            list: 评估后的QA数据列表，包含质量评分
        """
        start_time = time.time()
        logger.info(f"Starting quality evaluation for {len(converted_data)} questions")

        # 创建批次
        batches = [converted_data[i:i + self.batch_size] for i in range(0, len(converted_data), self.batch_size)]

        evaluated_results = []
        total_passed = 0
        total_failed = 0

        for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating question quality")):
            evaluator_inputs = []
            batch_metadata = []

            # 准备批处理输入
            for qa_item in batch:
                try:
                    paper_content = qa_item.get('paper_content', '')
                    question = qa_item.get('question', '')

                    if not question or not paper_content:
                        logger.warning(f"Missing question or content for {qa_item.get('paper_name', 'unknown')}")
                        # 直接标记为不合格
                        evaluated_results.append({
                            **qa_item,
                            'quality_score': 0,
                            'evaluation_result': '【否】缺少问题或内容',
                            'evaluation_time': datetime.now().isoformat()
                        })
                        total_failed += 1
                        continue

                    # 构建评估提示
                    evaluator_prompt = self.evaluator_template.replace(
                        "{academic_paper}", paper_content
                    ).replace(
                        "{academic_question}", question
                    )

                    messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": evaluator_prompt}
                    ]

                    # 应用聊天模板
                    prompt = self._format_messages(messages)

                    # 检查长度并截断
                    if self._encode_length(prompt) > self.config.max_model_len - 1024:
                        logger.warning(f"Prompt too long, truncating content")
                        max_content_len = self.config.max_model_len - 2048
                        truncated_content = paper_content[:max_content_len]
                        evaluator_prompt = self.evaluator_template.replace(
                            "{academic_paper}", truncated_content
                        ).replace(
                            "{academic_question}", question
                        )
                        messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": evaluator_prompt}
                        ]
                        prompt = self._format_messages(messages)

                    evaluator_inputs.append(prompt)
                    batch_metadata.append(qa_item)

                except Exception as e:
                    logger.error(f"Error preparing evaluation input: {e}")
                    evaluated_results.append({
                        **qa_item,
                        'quality_score': 0,
                        'evaluation_result': f'【否】准备评估时出错: {str(e)}',
                        'evaluation_time': datetime.now().isoformat()
                    })
                    total_failed += 1
                    continue

            # 批量推理
            if evaluator_inputs:
                try:
                    outputs = await self._generate_async(evaluator_inputs, use_tqdm=False)

                    # 处理输出
                    for output, qa_item in zip(outputs, batch_metadata):
                        evaluation_text = output.outputs[0].text.strip()

                        # 判断是否通过评估
                        is_high_quality = "【是】" in evaluation_text or "是】" in evaluation_text
                        quality_score = 1 if is_high_quality else 0

                        result = {
                            **qa_item,
                            'quality_score': quality_score,
                            'evaluation_result': evaluation_text,
                            'evaluation_time': datetime.now().isoformat()
                        }

                        evaluated_results.append(result)

                        if is_high_quality:
                            total_passed += 1
                        else:
                            total_failed += 1

                except Exception as e:
                    logger.error(f"Error in batch evaluation: {e}")
                    # 批量推理失败，将整批标记为不合格
                    for qa_item in batch_metadata:
                        evaluated_results.append({
                            **qa_item,
                            'quality_score': 0,
                            'evaluation_result': f'【否】批量评估时出错: {str(e)}',
                            'evaluation_time': datetime.now().isoformat()
                        })
                        total_failed += 1

        # 记录统计信息
        processing_time = time.time() - start_time
        logger.info(f"Question quality evaluation completed in {processing_time:.2f}s")
        logger.info(f"Total: {len(converted_data)}, Passed: {total_passed}, Failed: {total_failed}")

        return evaluated_results
#     def convert_questionlist_li_data(self, jsonl_file_questionlist: str, jsonl_file_li: str):
#         """
#         转换问题列表格式，将每个问题单独保存为一条记录
        
#         Args:
#             jsonl_file_questionlist: 输入文件路径（包含问题列表）
#             jsonl_file_li: 输出文件路径（每个问题一条记录）
#         """
#         logger.info(f"Converting question list from {jsonl_file_questionlist} to {jsonl_file_li}")
        
#         converted_count = 0
#         error_count = 0
        
#         with open(jsonl_file_questionlist, 'r', encoding='utf-8') as infile, \
#              open(jsonl_file_li, 'w', encoding='utf-8') as outfile:
            
#             for idx, line in enumerate(infile, 1):
#                 line = line.strip()
#                 if not line:
#                     continue
                
#                 try:
#                     # 解析原始数据
#                     original_data = json.loads(line)
                    
#                     # 提取需要的字段
#                     stats = original_data.get("stats", 0)
#                     paper_name = original_data.get("paper_name", "")
#                     paper_content = original_data.get("paper_content", "")
#                     score_text = original_data.get("score_text", "")
#                     question_list = original_data.get("question_list", [])
                    
#                     # 为每个问题创建单独的记录
#                     for question_li in question_list:
#                         processed_data = {
#                             "stats": stats,
#                             "paper_name": paper_name,
#                             "paper_content": paper_content,
#                             "score_text": score_text,
#                             "question_li": question_li,
#                             "source_line": idx,
#                             "timestamp": datetime.now().isoformat()
#                         }
                        
#                         # 写入新文件
#                         json.dump(processed_data, outfile, ensure_ascii=False)
#                         outfile.write('\n')
#                         converted_count += 1
                    
#                     logger.debug(f"Processed entry {idx} successfully, generated {len(question_list)} records")
                    
#                 except json.JSONDecodeError as e:
#                     logger.warning(f"Line {idx} JSON decode error: {e}")
#                     error_count += 1
#                 except Exception as e:
#                     logger.error(f"Error processing entry {idx}: {e}")
#                     error_count += 1
        
#         logger.info(f"Conversion completed! Converted {converted_count} questions, {error_count} errors")
    def convert_questionlist_li_data(self, question_data: list) -> list:
        """
        从问题数据列表转换为单独的问题记录列表
        
        Args:
            question_data: 包含分类问题的数据列表
            
        Returns:
            list: 转换后的单独问题列表
        """
        logger.info(f"Converting {len(question_data)} question data items to individual questions")
        
        # 添加调试信息
        if question_data:
            logger.info(f"第一个数据项类型: {type(question_data[0])}")
            logger.info(f"第一个数据项内容预览: {str(question_data[0])[:200]}...")
        
        converted_data = []
        converted_count = 0
        error_count = 0
        
        for item in question_data:
            try:
                # 处理字符串格式的数据（如果是JSON字符串）
                if isinstance(item, str):
                    try:
                        import json
                        item = json.loads(item)
                        logger.info(f"成功解析JSON字符串数据")
                    except json.JSONDecodeError as e:
                        logger.error(f"无法解析JSON字符串: {e}")
                        logger.error(f"字符串内容: {item[:100]}...")
                        error_count += 1
                        continue
                
                # 确保 item 是字典
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item: {type(item)} - {item}")
                    error_count += 1
                    continue
                    
                paper_name = item.get("paper_name", "")
                source_content = item.get("source_content", "")
                questions_dict = item.get("questions", {})
                source_info = item.get("source_info", {})
                
                logger.info(f"处理论文: {paper_name}")
                logger.info(f"问题字典类型: {type(questions_dict)}")
                logger.info(f"问题字典keys: {list(questions_dict.keys()) if isinstance(questions_dict, dict) else 'Not a dict'}")
                
                # 确保 questions_dict 是字典
                if not isinstance(questions_dict, dict):
                    logger.warning(f"Invalid questions format for {paper_name}: {type(questions_dict)}")
                    error_count += 1
                    continue
                    
                # 遍历所有问题类型
                for question_type, questions in questions_dict.items():
                    logger.info(f"处理问题类型: {question_type}, 问题列表类型: {type(questions)}")
                    
                    # 确保 questions 是列表
                    if not isinstance(questions, list):
                        logger.warning(f"Invalid question list for type {question_type} in {paper_name}: {type(questions)}")
                        error_count += 1
                        continue
                    
                    logger.info(f"问题列表长度: {len(questions)}")
                    if questions:
                        logger.info(f"第一个问题类型: {type(questions[0])}")
                        logger.info(f"第一个问题内容: {questions[0]}")
                        
                    for idx, question_data in enumerate(questions):
                        try:
                            logger.info(f"处理第{idx+1}个问题，类型: {type(question_data)}")
                            
                            # 根据实际数据格式，questions数组中的元素是字符串
                            if isinstance(question_data, str):
                                # 直接使用字符串作为问题文本
                                question_text = question_data.strip()
                            elif isinstance(question_data, dict):
                                # 如果是字典，尝试提取问题文本
                                question_text = question_data.get("question", "")
                                if not question_text:
                                    # 尝试其他可能的键
                                    question_text = question_data.get("text", "") or str(question_data)
                            else:
                                logger.warning(f"Unsupported question format in {paper_name}: {type(question_data)}")
                                error_count += 1
                                continue
                            
                            # 验证问题文本
                            if not question_text or question_text.strip() == "":
                                logger.warning(f"Empty question text in {paper_name}/{question_type}")
                                error_count += 1
                                continue
                                
                            # 创建转换后的记录
                            converted_item = {
                                "paper_name": paper_name,
                                "paper_content": source_content,
                                "question": question_text.strip(),
                                "question_type": question_type,
                                "source_info": source_info,
                                "stats": 1,
                                "timestamp": datetime.now().isoformat()
                            }
                            converted_data.append(converted_item)
                            converted_count += 1
                            
                            logger.info(f"成功转换问题: {question_text[:50]}...")
                            
                        except Exception as inner_e:
                            logger.error(f"Error processing question in {paper_name}/{question_type}: {inner_e}")
                            import traceback
                            logger.error(f"内部错误堆栈:\n{traceback.format_exc()}")
                            error_count += 1
                            
            except Exception as outer_e:
                paper_name = item.get("paper_name", "unknown") if isinstance(item, dict) else "unknown"
                logger.error(f"Error converting item {paper_name}: {outer_e}")
                error_count += 1
                import traceback
                logger.error(f"外部错误堆栈:\n{traceback.format_exc()}")
        
        logger.info(f"Conversion completed! Generated {converted_count} individual question records, with {error_count} errors")
        return converted_data

    async def run_full_pipeline(self, raw_folders: List[str], save_paths: List[str], 
                               output_dir: str = "/mnt/LLM/ldd/sft/rl_data/process_data") -> Dict[str, Any]:
        """
        运行完整的三步评估流程
        
        Args:
            raw_folders: 原始文件夹列表
            save_paths: 保存路径列表
            output_dir: 输出目录
            
        Returns:
            dict: 完整流程的统计结果
        """
        logger.info("Starting full semiconductor QA generation pipeline")
        start_time = time.time()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义中间文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        judge_output_path = os.path.join(output_dir, f"judge_txt_output_{timestamp}.jsonl")
        generate_output_path = os.path.join(output_dir, f"generate_question_output_{timestamp}.jsonl")
        generate_li_output_path = os.path.join(output_dir, f"generate_question_output_{timestamp}_converted.jsonl")
        
        # 步骤1：评估文本质量
        logger.info("Step 1: Judging text quality...")
        text_quality_stats = await self.judge_md_data(raw_folders, save_paths, judge_output_path)
        
        # 步骤2：生成问题
        logger.info("Step 2: Generating questions...")
        generation_stats = await self.generate_question_data(judge_output_path, generate_output_path)
        
        # 步骤3：转换问题格式
        logger.info("Step 3: Converting question format...")
        self.convert_questionlist_li_data(generate_output_path, generate_li_output_path)
        
        # 步骤4：评估问题质量
        logger.info("Step 4: Evaluating question quality...")
        evaluation_stats = await self.judge_question_data(generate_li_output_path, save_paths)
        
        # 汇总统计
        total_time = time.time() - start_time
        final_stats = {
            "pipeline_summary": {
                "total_processing_time": total_time,
                "total_files_processed": text_quality_stats["total_files"],
                "files_passed_text_quality": text_quality_stats["passed_files"],
                "total_questions_generated": generation_stats["total_questions"],
                "questions_passed_quality": evaluation_stats["passed_questions"],
                "success_rate": {
                    "text_quality_pass_rate": (
                        text_quality_stats["passed_files"] / text_quality_stats["total_files"] * 100
                        if text_quality_stats["total_files"] > 0 else 0
                    ),
                    "question_generation_success_rate": (
                        generation_stats["successful_generations"] / generation_stats["total_processed"] * 100
                        if generation_stats["total_processed"] > 0 else 0
                    ),
                    "question_quality_pass_rate": (
                        evaluation_stats["passed_questions"] / evaluation_stats["total_questions"] * 100
                        if evaluation_stats["total_questions"] > 0 else 0
                    )
                }
            },
            "text_quality_stats": text_quality_stats,
            "generation_stats": generation_stats,
            "evaluation_stats": evaluation_stats,
            "output_files": {
                "judge_output": judge_output_path,
                "generation_output": generate_output_path,
                "converted_output": generate_li_output_path,
                "final_results": save_paths
            }
        }
        
        # 保存汇总统计
        stats_path = os.path.join(output_dir, f"pipeline_stats_{timestamp}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        logger.info(f"Summary statistics saved to {stats_path}")
        
        return final_stats


# 便捷函数
async def run_semiconductor_qa_generation(
    raw_folders: list,
    save_paths: list,
    model_name: str = "qwq_32",
    batch_size: int = 4,
    gpu_devices: str = "0,1"
):
    """运行半导体QA生成 - 修复版本"""
    logger.info("初始化半导体QA生成器...")
    
    try:
        # 修复：正确创建生成器，不传递model_name参数
        generator = SemiconductorQAGenerator(
            batch_size=batch_size,
            gpu_devices=gpu_devices
        )
        
        # 通过属性设置模型名称
        generator.model_name = model_name
        logger.info(f"设置模型名称为: {model_name}")
        
        # 如果需要加载模型
        if hasattr(generator, 'load_model'):
            generator.load_model()
        
        logger.info("半导体QA生成器初始化完成")
        
        # 继续原有的处理逻辑...
        # 这里保持你原有的处理代码不变
        
    except Exception as e:
        logger.error(f"QA生成失败: {str(e)}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        raise
