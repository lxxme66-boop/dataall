import argparse
import os
import re
import asyncio
import json
from datetime import datetime
# Optional imports with fallbacks
try:
    from volcenginesdkarkruntime import Ark, AsyncArk
    VOLC_AVAILABLE = True
except ImportError:
    VOLC_AVAILABLE = False
    print("Warning: volcenginesdkarkruntime not available")

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available")

# Import local model support
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from LocalModels.local_model_manager import LocalModelManager
    LOCAL_MODEL_SUPPORT = True
except ImportError:
    LOCAL_MODEL_SUPPORT = False
    print("Warning: Local model support not available")

from .prompts_conf import system_prompt, user_prompts

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        use_local_models = config.get('api', {}).get('use_local_models', False)
        local_model_config = config.get('models', {}).get('local_models', {}).get('ollama', {})
else:
    use_local_models = False
    local_model_config = {}

# API configuration
ark_url = "http://0.0.0.0:8080/v1"
api_key = "fake-key"
model = "/mnt/data/LLM/lhy/models/Qwen3/Qwen3-235B-A22B"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("TextGeneration/logs", exist_ok=True)
current_time = os.path.getmtime(__file__) if os.path.exists(__file__) else 0
file_handler = logging.FileHandler(f"TextGeneration/logs/text_generation_{current_time}.log")


# def extract_text_chunks(text_content, chunk_size=2000, overlap=200):
#     """
#     Extract text chunks from the content for processing.
#     """
#     chunks = []
#     text_length = len(text_content)
    
#     if text_length <= chunk_size:
#         return [text_content]
    
#     start = 0
#     while start < text_length:
#         end = min(start + chunk_size, text_length)
        
#         # Try to find a natural break point (sentence end)
#         if end < text_length:
#             for sep in ['\n\n', '。\n', '。', '.\n', '.', '\n']:
#                 last_sep = text_content[start:end].rfind(sep)
#                 if last_sep != -1:
#                     end = start + last_sep + len(sep)
#                     break
        
#         chunk = text_content[start:end]
#         chunks.append(chunk)
        
#         # Move start position with overlap
#         start = end - overlap if end < text_length else end
    
#     return chunks

def extract_text_chunks(text_content, chunk_size=2000, overlap=200):
    """
    高效文本分块，支持大文件处理
    """
    # 处理空文本
    if not text_content:
        return []
    
    # 处理小文本
    text_length = len(text_content)
    if text_length <= chunk_size:
        return [text_content]
    
    chunks = []
    start = 0
    
    # 添加进度计数器
    total_chunks = (text_length - overlap) // (chunk_size - overlap) + 1
    processed_chunks = 0
    
    while start < text_length:
        # 计算当前分块结束位置
        end = min(start + chunk_size, text_length)
        
        # 查找最佳分句点（优先段落，再句子）
        break_positions = [
            text_content.rfind('\n\n', start, end),  # 段落分隔
            text_content.rfind('。', start, end),     # 中文句号
            text_content.rfind('.\n', start, end),    # 英文句号+换行
            text_content.rfind('\n', start, end)      # 普通换行
        ]
        
        # 选择最接近末尾的有效分句点
        valid_breaks = [pos for pos in break_positions if pos > start]
        if valid_breaks:
            end = max(valid_breaks) + 1  # 包含分句符号
        
        # 截取分块
        chunk = text_content[start:end]
        chunks.append(chunk)
        
        # 更新进度
        processed_chunks += 1
        if processed_chunks % 10 == 0:  # 每10个分块记录一次
            logger.info(f"已处理分块 {processed_chunks}/{total_chunks} ({processed_chunks/total_chunks:.1%})")
        
        # 移动起始位置（确保不倒退）
        next_start = end - overlap
        if next_start <= start:  # 防止死循环
            next_start = end
        
        start = max(next_start, end - overlap)  # 确保向前移动
    
    logger.info(f"共生成 {len(chunks)} 个文本分块")
#     return chunks
# async def parse_txt(file_path, index=9, config=None):
#     """
#     Parse txt file and create tasks for processing.
#     Now supports passing config for local model support.
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#     except Exception as e:
#         logger.error(f"Error reading file {file_path}: {e}")
#         return []
    
#     if not content.strip():
#         logger.warning(f"Empty file: {file_path}")
#         return []
    
#     # Extract text chunks
#     chunks = extract_text_chunks(content)
#     logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
    
#     tasks = []
#     for i, chunk in enumerate(chunks):
#         task = input_text_process(
#             chunk, 
#             os.path.basename(file_path),
#             chunk_index=i,
#             total_chunks=len(chunks),
#             prompt_index=index,
#             config=config
#         )
#         tasks.append(task)
    
#     return tasks
async def parse_txt(file_path: str, index: int, config: dict):
    """解析文本文件并生成处理任务"""
    logger.info(f"开始处理文件: {file_path}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 文本分块处理 (使用配置中的分块大小)
    #chunk_size = config.get("processing", {}).get("chunk_size", 1000)
    if config:
        chunk_size = config.get("processing", {}).get("chunk_size", 1000)
    else:
        chunk_size = 1000  # 默认值
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    # 创建任务列表
    tasks = []
    for idx, chunk in enumerate(chunks):
        task = {
            "file_path": file_path,
            "chunk_index": idx,
            "content": chunk,
            "timestamp": datetime.now().isoformat()
        }
        tasks.append(task)
    
    logger.info(f"为文件 {os.path.basename(file_path)} 创建 {len(tasks)} 个处理任务")
    return tasks

# async def input_text_process(text_content, source_file, chunk_index=0, total_chunks=1, prompt_index=9, config=None):
#     """
#     Process text content using the specified prompt.
#     Supports both API and local model backends.
#     """
#     # Check if we should use local models
#     use_local = False
#     use_vllm_http = False
#     local_model_manager = None
    
#     if config and LOCAL_MODEL_SUPPORT:
#         use_local = config.get('api', {}).get('use_local_models', False)
#         use_vllm_http = config.get('api', {}).get('use_vllm_http', False)
        
#         if use_local or use_vllm_http:
#             try:
#                 # 如果使用vLLM HTTP，确保配置正确
#                 if use_vllm_http:
#                     local_config = config.get('models', {}).get('local_models', {})
#                     local_config['default_backend'] = 'vllm_http'
#                     config['models']['local_models'] = local_config
                
#                 local_model_manager = LocalModelManager(config)
#                 if not local_model_manager.is_available():
#                     logger.warning("Local models enabled but not available, falling back to API")
#                     use_local = False
#             except Exception as e:
#                 logger.error(f"Failed to initialize local model manager: {e}")
#                 use_local = False
    
#     try:
#         user_prompt = user_prompts[prompt_index]
        
#         # Format the prompt with the text content
#         # Check if the prompt expects markdown_content or text_content
#         if '{markdown_content}' in user_prompt:
#             formatted_prompt = user_prompt.format(
#                 markdown_content=text_content,
#                 source_file=source_file,
#                 chunk_info=f"(Chunk {chunk_index + 1}/{total_chunks})" if total_chunks > 1 else ""
#             )
#         elif '{text_content}' in user_prompt:
#             formatted_prompt = user_prompt.format(
#                 text_content=text_content,
#                 source_file=source_file,
#                 chunk_info=f"(Chunk {chunk_index + 1}/{total_chunks})" if total_chunks > 1 else ""
#             )
#         else:
#             formatted_prompt = user_prompt
        
#         # Generate response using appropriate backend
#         if use_local and local_model_manager:
#             # Use local model
#             logger.info(f"Using local model backend: {local_model_manager.get_backend_name()}")
#             content = await local_model_manager.generate(
#                 prompt=formatted_prompt,
#                 system_prompt=system_prompt,
#                 temperature=config.get('models', {}).get('qa_generator_model', {}).get('temperature', 0.8),
#                 max_tokens=config.get('models', {}).get('qa_generator_model', {}).get('max_tokens', 4096),
#                 top_p=config.get('models', {}).get('qa_generator_model', {}).get('top_p', 0.9)
#             )
#         else:
#             # Use API backend (original code)
#             if not config:
#                 # Use default API configuration
#                 ark_url = "http://0.0.0.0:8080/v1"
#                 api_key = "fake-key"
#                 model = "/mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain"
#             else:
#                 # Use configuration from config
#                 api_config = config.get('api', {})
#                 ark_url = api_config.get('ark_url', "http://0.0.0.0:8080/v1")
#                 api_key = api_config.get('api_key', "fake-key")
#                 model = config.get('models', {}).get('default_model', "/mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain")
            
#             client = AsyncOpenAI(
#                 api_key=api_key,
#                 base_url=ark_url
#             )
            
#             response = await client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": system_prompt
#                     },
#                     {
#                         "role": "user", 
#                         "content": formatted_prompt
#                     }
#                 ],
#                 temperature=0.8,
#                 max_tokens=4096,
#                 top_p=0.9,
#             )
            
#             content = response.choices[0].message.content
        
#         # Structure the response
#         result = {
#             "content": content,
#             "source_file": source_file,
#             "chunk_index": chunk_index,
#             "total_chunks": total_chunks,
#             "text_content": text_content[:500] + "..." if len(text_content) > 500 else text_content
#         }
        
#         logger.info(f"Successfully processed chunk {chunk_index + 1}/{total_chunks} from {source_file}")
#         return result
        
#     except Exception as e:
#         logger.error(f"Error processing text from {source_file}: {e}")
#         return None

async def input_text_process(text_content, source_file, chunk_index=0, total_chunks=1, prompt_index=9, config=None):
    """
    Process text content using the specified prompt.
    Supports both API and local model backends.
    """
    # Check if we should use local models
    use_local = False
    use_vllm_http = False
    local_model_manager = None
    
    # 从配置中获取必要的API设置
    api_base = "http://localhost:8000/v1"
    api_key = "EMPTY"  # 默认值
    model_name = "qwen-vllm"
    timeout_value = 120.0
    # 添加分块标识
    logger.info(f"开始处理分块 {chunk_index+1}/{total_chunks} - 大小: {len(text_content)}字符")
    
    # # 在API请求前添加日志
    # logger.info(f"准备API请求: {formatted_prompt[:100]}...")
    
    # # 在请求后添加日志
    # logger.info(f"收到响应: {content[:100]}...")
    if config:
        api_config = config.get('api', {})
        # 从配置获取 API 设置
        api_base = api_config.get('vllm_server_url', "http://localhost:8000/v1")
        api_key = api_config.get('api_key', "EMPTY")  # 使用配置中的值
        use_local = api_config.get('use_local_models', False)
        use_vllm_http = api_config.get('use_vllm_http', False)
        
        models_config = config.get('models', {})
        model_name = models_config.get('qa_generator_model', {}).get('name', "qwen-vllm")
        
        # 获取超时设置，优先从 vllm_http 配置中获取
        vllm_http_config = config.get('models', {}).get('local_models', {}).get('vllm_http', {})
        timeout_value = vllm_http_config.get('timeout', 120.0)
        
        if use_local or use_vllm_http:
            try:
                # 如果使用vLLM HTTP，确保配置正确
                if use_vllm_http:
                    local_config = config.get('models', {}).get('local_models', {})
                    local_config['default_backend'] = 'vllm_http'
                    config['models']['local_models'] = local_config
                
                local_model_manager = LocalModelManager(config)
                if not local_model_manager.is_available():
                    logger.warning("Local models enabled but not available, falling back to API")
                    use_local = False
            except Exception as e:
                logger.error(f"Failed to initialize local model manager: {e}")
                use_local = False
                local_model_manager = None
    
    try:
        user_prompt = user_prompts[prompt_index]
        
        # Format the prompt with the text content
        if '{markdown_content}' in user_prompt:
            formatted_prompt = user_prompt.format(
                markdown_content=text_content,
                source_file=source_file,
                chunk_info=f"(Chunk {chunk_index + 1}/{total_chunks})" if total_chunks > 1 else ""
            )
        elif '{text_content}' in user_prompt:
            formatted_prompt = user_prompt.format(
                text_content=text_content,
                source_file=source_file,
                chunk_info=f"(Chunk {chunk_index + 1}/{total_chunks})" if total_chunks > 1 else ""
            )
        else:
            formatted_prompt = user_prompt
        
        # 记录使用的后端
        backend = "Local Model" if use_local and local_model_manager else "API"
        logger.info(f"📡 Using backend: {backend} for {source_file} chunk {chunk_index + 1}/{total_chunks}")
        
        # Generate response using appropriate backend
        if use_local and local_model_manager:
            # Use local model
            logger.info(f"🔧 Using local model backend: {local_model_manager.get_backend_name()}")
            content = await local_model_manager.generate(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=config.get('models', {}).get('qa_generator_model', {}).get('temperature', 0.8),
                max_tokens=config.get('models', {}).get('qa_generator_model', {}).get('max_tokens', 4096),
                top_p=config.get('models', {}).get('qa_generator_model', {}).get('top_p', 0.9)
            )
        else:
            # 使用 API 后端
            logger.info(f"🔌 Connecting to: {api_base}")
            logger.info(f"🔑 API key: {api_key[:3]}...")  # 显示部分 key
            logger.info(f"🤖 Model: {model_name}")
            logger.info(f"⏱️ Timeout: {timeout_value} seconds")
            
            # 创建异步客户端
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout_value  # 客户端级超时
            )
            
            try:
                # 发送请求
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user", 
                            "content": formatted_prompt
                        }
                    ],
                    temperature=config.get('models', {}).get('qa_generator_model', {}).get('temperature', 0.8),
                    max_tokens=config.get('models', {}).get('qa_generator_model', {}).get('max_tokens', 4096),
                    top_p=config.get('models', {}).get('qa_generator_model', {}).get('top_p', 0.9),
                    timeout=timeout_value  # 请求级超时
                )
                
                content = response.choices[0].message.content
                logger.info(f"✅ API request successful, response length: {len(content)} characters")
                
            except Exception as api_error:
                # 详细错误处理
                error_msg = str(api_error)
                logger.error(f"❌ API request failed: {error_msg}")
                
                # 根据错误类型提供建议
                if "connect" in error_msg.lower():
                    logger.error("💡 Connection failed, please check service status: "
                                 "1. Ensure vLLM service is running\n"
                                 "2. Check network connection\n"
                                 "3. Verify API address")
                elif "timeout" in error_msg.lower():
                    logger.error(f"💡 Request timeout, try increasing timeout (current: {timeout_value} seconds)")
                elif "authentication" in error_msg.lower():
                    logger.error("💡 Authentication failed, please check API key")
                elif "model not found" in error_msg.lower():
                    logger.error("💡 Model not found, please check model name")
                else:
                    logger.error("💡 Unknown error, please check logs")
                
                # 返回None表示失败
                return None
        
        # Structure the response
        result = {
            "content": content,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "text_content": text_content[:500] + "..." if len(text_content) > 500 else text_content
        }
        
        logger.info(f"✅ Successfully processed chunk {chunk_index + 1}/{total_chunks} from {source_file}")
        return result
        
    except Exception as e:
        # 处理所有其他异常
        logger.error(f"❌ Unknown error processing text ({source_file}): {e}")
        logger.exception("Detailed error:")  # 记录堆栈跟踪
        return None
def merge_chunk_responses(responses):
    """
    Merge responses from multiple chunks of the same file.
    """
    if not responses:
        return []
    
    # Group by source file
    file_groups = {}
    for resp in responses:
        if resp and 'source_file' in resp:
            source = resp['source_file']
            if source not in file_groups:
                file_groups[source] = []
            file_groups[source].append(resp)
    
    # Merge chunks for each file
    merged_responses = []
    for source_file, chunks in file_groups.items():
        # Sort by chunk index
        chunks.sort(key=lambda x: x.get('chunk_index', 0))
        
        # Combine content
        combined_content = []
        all_text_content = []
        
        for chunk in chunks:
            combined_content.append(chunk['content'])
            all_text_content.append(chunk.get('text_content', ''))
        
        merged_response = {
            "content": "\n\n".join(combined_content),
            "source_file": source_file,
            "full_text": "\n".join(all_text_content)
        }
        
        merged_responses.append(merged_response)
    
    return merged_responses


async def process_folder_async(folder_path, prompt_index=9, max_concurrent=5, config=None):
    """
    异步处理文件夹中的所有文本文件
    """
    tasks = []
    
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                # 使用parse_txt处理文件
                file_tasks = await parse_txt(file_path, prompt_index, config)
                tasks.extend(file_tasks)
    
    # 限制并发数量
    results = []
    for i in range(0, len(tasks), max_concurrent):
        batch = tasks[i:i + max_concurrent]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        # 过滤掉异常结果
        for result in batch_results:
            if not isinstance(result, Exception) and result is not None:
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed with error: {result}")
    
    return results


async def process_folder_async_with_history(folder_path, history_file=None, prompt_index=9, max_concurrent=5, config=None):
    """
    异步处理文件夹中的文本文件，支持历史记录
    """
    processed_files = set()
    
    # 读取历史记录
    if history_file and os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                for item in history_data:
                    if 'source_file' in item:
                        processed_files.add(item['source_file'])
            logger.info(f"Loaded {len(processed_files)} processed files from history")
        except Exception as e:
            logger.error(f"Error loading history file: {e}")
    
    tasks = []
    
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt') and file not in processed_files:
                file_path = os.path.join(root, file)
                # 使用parse_txt处理文件
                file_tasks = await parse_txt(file_path, prompt_index, config)
                tasks.extend(file_tasks)
    
    logger.info(f"Found {len(tasks)} new tasks to process")
    
    # 限制并发数量
    results = []
    for i in range(0, len(tasks), max_concurrent):
        batch = tasks[i:i + max_concurrent]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        # 过滤掉异常结果
        for result in batch_results:
            if not isinstance(result, Exception) and result is not None:
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed with error: {result}")
    
    return results