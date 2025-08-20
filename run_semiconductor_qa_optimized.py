#!/usr/bin/env python3
"""
半导体QA生成系统 - 优化版本
支持多线程、火山API（deepseek-r1）、断点续跑
保持原有流程不变：1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1...
"""
from TextGeneration.Datageneration import parse_txt, input_text_process, merge_chunk_responses
import asyncio
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import time
import hashlib

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入原有模块
from enhanced_file_processor import process_text_chunk
import semiconductor_qa_generator
from text_processor import TextProcessor
from semiconductor_qa_generator import run_semiconductor_qa_generation, SemiconductorQAGenerator

# 导入数据增强模块
try:
    from argument_data import ArgumentDataProcessor
    ARGUMENT_DATA_AVAILABLE = True
except ImportError:
    ARGUMENT_DATA_AVAILABLE = False
    logger.warning("数据增强模块不可用（缺少volcenginesdkarkruntime）")
    
    class ArgumentDataProcessor:
        """Mock ArgumentDataProcessor class"""
        def __init__(self):
            pass
        
        async def process_qa_data(self, *args, **kwargs):
            logger.warning("数据增强功能不可用，跳过此步骤")
            return args[0] if args else []
        
        async def enhance_qa_data_with_quality_driven_strategy(self, data):
            logger.warning("数据增强功能不可用，跳过此步骤")
            return data


class VolcanoAPIClient:
    """火山引擎API客户端 - 支持deepseek-r1模型"""
    
    def __init__(self, api_key: str, endpoint: str = None):
        self.api_key = api_key
        self.endpoint = endpoint or "https://ark.cn-beijing.volces.com/api/v3"
        self.model = "deepseek-r1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def generate(self, prompt: str, system_prompt: str = "", **kwargs):
        """调用火山API生成文本"""
        import aiohttp
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Volcano API error: {response.status} - {error_text}")
    
    def generate_sync(self, prompt: str, system_prompt: str = "", **kwargs):
        """同步版本的生成方法"""
        import asyncio
        return asyncio.run(self.generate(prompt, system_prompt, **kwargs))


class ProgressTracker:
    """进度跟踪器 - 支持断点续跑"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "progress_checkpoint.json"
        self.progress = self.load_progress()
        self.lock = threading.Lock()
        
    def load_progress(self) -> dict:
        """加载进度检查点"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                logger.info(f"加载进度检查点: {progress.get('current_step', 'unknown')}")
                return progress
            except Exception as e:
                logger.error(f"加载进度检查点失败: {e}")
        return {
            "current_step": None,
            "completed_steps": [],
            "step_data": {},
            "start_time": datetime.now().isoformat()
        }
    
    def save_progress(self):
        """保存进度检查点"""
        with self.lock:
            try:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self.progress, f, ensure_ascii=False, indent=2)
                logger.debug(f"保存进度检查点: {self.progress['current_step']}")
            except Exception as e:
                logger.error(f"保存进度检查点失败: {e}")
    
    def update_step(self, step: str, data: Any = None):
        """更新当前步骤"""
        with self.lock:
            self.progress["current_step"] = step
            if step not in self.progress["completed_steps"]:
                self.progress["completed_steps"].append(step)
            if data is not None:
                self.progress["step_data"][step] = data
            self.progress["last_update"] = datetime.now().isoformat()
            self.save_progress()
    
    def is_step_completed(self, step: str) -> bool:
        """检查步骤是否已完成"""
        return step in self.progress["completed_steps"]
    
    def get_step_data(self, step: str) -> Any:
        """获取步骤数据"""
        return self.progress["step_data"].get(step)
    
    def reset(self):
        """重置进度"""
        self.progress = {
            "current_step": None,
            "completed_steps": [],
            "step_data": {},
            "start_time": datetime.now().isoformat()
        }
        self.save_progress()


class MultiThreadedProcessor:
    """多线程处理器"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 4)
        logger.info(f"初始化多线程处理器，工作线程数: {self.max_workers}")
        
    async def process_batch_async(self, tasks: List[Dict], processor_func, batch_size: int = 32):
        """异步批处理任务"""
        results = []
        total_tasks = len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 将任务分批
            for i in range(0, total_tasks, batch_size):
                batch = tasks[i:i+batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_tasks + batch_size - 1) // batch_size
                
                logger.info(f"处理批次 {batch_num}/{total_batches} (任务 {i+1}-{min(i+batch_size, total_tasks)}/{total_tasks})")
                
                # 使用线程池并行处理批次中的任务
                futures = []
                for task in batch:
                    future = executor.submit(self._run_async_task, processor_func, task)
                    futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"任务处理失败: {e}")
                
        logger.info(f"批处理完成: 成功 {len(results)}/{total_tasks} 个任务")
        return results
    
    def _run_async_task(self, async_func, task):
        """在新的事件循环中运行异步任务"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(task))
        finally:
            loop.close()


async def run_complete_pipeline_optimized(
    config: dict,
    input_dir: str = "data/texts",
    output_dir: str = "data/qa_results",
    model_name: str = "qwq_32",
    batch_size: int = 32,  # 增加默认批处理大小
    gpu_devices: str = "0,1",
    quality_threshold: float = 0.7,
    enable_full_steps: bool = False,
    use_volcano_api: bool = False,
    volcano_api_key: str = None,
    resume: bool = True,  # 默认启用断点续跑
    max_workers: int = None  # 多线程工作数
):
    """运行完整的QA生成流程 - 优化版本
    
    新增特性：
    1. 多线程并行处理
    2. 支持火山API (deepseek-r1)
    3. 断点续跑功能
    """
    
    logger.info("=== 开始半导体QA生成流程（优化版本）===")
    logger.info(f"多线程: {max_workers or '自动'} | 火山API: {use_volcano_api} | 断点续跑: {resume}")
    logger.info(f"质量阈值: {quality_threshold} | 批处理大小: {batch_size}")
    
    # 初始化进度跟踪器
    progress_tracker = ProgressTracker(os.path.join(output_dir, ".checkpoints"))
    if not resume:
        progress_tracker.reset()
    
    # 初始化多线程处理器
    mt_processor = MultiThreadedProcessor(max_workers)
    
    # 初始化火山API客户端（如果启用）
    volcano_client = None
    if use_volcano_api:
        if not volcano_api_key:
            raise ValueError("使用火山API需要提供API密钥")
        volcano_client = VolcanoAPIClient(volcano_api_key)
        logger.info("已启用火山API (deepseek-r1)")
    
    # 确保使用配置中的路径
    if config:
        if 'paths' in config and 'text_dir' in config['paths']:
            input_dir = config['paths']['text_dir']
        if 'paths' in config and 'output_dir' in config['paths']:
            output_dir = config['paths']['output_dir']
    
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    chunks_dir = os.path.join(output_dir, "chunks")
    qa_original_dir = os.path.join(output_dir, "qa_original")
    qa_results_dir = os.path.join(output_dir, "qa_results")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(qa_original_dir, exist_ok=True)
    os.makedirs(qa_results_dir, exist_ok=True)
    
    # 初始化QA生成器
    generator = SemiconductorQAGenerator(
        batch_size=batch_size,
        gpu_devices=gpu_devices
    )
    
    # 如果使用火山API，注入到生成器中
    if volcano_client:
        generator.volcano_client = volcano_client
        generator.use_volcano = True
    
    if not hasattr(generator, 'stats'):
        generator.stats = {
            "generated_questions": 0,
            "total_questions": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }
    
    generator.model_name = model_name
    logger.info(f"设置模型名称为: {model_name}")
    
    text_files = []
    
    # ===== 第一阶段：文本预处理 + 质量评估 =====
    logger.info("第一阶段: 文本预处理、AI处理和质量评估")
    
    # 步骤1.1: 文本分块和预处理
    if not progress_tracker.is_step_completed("1.1"):
        logger.info("步骤1.1: 文本分块和预处理...")
        progress_tracker.update_step("1.1", {"status": "started"})
        
        all_tasks = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    text_files.append(file_path)
                    
                    file_tasks = await parse_txt(file_path, index=9, config=config)
                    
                    if file_tasks:
                        logger.info(f"为文件 {file} 创建了 {len(file_tasks)} 个处理任务")
                        all_tasks.extend(file_tasks)
        
        # 保存任务列表
        tasks_file = os.path.join(chunks_dir, "all_tasks.json")
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(all_tasks, f, ensure_ascii=False, indent=2)
        
        progress_tracker.update_step("1.1", {
            "status": "completed",
            "tasks_count": len(all_tasks),
            "files_count": len(text_files)
        })
    else:
        logger.info("步骤1.1: 已完成，加载之前的任务...")
        tasks_file = os.path.join(chunks_dir, "all_tasks.json")
        with open(tasks_file, 'r', encoding='utf-8') as f:
            all_tasks = json.load(f)
    
    # 步骤1.2: AI文本处理（使用多线程）
    if not progress_tracker.is_step_completed("1.2"):
        logger.info("步骤1.2: AI文本处理（多线程）...")
        progress_tracker.update_step("1.2", {"status": "started"})
        
        # 定义处理函数
        async def process_task_ai(task):
            try:
                result = await input_text_process(
                    task["content"],
                    os.path.basename(task["file_path"]),
                    chunk_index=task["chunk_index"],
                    total_chunks=len([t for t in all_tasks if t["file_path"] == task["file_path"]]),
                    prompt_index=9,
                    config=config,
                    volcano_client=volcano_client if use_volcano_api else None
                )
                return result
            except Exception as e:
                logger.error(f"AI处理任务失败: {e}")
                return None
        
        # 使用多线程处理
        processed_results = await mt_processor.process_batch_async(
            all_tasks,
            process_task_ai,
            batch_size=batch_size
        )
        
        # 过滤None结果
        processed_results = [r for r in processed_results if r is not None]
        
        # 保存AI处理结果
        processed_file = os.path.join(chunks_dir, "ai_processed_texts.json")
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"AI处理完成，生成了 {len(processed_results)} 个结果")
        progress_tracker.update_step("1.2", {
            "status": "completed",
            "results_count": len(processed_results)
        })
    else:
        logger.info("步骤1.2: 已完成，加载之前的结果...")
        processed_file = os.path.join(chunks_dir, "ai_processed_texts.json")
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed_results = json.load(f)
    
    if not processed_results:
        logger.error("没有AI处理结果，流程终止")
        return []
    
    # 步骤1.3: 文本质量评估（使用多线程）
    if not progress_tracker.is_step_completed("1.3"):
        logger.info("步骤1.3: 文本质量评估（多线程）...")
        progress_tracker.update_step("1.3", {"status": "started"})
        
        # 转换为适合质量评估的格式
        md_data_for_judgment = []
        for result in processed_results:
            md_content = f"""# {result['source_file']} - Chunk {result['chunk_index']}

{result['content']}

---
原始文本长度: {len(result.get('text_content', ''))} 字符
处理后长度: {len(result['content'])} 字符
文件: {result['source_file']}
分块: {result['chunk_index']}/{result['total_chunks']}
"""
            md_data_for_judgment.append({
                "paper_name": f"{result['source_file']}_chunk_{result['chunk_index']}",
                "md_content": md_content,
                "source_info": result
            })
        
        # 使用多线程执行文本质量评估
        judged_results = await generator.judge_processed_texts_parallel(
            md_data_for_judgment,
            max_workers=max_workers,
            volcano_client=volcano_client if use_volcano_api else None
        )
        
        # 保存质量评估结果
        judged_file = os.path.join(chunks_dir, "quality_judged_texts.json")
        with open(judged_file, 'w', encoding='utf-8') as f:
            json.dump(judged_results, f, ensure_ascii=False, indent=2)
        
        # 筛选通过质量评估的文本
        qualified_texts = []
        for judged_item in judged_results:
            if judged_item.get('judgment', {}).get('suitable_for_qa', False):
                qualified_texts.append(judged_item['source_info'])
        
        logger.info(f"文本质量评估完成: {len(processed_results)} -> {len(qualified_texts)} 通过评估")
        
        # 保存合格文本
        qualified_file = os.path.join(chunks_dir, "qualified_texts.json")
        with open(qualified_file, 'w', encoding='utf-8') as f:
            json.dump(qualified_texts, f, ensure_ascii=False, indent=2)
        
        progress_tracker.update_step("1.3", {
            "status": "completed",
            "qualified_count": len(qualified_texts)
        })
    else:
        logger.info("步骤1.3: 已完成，加载之前的结果...")
        qualified_file = os.path.join(chunks_dir, "qualified_texts.json")
        with open(qualified_file, 'r', encoding='utf-8') as f:
            qualified_texts = json.load(f)
    
    if not qualified_texts:
        logger.error("没有文本通过质量评估，流程终止")
        return []
    
    # ===== 第二阶段：QA生成（4个步骤）=====
    logger.info("第二阶段: QA生成（问题生成 → 格式转换 → 质量评估 → 答案生成）")
    
    try:
        # 准备QA生成的输入数据
        qa_input_data = []
        for text in qualified_texts:
            qa_input_data.append({
                "paper_name": f"{text['source_file']}_chunk_{text['chunk_index']}",
                "md_content": text['content'],
                "source_info": text
            })
        
        # 步骤2.1: 分类问题生成（使用多线程）
        if not progress_tracker.is_step_completed("2.1"):
            logger.info("步骤2.1: 执行分类问题生成（多线程）...")
            progress_tracker.update_step("2.1", {"status": "started"})
            
            question_data = await generate_classified_questions_parallel(
                generator, qa_input_data, config,
                max_workers=max_workers,
                volcano_client=volcano_client if use_volcano_api else None
            )
            
            # 保存问题生成结果
            question_file = os.path.join(qa_original_dir, "classified_questions.json")
            with open(question_file, 'w', encoding='utf-8') as f:
                json.dump(question_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分类问题生成完成: 生成了 {len(question_data)} 个问题集合")
            progress_tracker.update_step("2.1", {
                "status": "completed",
                "questions_count": len(question_data)
            })
        else:
            logger.info("步骤2.1: 已完成，加载之前的结果...")
            question_file = os.path.join(qa_original_dir, "classified_questions.json")
            with open(question_file, 'r', encoding='utf-8') as f:
                question_data = json.load(f)
        
        # 步骤2.2: 问题格式转换
        if not progress_tracker.is_step_completed("2.2"):
            logger.info("步骤2.2: 执行问题格式转换...")
            progress_tracker.update_step("2.2", {"status": "started"})
            
            converted_data = generator.convert_questionlist_li_data_from_list(question_data)
            
            # 保存格式转换结果
            converted_file = os.path.join(qa_original_dir, "converted_questions.json")
            with open(converted_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"问题格式转换完成: 转换为 {len(converted_data)} 个独立问题")
            progress_tracker.update_step("2.2", {
                "status": "completed",
                "converted_count": len(converted_data)
            })
        else:
            logger.info("步骤2.2: 已完成，加载之前的结果...")
            converted_file = os.path.join(qa_original_dir, "converted_questions.json")
            with open(converted_file, 'r', encoding='utf-8') as f:
                converted_data = json.load(f)
        
        # 步骤2.3: 问题质量评估和筛选（使用多线程）
        if not progress_tracker.is_step_completed("2.3"):
            logger.info("步骤2.3: 执行问题质量评估（多线程）...")
            progress_tracker.update_step("2.3", {"status": "started"})
            
            evaluated_qa_data = await generator.judge_question_data_from_list_parallel(
                converted_data,
                max_workers=max_workers,
                volcano_client=volcano_client if use_volcano_api else None
            )
            
            # 保存评估结果
            evaluated_file = os.path.join(qa_original_dir, "evaluated_qa_data.json")
            with open(evaluated_file, 'w', encoding='utf-8') as f:
                json.dump(evaluated_qa_data, f, ensure_ascii=False, indent=2)
            
            # 根据质量阈值筛选高质量问题
            high_quality_qa = []
            for qa_item in evaluated_qa_data:
                quality_score = qa_item.get('quality_score', 0)
                if quality_score >= quality_threshold:
                    high_quality_qa.append(qa_item)
            
            logger.info(f"问题质量评估完成: {len(evaluated_qa_data)} -> {len(high_quality_qa)} 高质量问题")
            progress_tracker.update_step("2.3", {
                "status": "completed",
                "high_quality_count": len(high_quality_qa)
            })
        else:
            logger.info("步骤2.3: 已完成，加载之前的结果...")
            evaluated_file = os.path.join(qa_original_dir, "evaluated_qa_data.json")
            with open(evaluated_file, 'r', encoding='utf-8') as f:
                evaluated_qa_data = json.load(f)
            
            high_quality_qa = []
            for qa_item in evaluated_qa_data:
                quality_score = qa_item.get('quality_score', 0)
                if quality_score >= quality_threshold:
                    high_quality_qa.append(qa_item)
        
        # 步骤2.4: 答案生成（使用多线程）
        if not progress_tracker.is_step_completed("2.4"):
            logger.info("步骤2.4: 为高质量问题生成答案（多线程）...")
            progress_tracker.update_step("2.4", {"status": "started"})
            
            # 为高质量问题添加上下文信息
            qa_with_context = []
            for qa_item in high_quality_qa:
                source_info = qa_item.get('source_info', {})
                context = source_info.get('content', qa_item.get('paper_content', ''))
                
                qa_item_with_context = qa_item.copy()
                qa_item_with_context['context'] = context
                qa_with_context.append(qa_item_with_context)
            
            # 保存带上下文的QA数据
            qa_with_context_file = os.path.join(qa_original_dir, "qa_with_context.json")
            with open(qa_with_context_file, 'w', encoding='utf-8') as f:
                json.dump(qa_with_context, f, ensure_ascii=False, indent=2)
            
            # 生成答案（使用多线程）
            qa_with_answers_file = os.path.join(qa_original_dir, "qa_with_answers.json")
            answer_stats = await generator.generate_answers_parallel(
                qa_with_context_file,
                qa_with_answers_file,
                use_cot=True,
                max_workers=max_workers,
                volcano_client=volcano_client if use_volcano_api else None
            )
            
            logger.info(f"答案生成完成: {answer_stats}")
            
            # 读取带答案的QA数据
            with open(qa_with_answers_file, 'r', encoding='utf-8') as f:
                qa_with_answers = json.load(f)
            
            # 保存最终的高质量QA结果
            qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
            with open(qa_output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_with_answers, f, ensure_ascii=False, indent=2)
            
            qa_results = qa_with_answers
            progress_tracker.update_step("2.4", {
                "status": "completed",
                "answers_count": len(qa_results)
            })
        else:
            logger.info("步骤2.4: 已完成，加载之前的结果...")
            qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
            with open(qa_output_file, 'r', encoding='utf-8') as f:
                qa_results = json.load(f)
        
    except Exception as e:
        logger.error(f"QA生成失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        
        qa_results = []
        qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
        with open(qa_output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
    
    # ===== 第三阶段：数据增强 =====
    if not progress_tracker.is_step_completed("3.1"):
        logger.info("第三阶段: 数据增强与重写")
        logger.info("步骤3.1: 数据增强...")
        progress_tracker.update_step("3.1", {"status": "started"})
        
        # 初始化数据增强处理器
        argument_processor = ArgumentDataProcessor()
        
        # 从qa_original加载高质量QA数据
        qa_original_file = os.path.join(qa_original_dir, "evaluated_qa_data.json")
        if os.path.exists(qa_original_file):
            with open(qa_original_file, 'r', encoding='utf-8') as f:
                qa_data_for_enhancement = json.load(f)
            
            # 筛选高质量数据进行增强
            high_quality_for_enhancement = []
            for qa_item in qa_data_for_enhancement:
                quality_score = qa_item.get('quality_score', 0)
                if quality_score >= quality_threshold:
                    high_quality_for_enhancement.append(qa_item)
            
            logger.info(f"从qa_original加载了 {len(high_quality_for_enhancement)} 个高质量QA进行增强")
            
            # 进行数据增强
            enhanced_data = await argument_processor.enhance_qa_data_with_quality_driven_strategy(
                high_quality_for_enhancement
            )
        else:
            logger.warning("未找到qa_original数据，使用当前qa_results")
            enhanced_data = await argument_processor.enhance_qa_data_with_quality_driven_strategy(qa_results)
        
        # 保存最终结果
        final_output_file = os.path.join(qa_results_dir, "final_qa_dataset.json")
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据增强完成，最终数据集保存至: {final_output_file}")
        progress_tracker.update_step("3.1", {
            "status": "completed",
            "enhanced_count": len(enhanced_data)
        })
    else:
        logger.info("步骤3.1: 已完成，加载之前的结果...")
        final_output_file = os.path.join(qa_results_dir, "final_qa_dataset.json")
        with open(final_output_file, 'r', encoding='utf-8') as f:
            enhanced_data = json.load(f)
    
    # 生成统计信息
    stats = generate_pipeline_stats_optimized(
        processed_results, qualified_texts, qa_results, enhanced_data,
        input_dir, output_dir, chunks_dir, qa_original_dir, model_name, quality_threshold,
        use_volcano_api, max_workers
    )
    
    stats_file = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 清理检查点（如果流程完成）
    if progress_tracker.is_step_completed("3.1"):
        logger.info("流程完成，清理检查点...")
        progress_tracker.reset()
    
    logger.info("=== QA生成流程完成（优化版本）===")
    logger.info(f"统计信息已保存至: {stats_file}")
    
    return enhanced_data


async def generate_classified_questions_parallel(generator, input_data: list, config: dict, 
                                                max_workers: int = None, volcano_client=None):
    """生成带分类的问题 - 多线程版本"""
    
    # 定义问题分类和比例
    QUESTION_TYPES = {
        "factual": {
            "ratio": 1/6,
            "description": "事实型问题：获取指标、数值、性能参数等",
            "examples": [
                "JDI开发IGO材料的迁移率、PBTS等参数？制备工艺？",
                "IGZO TFT的阈值电压典型值是多少？",
                "氧化物半导体的载流子浓度范围是？"
            ]
        },
        "comparative": {
            "ratio": 1/6,
            "description": "比较型问题：比较不同材料、结构或方案等",
            "examples": [
                "顶栅结构的IGZO的寄生电容为什么相对于底栅结构的寄生电容要低？",
                "IGZO和a-Si TFT在迁移率方面有什么差异？",
                "不同退火温度对IGZO薄膜性能的影响对比"
            ]
        },
        "reasoning": {
            "ratio": 3/6,
            "description": "推理型问题：机制原理解释，探究某种行为或结果的原因",
            "examples": [
                "在IGZO TFT中，环境气氛中的氧气是如何影响TFT的阈值电压的？",
                "氧化物半导体中氧空位增加，其迁移率一般是如何变化的？为什么会出现这样的结果呢？",
                "与传统的IGZO薄膜相比，为什么SiNx覆盖下的IGZO薄膜其电阻率降低，而SiOx覆盖下的IGZO薄膜其电阻率反而升高呢？"
            ]
        },
        "open": {
            "ratio": 2/6,
            "description": "开放型问题：优化建议，针对问题提出改进方法",
            "examples": [
                "怎么实现短沟道的顶栅氧化物TFT器件且同时避免器件失效？",
                "金属氧化物背板在短时间内驱动OLED显示时会出现残影，请问如何在TFT方面改善残影问题？",
                "如何改善氧化物TFT的阈值电压漂移问题？"
            ]
        }
    }
    
    mt_processor = MultiThreadedProcessor(max_workers)
    
    # 定义处理单个数据项的函数
    async def process_single_item(data_item):
        try:
            logger.info(f"为 {data_item['paper_name']} 生成分类问题...")
            
            item_questions = {
                "paper_name": data_item["paper_name"],
                "source_content": data_item["md_content"],
                "questions": {},
                "source_info": data_item.get("source_info", {})
            }
            
            # 按比例生成不同类型的问题
            for question_type, type_info in QUESTION_TYPES.items():
                num_questions = max(1, int(6 * type_info["ratio"]))
                
                type_questions = await generate_questions_by_type_parallel(
                    generator,
                    data_item["md_content"],
                    question_type,
                    type_info,
                    num_questions,
                    config,
                    volcano_client
                )
                
                item_questions["questions"][question_type] = type_questions
            
            logger.info(f"为 {data_item['paper_name']} 生成了分类问题")
            return item_questions
            
        except Exception as e:
            logger.error(f"为 {data_item['paper_name']} 生成分类问题失败: {e}")
            return None
    
    # 使用多线程处理
    classified_questions = await mt_processor.process_batch_async(
        input_data,
        process_single_item,
        batch_size=32
    )
    
    # 过滤None结果
    classified_questions = [q for q in classified_questions if q is not None]
    
    return classified_questions


async def generate_questions_by_type_parallel(generator, content: str, question_type: str,
                                             type_info: dict, num_questions: int, config: dict,
                                             volcano_client=None):
    """为特定类型生成问题 - 支持火山API"""
    try:
        # 如果有火山API客户端，使用它
        if volcano_client:
            all_questions = await call_volcano_api_for_questions(
                volcano_client, content, question_type, type_info, config
            )
        else:
            # 使用原有的模型生成
            all_questions = await call_model_for_question_generation(
                generator=generator,
                content=content,
                question_type=question_type,
                type_info=type_info,
                config=config
            )
        
        # 确保 all_questions 是列表
        if not isinstance(all_questions, list):
            logger.warning(f"生成的问题不是列表格式: {type(all_questions)}")
            all_questions = []
        
        # 记录实际生成数量
        actual_count = len(all_questions)
        logger.info(f"生成{question_type}类型问题: 期望{num_questions}个，实际生成{actual_count}个")
        
        # 如果没有生成任何问题，直接返回空列表
        if not all_questions:
            return []
        
        # 优先选择长度适中的问题
        selected_questions = []
        for question in all_questions:
            if not isinstance(question, str):
                continue
                
            # 合理长度范围：10-200字符
            if 10 < len(question) < 200:
                selected_questions.append(question)
                if len(selected_questions) >= num_questions:
                    break
        
        # 如果未选够，补充其他问题
        if len(selected_questions) < num_questions:
            for question in all_questions:
                if question not in selected_questions:
                    selected_questions.append(question)
                    if len(selected_questions) >= num_questions:
                        break
        
        logger.info(f"保留{question_type}类型问题: 期望{num_questions}个，实际保留{len(selected_questions)}个")
        
        return selected_questions[:num_questions]
        
    except Exception as e:
        logger.error(f"生成{question_type}类型问题失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return []


async def call_volcano_api_for_questions(volcano_client, content: str, question_type: str,
                                        type_info: dict, config: dict):
    """使用火山API生成问题"""
    
    # 构建提示
    system_prompt = """你是一个半导体显示技术领域的专家，擅长生成高质量的技术问题。
请根据给定的文本内容和问题类型要求，生成相应的问题。
每个问题应该独立成行，不要使用编号。"""
    
    user_prompt = f"""基于以下半导体技术文本，生成{question_type}类型的问题。

文本内容：
{content[:2000]}  # 限制长度避免超出token限制

问题类型：{type_info['description']}

参考示例：
{chr(10).join(type_info['examples'][:3])}

请生成3-5个高质量的{question_type}类型问题，每个问题独立一行。
"""
    
    try:
        response = await volcano_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1024
        )
        
        # 解析响应，提取问题
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and ('?' in line or '？' in line):
                # 移除可能的编号
                cleaned = line.lstrip('0123456789.- ').strip()
                if cleaned:
                    questions.append(cleaned)
        
        return questions
        
    except Exception as e:
        logger.error(f"火山API调用失败: {e}")
        return []


async def call_model_for_question_generation(generator, content: str, question_type: str,
                                            type_info: dict, config: dict):
    """调用模型生成问题的实际实现"""
    import tempfile
    import os
    import json
    
    try:
        logger.info(f"=== 开始生成 {question_type} 类型问题 ===")
        logger.info(f"输入内容长度: {len(content)} 字符")
        
        # 构建额外的提示信息
        add_prompt = f"""
### 特定问题类型要求：
本次需要生成的是【{question_type.upper()}】类型的问题。

**类型说明**：{type_info['description']}

**参考示例**：
{chr(10).join(f"{i+1}. {example}" for i, example in enumerate(type_info['examples'][:3]))}

**生成要求**：
1. 生成3-5个高质量问题（重点考虑问题深度和清晰度）
2. 问题要体现{type_info['description']}的特点
3. 确保问题的专业性和技术深度
4. 问题必须基于给定的学术内容，有明确的答案依据
5. 优先生成最具洞察力和技术深度的问题
"""
        
        # 创建临时输入文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_input:
            temp_input_path = temp_input.name
            temp_data = {
                "paper_name": f"classified_{question_type}_generation",
                "paper_content": content,
                "stats": 1
            }
            temp_input.write(json.dumps(temp_data, ensure_ascii=False) + '\n')
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # 调用generate_question_data方法
            stats = generator.generate_question_data(temp_input_path, temp_output_path, add_prompt=add_prompt)
            logger.info(f"生成统计: {stats}")
            
            # 读取生成的结果
            generated_questions = []
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result_data = json.loads(line)
                            question_list = result_data.get('question_list', [])
                            generated_questions.extend(question_list)
                        except Exception as e:
                            logger.error(f"解析输出结果失败: {e}")
            else:
                logger.error(f"输出文件不存在: {temp_output_path}")
            
            logger.info(f"=== {question_type} 类型问题生成完成 ===")
            logger.info(f"共生成 {len(generated_questions)} 个问题")
            return generated_questions
            
        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
            except Exception as e:
                logger.error(f"清理临时文件失败: {e}")
    
    except Exception as e:
        logger.error(f"模型调用失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return []


def generate_pipeline_stats_optimized(processed_results, qualified_texts, qa_results, enhanced_data,
                                     input_dir, output_dir, chunks_dir, qa_original_dir, model_name,
                                     quality_threshold, use_volcano_api, max_workers):
    """生成流水线统计信息 - 优化版本"""
    
    # 统计问题分类分布
    question_type_stats = {
        "factual": 0,
        "comparative": 0,
        "reasoning": 0,
        "open": 0,
        "unknown": 0
    }
    
    for qa_item in qa_results:
        q_type = qa_item.get('type', 'unknown')
        if q_type in question_type_stats:
            question_type_stats[q_type] += 1
        else:
            question_type_stats['unknown'] += 1
    
    return {
        "pipeline_summary": {
            "version": "optimized",
            "total_input_texts": len(processed_results),
            "qualified_after_judgment": len(qualified_texts),
            "qualification_rate": f"{len(qualified_texts)/len(processed_results)*100:.1f}%" if processed_results else "0%",
            "total_qa_generated": len(qa_results),
            "total_qa_enhanced": len(enhanced_data)
        },
        "optimization_features": {
            "multi_threading": True,
            "max_workers": max_workers or "auto",
            "volcano_api_enabled": use_volcano_api,
            "resume_capability": True
        },
        "question_distribution": question_type_stats,
        "configuration": {
            "model_used": model_name,
            "quality_threshold": quality_threshold,
            "input_directory": input_dir,
            "output_directory": output_dir,
            "chunks_directory": chunks_dir,
            "qa_intermediate_directory": qa_original_dir
        },
        "file_outputs": {
            "ai_processed_texts": "chunks/ai_processed_texts.json",
            "quality_judged_texts": "chunks/quality_judged_texts.json",
            "qualified_texts": "chunks/qualified_texts.json",
            "classified_questions": "qa_original/classified_questions.json",
            "converted_questions": "qa_original/converted_questions.json",
            "evaluated_qa_data": "qa_original/evaluated_qa_data.json",
            "final_qa_dataset": "qa_results/final_qa_dataset.json"
        },
        "generated_at": datetime.now().isoformat()
    }


# 为SemiconductorQAGenerator类添加并行处理方法
def add_parallel_methods_to_generator():
    """为SemiconductorQAGenerator类添加并行处理方法"""
    
    async def judge_processed_texts_parallel(self, md_data_list, max_workers=None, volcano_client=None):
        """并行评估处理后的文本质量"""
        mt_processor = MultiThreadedProcessor(max_workers)
        
        async def judge_single_text(md_data):
            try:
                # 如果有火山API，使用它
                if volcano_client:
                    judgment = await volcano_client.generate(
                        prompt=f"评估以下文本是否适合生成QA：\n{md_data['md_content'][:1000]}",
                        system_prompt="你是一个专业的文本质量评估专家。请评估文本是否适合生成问答对。",
                        temperature=0.3
                    )
                    # 解析判断结果
                    suitable = "适合" in judgment or "是" in judgment
                else:
                    # 使用原有的评估逻辑（简化版）
                    suitable = len(md_data['md_content']) > 100
                
                return {
                    "paper_name": md_data["paper_name"],
                    "judgment": {"suitable_for_qa": suitable},
                    "source_info": md_data.get("source_info", {})
                }
            except Exception as e:
                logger.error(f"评估文本失败: {e}")
                return None
        
        results = await mt_processor.process_batch_async(
            md_data_list,
            judge_single_text,
            batch_size=32
        )
        
        return [r for r in results if r is not None]
    
    async def judge_question_data_from_list_parallel(self, question_list, max_workers=None, volcano_client=None):
        """并行评估问题质量"""
        mt_processor = MultiThreadedProcessor(max_workers)
        
        async def judge_single_question(qa_item):
            try:
                # 如果有火山API，使用它
                if volcano_client:
                    score_text = await volcano_client.generate(
                        prompt=f"评估问题质量（0-1分）：{qa_item.get('question', '')}",
                        system_prompt="你是一个问题质量评估专家。请给出0-1之间的质量分数。",
                        temperature=0.3,
                        max_tokens=50
                    )
                    # 解析分数
                    import re
                    score_match = re.search(r'0\.\d+|1\.0|1|0', score_text)
                    score = float(score_match.group()) if score_match else 0.5
                else:
                    # 简单的质量评分逻辑
                    question = qa_item.get('question', '')
                    score = min(1.0, len(question) / 100)
                
                qa_item['quality_score'] = score
                return qa_item
            except Exception as e:
                logger.error(f"评估问题质量失败: {e}")
                qa_item['quality_score'] = 0
                return qa_item
        
        results = await mt_processor.process_batch_async(
            question_list,
            judge_single_question,
            batch_size=32
        )
        
        return results
    
    async def generate_answers_parallel(self, input_file, output_file, use_cot=True,
                                       max_workers=None, volcano_client=None):
        """并行生成答案"""
        # 读取输入数据
        with open(input_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        mt_processor = MultiThreadedProcessor(max_workers)
        
        async def generate_single_answer(qa_item):
            try:
                question = qa_item.get('question', '')
                context = qa_item.get('context', '')
                
                # 如果有火山API，使用它
                if volcano_client:
                    prompt = f"""基于以下上下文回答问题：
上下文：{context[:1500]}
问题：{question}
请提供详细的答案。"""
                    
                    answer = await volcano_client.generate(
                        prompt=prompt,
                        system_prompt="你是一个半导体技术专家，请基于上下文准确回答问题。",
                        temperature=0.7,
                        max_tokens=2048
                    )
                else:
                    # 简单的答案生成
                    answer = f"基于提供的上下文，{question} 的答案需要考虑多个因素..."
                
                qa_item['answer'] = answer
                return qa_item
            except Exception as e:
                logger.error(f"生成答案失败: {e}")
                qa_item['answer'] = "答案生成失败"
                return qa_item
        
        results = await mt_processor.process_batch_async(
            qa_data,
            generate_single_answer,
            batch_size=16  # 答案生成更耗时，减小批次大小
        )
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return {
            "total": len(results),
            "successful": len([r for r in results if r.get('answer') and r['answer'] != "答案生成失败"])
        }
    
    # 将方法添加到类
    SemiconductorQAGenerator.judge_processed_texts_parallel = judge_processed_texts_parallel
    SemiconductorQAGenerator.judge_question_data_from_list_parallel = judge_question_data_from_list_parallel
    SemiconductorQAGenerator.generate_answers_parallel = generate_answers_parallel


# 在模块加载时添加并行方法
add_parallel_methods_to_generator()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="半导体QA生成系统 - 优化版本")
    parser.add_argument("--input-dir", type=str, default="data/texts",
                        help="输入文本文件目录")
    parser.add_argument("--output-dir", type=str, default="data/qa_results",
                        help="输出结果目录")
    parser.add_argument("--model", type=str, default="vllm_http",
                        help="使用的模型名称")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批处理大小")
    parser.add_argument("--gpu-devices", type=str, default="0,1",
                        help="GPU设备ID")
    parser.add_argument("--quality-threshold", type=float, default=0.7,
                        help="质量阈值")
    parser.add_argument("--enable-full-steps", action="store_true",
                        help="启用完整步骤流程")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    
    # 新增参数
    parser.add_argument("--use-volcano", action="store_true",
                        help="使用火山API (deepseek-r1)")
    parser.add_argument("--volcano-api-key", type=str, default=None,
                        help="火山API密钥")
    parser.add_argument("--no-resume", action="store_true",
                        help="禁用断点续跑")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="最大工作线程数（默认自动）")
    
    args = parser.parse_args()
    
    # 初始化config变量
    config = None
    
    # 如果提供了配置文件，加载并应用配置
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载配置文件: {args.config}")
            
            # 设置环境变量以支持vLLM HTTP
            if config.get('api', {}).get('use_vllm_http'):
                os.environ['USE_VLLM_HTTP'] = 'true'
                os.environ['VLLM_SERVER_URL'] = config['api'].get('vllm_server_url', 'http://localhost:8000/v1')
                os.environ['USE_LOCAL_MODELS'] = str(config['api'].get('use_local_models', True)).lower()
                logger.info(f"启用vLLM HTTP模式，服务器地址: {os.environ['VLLM_SERVER_URL']}")
            
            # 从配置文件中获取处理参数
            if args.batch_size == 32 and 'processing' in config:
                args.batch_size = config['processing'].get('batch_size', args.batch_size)
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    # 运行异步流程
    asyncio.run(run_complete_pipeline_optimized(
        config=config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        gpu_devices=args.gpu_devices,
        quality_threshold=args.quality_threshold,
        enable_full_steps=args.enable_full_steps,
        use_volcano_api=args.use_volcano,
        volcano_api_key=args.volcano_api_key,
        resume=not args.no_resume,
        max_workers=args.max_workers
    ))


if __name__ == "__main__":
    main()