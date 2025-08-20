#!/usr/bin/env python3
"""
半导体QA生成系统 - 模块化版本
每个步骤都可以独立运行的模块化流水线

支持的步骤：
1.1 文本预处理
1.2 AI文本处理  
1.3 文本质量评估
2.1 分类问题生成
2.2 问题格式转换
2.3 问题质量评估
2.4 答案生成
3.0 数据增强
"""

import asyncio
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入模块
try:
    from TextGeneration.Datageneration import parse_txt, input_text_process
    from enhanced_file_processor import process_text_chunk
    from semiconductor_qa_generator import SemiconductorQAGenerator
    from text_processor import TextProcessor
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"核心模块导入失败: {e}")
    CORE_MODULES_AVAILABLE = False

# 可选模块
try:
    from argument_data import ArgumentDataProcessor
    ARGUMENT_DATA_AVAILABLE = True
except ImportError:
    ARGUMENT_DATA_AVAILABLE = False
    logger.warning("数据增强模块不可用")
    
    class ArgumentDataProcessor:
        def __init__(self):
            pass
        async def enhance_qa_data(self, *args, **kwargs):
            logger.warning("数据增强功能不可用，跳过此步骤")
            return args[0] if args else []
        async def enhance_qa_data_with_quality_driven_strategy(self, *args, **kwargs):
            logger.warning("数据增强功能不可用，跳过此步骤")
            return args[0] if args else []

# 火山API支持
try:
    from run_semiconductor_qa_optimized import VolcanoAPIClient
    VOLCANO_API_AVAILABLE = True
except ImportError:
    VOLCANO_API_AVAILABLE = False
    logger.warning("火山API模块不可用")
    
    class VolcanoAPIClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
        
        async def generate(self, prompt, temperature=0.7, max_tokens=4096):
            logger.warning("火山API不可用，使用本地模型")
            return None


class ModularQAPipeline:
    """模块化的QA生成流水线"""
    
    def __init__(self, config: dict = None, use_volcano_api: bool = False, 
                 volcano_api_key: str = None, max_workers: int = None):
        self.config = config or {}
        self.output_dir = None
        self.chunks_dir = None
        self.qa_original_dir = None
        self.qa_results_dir = None
        
        # QA生成器（延迟初始化）
        self.generator = None
        
        # 火山API客户端
        self.use_volcano_api = use_volcano_api
        self.volcano_client = None
        if use_volcano_api and VOLCANO_API_AVAILABLE:
            if not volcano_api_key:
                volcano_api_key = os.environ.get('VOLCANO_API_KEY')
            if volcano_api_key:
                self.volcano_client = VolcanoAPIClient(volcano_api_key)
                logger.info("火山API客户端已初始化")
            else:
                logger.warning("未提供火山API密钥，将使用本地模型")
                self.use_volcano_api = False
        
        # 多线程配置
        self.max_workers = max_workers or os.cpu_count() or 4
        self.thread_pool = None
        
        # 步骤状态跟踪（增强版，包含更多信息）
        self.step_status = {
            "1.1": {"completed": False, "output": None, "start_time": None, "end_time": None, "progress": 0},
            "1.2": {"completed": False, "output": None, "start_time": None, "end_time": None, "progress": 0},
            "1.3": {"completed": False, "output": None, "start_time": None, "end_time": None, "progress": 0},
            "2.1": {"completed": False, "output": None, "start_time": None, "end_time": None, "progress": 0},
            "2.2": {"completed": False, "output": None, "start_time": None, "end_time": None, "progress": 0},
            "2.3": {"completed": False, "output": None, "start_time": None, "end_time": None, "progress": 0},
            "2.4": {"completed": False, "output": None, "start_time": None, "end_time": None, "progress": 0},
            "3.0": {"completed": False, "output": None, "start_time": None, "end_time": None, "progress": 0}
        }
        
        # 断点恢复信息
        self.checkpoint_file = None
        self.resume_enabled = True
    
    def setup_directories(self, output_dir: str):
        """设置输出目录结构"""
        self.output_dir = output_dir
        self.chunks_dir = os.path.join(output_dir, "chunks")
        self.qa_original_dir = os.path.join(output_dir, "qa_original")
        self.qa_results_dir = os.path.join(output_dir, "qa_results")
        
        for dir_path in [self.output_dir, self.chunks_dir, self.qa_original_dir, self.qa_results_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # 加载已有的步骤状态
        status_file = os.path.join(self.output_dir, "step_status.json")
        if os.path.exists(status_file):
            with open(status_file, 'r', encoding='utf-8') as f:
                self.step_status = json.load(f)
        
        logger.info(f"输出目录结构已创建: {output_dir}")
    
    def save_step_status(self, step_id: str = None):
        """保存步骤状态（增强版，支持更详细的断点信息）"""
        status_file = os.path.join(self.output_dir, "step_status.json")
        
        # 如果指定了步骤ID，更新该步骤的时间戳
        if step_id and step_id in self.step_status:
            if self.step_status[step_id]["completed"]:
                self.step_status[step_id]["end_time"] = datetime.now().isoformat()
            elif self.step_status[step_id]["start_time"] is None:
                self.step_status[step_id]["start_time"] = datetime.now().isoformat()
        
        # 保存完整状态信息
        full_status = {
            "step_status": self.step_status,
            "config": self.config,
            "use_volcano_api": self.use_volcano_api,
            "max_workers": self.max_workers,
            "last_update": datetime.now().isoformat(),
            "output_dir": self.output_dir
        }
        
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(full_status, f, ensure_ascii=False, indent=2)
        
        # 同时保存checkpoint文件用于快速恢复
        if self.checkpoint_file:
            checkpoint_file = os.path.join(self.output_dir, "checkpoint.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "last_completed_step": self._get_last_completed_step(),
                    "timestamp": datetime.now().isoformat(),
                    "can_resume": True
                }, f, ensure_ascii=False, indent=2)
    
    def _get_last_completed_step(self):
        """获取最后完成的步骤"""
        completed_steps = []
        for step_id, status in self.step_status.items():
            if status["completed"]:
                completed_steps.append(step_id)
        return completed_steps[-1] if completed_steps else None
    
    def load_checkpoint(self, output_dir: str):
        """从断点恢复"""
        status_file = os.path.join(output_dir, "step_status.json")
        if os.path.exists(status_file):
            with open(status_file, 'r', encoding='utf-8') as f:
                full_status = json.load(f)
            
            self.step_status = full_status.get("step_status", self.step_status)
            self.config = full_status.get("config", self.config)
            self.use_volcano_api = full_status.get("use_volcano_api", False)
            self.max_workers = full_status.get("max_workers", self.max_workers)
            
            logger.info(f"从断点恢复，最后更新时间: {full_status.get('last_update')}")
            
            # 显示已完成的步骤
            completed_steps = [step for step, status in self.step_status.items() if status["completed"]]
            if completed_steps:
                logger.info(f"已完成的步骤: {', '.join(completed_steps)}")
            
            return True
        return False
    
    def get_generator(self, model_name: str = "qwq_32", batch_size: int = 2, gpu_devices: str = "0,1"):
        """获取或初始化QA生成器"""
        if self.generator is None:
            logger.info(f"初始化QA生成器: model={model_name}, batch_size={batch_size}")
            self.generator = SemiconductorQAGenerator(
                batch_size=batch_size,
                gpu_devices=gpu_devices
            )
            self.generator.model_name = model_name
            
            # 确保统计信息初始化
            if not hasattr(self.generator, 'stats'):
                self.generator.stats = {
                    "generated_questions": 0,
                    "total_questions": 0,
                    "successful": 0,
                    "failed": 0,
                    "skipped": 0,
                }
        
        return self.generator
    
    async def step_1_1_text_preprocessing(self, input_dir: str):
        """步骤1.1: 文本预处理和分块"""
        logger.info("=== 执行步骤1.1: 文本预处理和分块 ===")
        
        if not CORE_MODULES_AVAILABLE:
            raise RuntimeError("核心模块不可用，无法执行文本预处理")
        
        text_files = []
        all_tasks = []
        
        # 遍历输入目录，创建处理任务
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    text_files.append(file_path)
                    
                    file_tasks = await parse_txt(file_path, index=9, config=self.config)
                    
                    if file_tasks:
                        logger.info(f"为文件 {file} 创建了 {len(file_tasks)} 个处理任务")
                        all_tasks.extend(file_tasks)
        
        # 保存任务信息
        tasks_file = os.path.join(self.chunks_dir, "preprocessing_tasks.json")
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(all_tasks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文本预处理完成: 共创建 {len(all_tasks)} 个处理任务")
        logger.info(f"任务信息保存至: {tasks_file}")
        
        # 更新步骤状态
        self.step_status["1.1"]["completed"] = True
        self.step_status["1.1"]["output"] = tasks_file
        self.save_step_status()
        
        return all_tasks, text_files
    
    async def step_1_2_ai_processing(self, input_tasks=None, batch_size: int = 2, use_multithread: bool = True):
        """步骤1.2: AI文本处理（支持多线程和火山API）"""
        logger.info("=== 执行步骤1.2: AI文本处理 ===")
        logger.info(f"配置: 批大小={batch_size}, 多线程={use_multithread}, 火山API={self.use_volcano_api}")
        
        # 记录开始时间
        self.step_status["1.2"]["start_time"] = datetime.now().isoformat()
        self.save_step_status("1.2")
        
        # 获取输入数据
        if input_tasks is None:
            # 从步骤1.1的输出加载
            if not self.step_status["1.1"]["completed"]:
                raise RuntimeError("步骤1.1未完成，请先运行步骤1.1")
            
            with open(self.step_status["1.1"]["output"], 'r', encoding='utf-8') as f:
                input_tasks = json.load(f)
        
        if not input_tasks:
            logger.error("没有可处理的任务")
            return []
        
        # 检查是否有未完成的任务（断点恢复）
        checkpoint_file = os.path.join(self.chunks_dir, "ai_processing_checkpoint.json")
        completed_indices = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                completed_indices = set(checkpoint_data.get("completed_indices", []))
                logger.info(f"从断点恢复: 已完成 {len(completed_indices)}/{len(input_tasks)} 个任务")
        
        # 过滤未完成的任务
        pending_tasks = [(i, task) for i, task in enumerate(input_tasks) if i not in completed_indices]
        
        if not pending_tasks:
            logger.info("所有任务已完成")
            # 加载已有结果
            output_file = os.path.join(self.chunks_dir, "ai_processed_texts.json")
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    processed_results = json.load(f)
                return processed_results
            return []
        
        # 定义处理函数（支持火山API）
        async def process_single_task(task_data):
            task_index, task = task_data
            try:
                # 如果使用火山API
                if self.use_volcano_api and self.volcano_client:
                    prompt = f"""请处理以下文本内容：
                    
文件: {os.path.basename(task['file_path'])}
分块: {task['chunk_index']}
内容: {task['content']}

请提取关键信息并整理为结构化的格式。"""
                    
                    result = await self.volcano_client.generate(
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=4096
                    )
                    
                    if result:
                        return {
                            "content": result,
                            "source_file": os.path.basename(task["file_path"]),
                            "chunk_index": task["chunk_index"],
                            "total_chunks": len([t for t in input_tasks if t[1]["file_path"] == task["file_path"]]),
                            "text_content": task["content"],
                            "task_index": task_index
                        }
                
                # 使用本地模型
                result = await input_text_process(
                    task["content"], 
                    os.path.basename(task["file_path"]),
                    chunk_index=task["chunk_index"],
                    total_chunks=len([t for t in input_tasks if t[1]["file_path"] == task["file_path"] if isinstance(t, tuple) else t["file_path"] == task["file_path"]]),
                    prompt_index=9,
                    config=self.config
                )
                if result:
                    result["task_index"] = task_index
                return result
                
            except Exception as e:
                logger.error(f"处理任务 {task_index} 失败: {e}")
                return None
        
        # 执行处理
        processed_results = []
        total_batches = (len(pending_tasks) + batch_size - 1) // batch_size
        
        if use_multithread and len(pending_tasks) > 1:
            # 多线程处理
            logger.info(f"使用多线程处理 (线程数: {self.max_workers})")
            
            for i in range(0, len(pending_tasks), batch_size):
                batch = pending_tasks[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"处理批次 {batch_num}/{total_batches}...")
                self.step_status["1.2"]["progress"] = int((i / len(pending_tasks)) * 100)
                self.save_step_status("1.2")
                
                # 使用asyncio并发处理批次
                batch_results = await asyncio.gather(
                    *(process_single_task(task_data) for task_data in batch),
                    return_exceptions=True
                )
                
                # 收集成功的结果
                for result in batch_results:
                    if result and not isinstance(result, Exception):
                        processed_results.append(result)
                        # 更新断点信息
                        if "task_index" in result:
                            completed_indices.add(result["task_index"])
                    elif isinstance(result, Exception):
                        logger.error(f"任务处理异常: {result}")
                
                # 保存断点
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({"completed_indices": list(completed_indices)}, f)
        else:
            # 单线程处理
            logger.info("使用单线程顺序处理")
            
            for i, task_data in enumerate(pending_tasks):
                if i % 10 == 0:
                    logger.info(f"处理进度: {i+1}/{len(pending_tasks)}")
                    self.step_status["1.2"]["progress"] = int((i / len(pending_tasks)) * 100)
                    self.save_step_status("1.2")
                
                result = await process_single_task(task_data)
                if result:
                    processed_results.append(result)
                    if "task_index" in result:
                        completed_indices.add(result["task_index"])
                    
                    # 定期保存断点
                    if i % 5 == 0:
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump({"completed_indices": list(completed_indices)}, f)
        
        # 加载之前已处理的结果（如果有）
        output_file = os.path.join(self.chunks_dir, "ai_processed_texts.json")
        if os.path.exists(output_file) and len(completed_indices) > len(pending_tasks):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                # 合并结果
                existing_indices = {r.get("task_index", -1) for r in existing_results}
                for result in processed_results:
                    if result.get("task_index", -1) not in existing_indices:
                        existing_results.append(result)
                processed_results = existing_results
        
        # 清理task_index字段
        for result in processed_results:
            if "task_index" in result:
                del result["task_index"]
        
        # 保存最终结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)
        
        # 清理断点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        logger.info(f"AI处理完成: 生成了 {len(processed_results)} 个结果")
        logger.info(f"结果保存至: {output_file}")
        
        # 更新步骤状态
        self.step_status["1.2"]["completed"] = True
        self.step_status["1.2"]["output"] = output_file
        self.step_status["1.2"]["progress"] = 100
        self.save_step_status("1.2")
        
        return processed_results
    
    async def step_1_3_text_quality_judgment(self, input_results=None, model_name="qwq_32", batch_size=2, gpu_devices="0,1"):
        """步骤1.3: 文本质量评估"""
        logger.info("=== 执行步骤1.3: 文本质量评估 ===")
        
        # 获取输入数据
        if input_results is None:
            if not self.step_status["1.2"]["completed"]:
                raise RuntimeError("步骤1.2未完成，请先运行步骤1.2")
            
            with open(self.step_status["1.2"]["output"], 'r', encoding='utf-8') as f:
                input_results = json.load(f)
        
        if not input_results:
            logger.error("没有可评估的文本")
            return []
        
        # 获取QA生成器
        generator = self.get_generator(model_name, batch_size, gpu_devices)
        
        # 转换为适合质量评估的格式
        md_data_for_judgment = []
        for result in input_results:
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
        
        # 执行质量评估
        judged_results = await generator.judge_processed_texts(md_data_for_judgment)
        
        # 保存评估结果
        judged_file = os.path.join(self.chunks_dir, "quality_judged_texts.json")
        with open(judged_file, 'w', encoding='utf-8') as f:
            json.dump(judged_results, f, ensure_ascii=False, indent=2)
        
        # 筛选合格文本
        qualified_texts = []
        for judged_item in judged_results:
            if judged_item.get('judgment', {}).get('suitable_for_qa', False):
                qualified_texts.append(judged_item['source_info'])
        
        qualified_file = os.path.join(self.chunks_dir, "qualified_texts.json")
        with open(qualified_file, 'w', encoding='utf-8') as f:
            json.dump(qualified_texts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文本质量评估完成: {len(input_results)} -> {len(qualified_texts)} 通过评估")
        
        # 更新步骤状态
        self.step_status["1.3"]["completed"] = True
        self.step_status["1.3"]["output"] = qualified_file
        self.save_step_status()
        
        return qualified_texts
    
    async def step_2_1_classified_question_generation(self, input_texts=None, model_name="qwq_32", batch_size=2, gpu_devices="0,1"):
        """步骤2.1: 分类问题生成"""
        logger.info("=== 执行步骤2.1: 分类问题生成 ===")
        
        # 获取输入数据
        if input_texts is None:
            if not self.step_status["1.3"]["completed"]:
                raise RuntimeError("步骤1.3未完成，请先运行步骤1.3")
            
            with open(self.step_status["1.3"]["output"], 'r', encoding='utf-8') as f:
                input_texts = json.load(f)
        
        if not input_texts:
            logger.error("没有合格的文本用于问题生成")
            return []
        
        # 获取QA生成器
        generator = self.get_generator(model_name, batch_size, gpu_devices)
        
        # 准备QA生成的输入数据
        qa_input_data = []
        for text in input_texts:
            qa_input_data.append({
                "paper_name": f"{text['source_file']}_chunk_{text['chunk_index']}",
                "md_content": text['content'],
                "source_info": text
            })
        
        # 生成分类问题
        question_data = await self.generate_classified_questions(generator, qa_input_data)
        
        # 保存结果
        output_file = os.path.join(self.qa_original_dir, "classified_questions.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分类问题生成完成: 生成了 {len(question_data)} 个问题集合")
        logger.info(f"结果保存至: {output_file}")
        
        # 更新步骤状态
        self.step_status["2.1"]["completed"] = True
        self.step_status["2.1"]["output"] = output_file
        self.save_step_status()
        
        return question_data
    
    async def step_2_2_question_format_conversion(self, input_questions=None, model_name="qwq_32", batch_size=2, gpu_devices="0,1"):
        """步骤2.2: 问题格式转换"""
        logger.info("=== 执行步骤2.2: 问题格式转换 ===")
        
        # 获取输入数据
        if input_questions is None:
            if not self.step_status["2.1"]["completed"]:
                raise RuntimeError("步骤2.1未完成，请先运行步骤2.1")
            
            with open(self.step_status["2.1"]["output"], 'r', encoding='utf-8') as f:
                input_questions = json.load(f)
        
        if not input_questions:
            logger.error("没有问题数据可供转换")
            return []
        
        # 获取QA生成器
        generator = self.get_generator(model_name, batch_size, gpu_devices)
        
        # 执行格式转换
        converted_data = generator.convert_questionlist_li_data_from_list(input_questions)
        
        # 保存结果
        output_file = os.path.join(self.qa_original_dir, "converted_questions.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"问题格式转换完成: 转换为 {len(converted_data)} 个独立问题")
        logger.info(f"结果保存至: {output_file}")
        
        # 更新步骤状态
        self.step_status["2.2"]["completed"] = True
        self.step_status["2.2"]["output"] = output_file
        self.save_step_status()
        
        return converted_data
    
    async def step_2_3_question_quality_evaluation(self, input_questions=None, quality_threshold=0.7, model_name="qwq_32", batch_size=2, gpu_devices="0,1"):
        """步骤2.3: 问题质量评估"""
        logger.info("=== 执行步骤2.3: 问题质量评估 ===")
        
        # 获取输入数据
        if input_questions is None:
            if not self.step_status["2.2"]["completed"]:
                raise RuntimeError("步骤2.2未完成，请先运行步骤2.2")
            
            with open(self.step_status["2.2"]["output"], 'r', encoding='utf-8') as f:
                input_questions = json.load(f)
        
        if not input_questions:
            logger.error("没有问题可供评估")
            return []
        
        # 获取QA生成器
        generator = self.get_generator(model_name, batch_size, gpu_devices)
        
        # 执行质量评估
        evaluated_qa_data = await generator.judge_question_data_from_list(input_questions)
        
        # 保存评估结果
        evaluated_file = os.path.join(self.qa_original_dir, "evaluated_qa_data.json")
        with open(evaluated_file, 'w', encoding='utf-8') as f:
            json.dump(evaluated_qa_data, f, ensure_ascii=False, indent=2)
        
        # 根据质量阈值筛选高质量问题
        high_quality_qa = []
        for qa_item in evaluated_qa_data:
            quality_score = qa_item.get('quality_score', 0)
            if quality_score >= quality_threshold:
                high_quality_qa.append(qa_item)
        
        # 保存高质量问题
        high_quality_file = os.path.join(self.qa_original_dir, "high_quality_questions.json")
        with open(high_quality_file, 'w', encoding='utf-8') as f:
            json.dump(high_quality_qa, f, ensure_ascii=False, indent=2)
        
        logger.info(f"问题质量评估完成: {len(evaluated_qa_data)} -> {len(high_quality_qa)} 高质量问题")
        logger.info(f"质量阈值: {quality_threshold}")
        
        # 更新步骤状态
        self.step_status["2.3"]["completed"] = True
        self.step_status["2.3"]["output"] = high_quality_file
        self.save_step_status()
        
        return high_quality_qa
    
    async def step_2_4_answer_generation(self, input_qa=None, model_name="qwq_32", batch_size=2, gpu_devices="0,1"):
        """步骤2.4: 答案生成"""
        logger.info("=== 执行步骤2.4: 答案生成 ===")
        
        # 获取输入数据
        if input_qa is None:
            if not self.step_status["2.3"]["completed"]:
                raise RuntimeError("步骤2.3未完成，请先运行步骤2.3")
            
            with open(self.step_status["2.3"]["output"], 'r', encoding='utf-8') as f:
                input_qa = json.load(f)
        
        if not input_qa:
            logger.error("没有高质量问题可供答案生成")
            return []
        
        # 获取QA生成器
        generator = self.get_generator(model_name, batch_size, gpu_devices)
        
        # 为问题添加上下文信息
        qa_with_context = []
        for qa_item in input_qa:
            source_info = qa_item.get('source_info', {})
            context = source_info.get('content', qa_item.get('paper_content', ''))
            
            qa_item_with_context = qa_item.copy()
            qa_item_with_context['context'] = context
            qa_with_context.append(qa_item_with_context)
        
        # 保存带上下文的QA数据
        qa_with_context_file = os.path.join(self.qa_original_dir, "qa_with_context.json")
        with open(qa_with_context_file, 'w', encoding='utf-8') as f:
            json.dump(qa_with_context, f, ensure_ascii=False, indent=2)
        
        # 生成答案
        qa_with_answers_file = os.path.join(self.qa_original_dir, "qa_with_answers.json")
        answer_stats = generator.generate_answers(
            qa_with_context_file,
            qa_with_answers_file,
            use_cot=True
        )
        
        # 读取带答案的QA数据
        with open(qa_with_answers_file, 'r', encoding='utf-8') as f:
            qa_with_answers = json.load(f)
        
        # 保存最终QA结果
        final_qa_file = os.path.join(self.qa_results_dir, "qa_generated.json")
        with open(final_qa_file, 'w', encoding='utf-8') as f:
            json.dump(qa_with_answers, f, ensure_ascii=False, indent=2)
        
        logger.info(f"答案生成完成: {answer_stats}")
        logger.info(f"最终QA结果保存至: {final_qa_file}")
        
        # 更新步骤状态
        self.step_status["2.4"]["completed"] = True
        self.step_status["2.4"]["output"] = final_qa_file
        self.save_step_status()
        
        return qa_with_answers
    
    async def step_3_0_data_enhancement(self, input_qa=None):
        """步骤3.0: 数据增强"""
        logger.info("=== 执行步骤3.0: 数据增强 ===")
        
        # 获取输入数据
        if input_qa is None:
            if not self.step_status["2.4"]["completed"]:
                raise RuntimeError("步骤2.4未完成，请先运行步骤2.4")
            
            with open(self.step_status["2.4"]["output"], 'r', encoding='utf-8') as f:
                input_qa = json.load(f)
        
        if not input_qa:
            logger.error("没有QA数据可供增强")
            return []
        
        # 初始化数据增强处理器
        argument_processor = ArgumentDataProcessor()
        
        # 执行数据增强
        enhanced_data = await argument_processor.enhance_qa_data(input_qa)
        
        # 保存增强结果
        output_file = os.path.join(self.qa_results_dir, "final_qa_dataset.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据增强完成: {len(input_qa)} -> {len(enhanced_data)} QA对")
        logger.info(f"最终数据集保存至: {output_file}")
        
        # 更新步骤状态
        self.step_status["3.0"]["completed"] = True
        self.step_status["3.0"]["output"] = output_file
        self.save_step_status()
        
        return enhanced_data
    
    async def generate_classified_questions(self, generator, input_data: list):
        """生成带分类的问题"""
        
        QUESTION_TYPES = {
            "factual": {
                "ratio": 1/3,
                "description": "事实型问题：获取指标、数值、性能参数等",
                "examples": [
                    "JDI开发IGO材料的迁移率、PBTS等参数？制备工艺？",
                    "IGZO TFT的阈值电压典型值是多少？",
                    "氧化物半导体的载流子浓度范围是？"
                ]
            },
            "comparative": {
                "ratio": 1/3,
                "description": "比较型问题：比较不同材料、结构或方案等",
                "examples": [
                    "顶栅结构的IGZO的寄生电容为什么相对于底栅结构的寄生电容要低？",
                    "IGZO和a-Si TFT在迁移率方面有什么差异？",
                    "不同退火温度对IGZO薄膜性能的影响对比"
                ]
            },
            "reasoning": {
                "ratio": 2/3,
                "description": "推理型问题：机制原理解释，探究某种行为或结果的原因",
                "examples": [
                    "在IGZO TFT中，环境气氛中的氧气是如何影响TFT的阈值电压的？",
                    "氧化物半导体中氧空位增加，其迁移率一般是如何变化的？为什么会出现这样的结果呢？",
                    "与传统的IGZO薄膜相比，为什么SiNx覆盖下的IGZO薄膜其电阻率降低，而SiOx覆盖下的IGZO薄膜其电阻率反而升高呢？"
                ]
            },
            "open": {
                "ratio": 1/3,
                "description": "开放型问题：优化建议，针对问题提出改进方法",
                "examples": [
                    "怎么实现短沟道的顶栅氧化物TFT器件且同时避免器件失效？",
                    "金属氧化物背板在短时间内驱动OLED显示时会出现残影，请问如何在TFT方面改善残影问题？",
                    "如何改善氧化物TFT的阈值电压漂移问题？"
                ]
            }
        }
        
        classified_questions = []
        
        for data_item in input_data:
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
                    num_questions = max(1, int(3 * type_info["ratio"]))
                    
                    type_questions = await self.generate_questions_by_type(
                        generator, 
                        data_item["md_content"], 
                        question_type,
                        type_info,
                        num_questions
                    )
                    
                    item_questions["questions"][question_type] = type_questions
                
                classified_questions.append(item_questions)
                logger.info(f"为 {data_item['paper_name']} 生成了分类问题")
                
            except Exception as e:
                logger.error(f"为 {data_item['paper_name']} 生成分类问题失败: {e}")
                continue
        
        return classified_questions
    
    async def generate_questions_by_type(self, generator, content: str, question_type: str, type_info: dict, num_questions: int):
        """为特定类型生成问题"""
        try:
            # 构建提示
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
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_input:
                temp_input_path = temp_input.name
                temp_data = {
                    "paper_name": f"classified_{question_type}_generation",
                    "paper_content": content,
                    "stats": 1
                }
                temp_input.write(json.dumps(temp_data, ensure_ascii=False) + '\n')
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # 调用生成方法
                stats = generator.generate_question_data(temp_input_path, temp_output_path, add_prompt=add_prompt)
                
                # 读取结果
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
                
                logger.info(f"生成{question_type}类型问题: 期望{num_questions}个，实际生成{len(generated_questions)}个")
                return generated_questions[:num_questions]
                
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
            logger.error(f"生成{question_type}类型问题失败: {e}")
            return []
    
    def show_status(self):
        """显示所有步骤的状态"""
        logger.info("=== 流水线步骤状态 ===")
        
        step_names = {
            "1.1": "文本预处理和分块",
            "1.2": "AI文本处理",
            "1.3": "文本质量评估",
            "2.1": "分类问题生成",
            "2.2": "问题格式转换",
            "2.3": "问题质量评估",
            "2.4": "答案生成",
            "3.0": "数据增强"
        }
        
        for step_id, step_name in step_names.items():
            status = self.step_status[step_id]
            status_text = "✓ 已完成" if status["completed"] else "✗ 未完成"
            output_text = f" -> {status['output']}" if status["output"] else ""
            logger.info(f"步骤 {step_id}: {step_name} {status_text}{output_text}")
    
    def generate_final_report(self):
        """生成最终报告"""
        if not self.output_dir:
            logger.error("输出目录未设置，无法生成报告")
            return
        
        report = {
            "pipeline_summary": {
                "execution_time": datetime.now().isoformat(),
                "output_directory": self.output_dir,
                "step_completion": {}
            },
            "step_details": {}
        }
        
        step_names = {
            "1.1": "文本预处理和分块",
            "1.2": "AI文本处理",
            "1.3": "文本质量评估",
            "2.1": "分类问题生成",
            "2.2": "问题格式转换",
            "2.3": "问题质量评估",
            "2.4": "答案生成",
            "3.0": "数据增强"
        }
        
        completed_steps = 0
        for step_id, step_name in step_names.items():
            status = self.step_status[step_id]
            report["pipeline_summary"]["step_completion"][step_id] = {
                "name": step_name,
                "completed": status["completed"],
                "output_file": status["output"]
            }
            
            if status["completed"]:
                completed_steps += 1
        
        report["pipeline_summary"]["completion_rate"] = f"{completed_steps}/{len(step_names)}"
        
        # 保存报告
        report_file = os.path.join(self.output_dir, "pipeline_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"流水线报告已保存至: {report_file}")
        return report


async def main():
    """主函数 - 支持单步骤运行、火山API、多线程和断点恢复"""
    parser = argparse.ArgumentParser(description="半导体QA生成系统 - 模块化版本（增强版）")
    
    # 基本参数
    parser.add_argument("--step", type=str, required=True,
                        choices=["1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "2.4", "3.0", "all", "status", "report"],
                        help="要执行的步骤")
    parser.add_argument("--input-dir", type=str, default="data/texts",
                        help="输入文本文件目录（仅步骤1.1需要）")
    parser.add_argument("--output-dir", type=str, default="data/qa_results",
                        help="输出结果目录")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="qwq_32",
                        help="使用的模型名称")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="批处理大小")
    parser.add_argument("--gpu-devices", type=str, default="0,1",
                        help="GPU设备ID")
    parser.add_argument("--quality-threshold", type=float, default=0.7,
                        help="问题质量阈值")
    
    # 火山API参数
    parser.add_argument("--use-volcano", action="store_true",
                        help="使用火山引擎API")
    parser.add_argument("--volcano-api-key", type=str, default=None,
                        help="火山引擎API密钥")
    
    # 多线程参数
    parser.add_argument("--max-workers", type=int, default=None,
                        help="最大工作线程数（默认为CPU核心数）")
    parser.add_argument("--no-multithread", action="store_true",
                        help="禁用多线程处理")
    
    # 断点恢复参数
    parser.add_argument("--resume", action="store_true",
                        help="从断点恢复执行")
    parser.add_argument("--no-resume", action="store_true",
                        help="强制重新开始，忽略断点")
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载配置文件: {args.config}")
            
            # 设置环境变量
            if config.get('api', {}).get('use_vllm_http'):
                os.environ['USE_VLLM_HTTP'] = 'true'
                os.environ['VLLM_SERVER_URL'] = config['api'].get('vllm_server_url', 'http://localhost:8000/v1')
                os.environ['USE_LOCAL_MODELS'] = str(config['api'].get('use_local_models', True)).lower()
                logger.info(f"启用vLLM HTTP模式，服务器地址: {os.environ['VLLM_SERVER_URL']}")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    # 初始化流水线（支持火山API和多线程）
    pipeline = ModularQAPipeline(
        config=config,
        use_volcano_api=args.use_volcano,
        volcano_api_key=args.volcano_api_key,
        max_workers=args.max_workers
    )
    
    # 设置目录
    pipeline.setup_directories(args.output_dir)
    
    # 如果启用断点恢复，尝试加载之前的状态
    if args.resume and not args.no_resume:
        if pipeline.load_checkpoint(args.output_dir):
            logger.info("成功从断点恢复")
        else:
            logger.info("未找到断点信息，从头开始")
    elif args.no_resume:
        logger.info("强制重新开始，忽略断点")
    
    # 确定是否使用多线程
    use_multithread = not args.no_multithread
    
    try:
        # 根据指定步骤执行
        if args.step == "status":
            pipeline.show_status()
        elif args.step == "report":
            pipeline.generate_final_report()
        elif args.step == "1.1":
            await pipeline.step_1_1_text_preprocessing(args.input_dir)
        elif args.step == "1.2":
            await pipeline.step_1_2_ai_processing(
                batch_size=args.batch_size,
                use_multithread=use_multithread
            )
        elif args.step == "1.3":
            await pipeline.step_1_3_text_quality_judgment(
                model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
            )
        elif args.step == "2.1":
            await pipeline.step_2_1_classified_question_generation(
                model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
            )
        elif args.step == "2.2":
            await pipeline.step_2_2_question_format_conversion(
                model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
            )
        elif args.step == "2.3":
            await pipeline.step_2_3_question_quality_evaluation(
                quality_threshold=args.quality_threshold,
                model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
            )
        elif args.step == "2.4":
            await pipeline.step_2_4_answer_generation(
                model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
            )
        elif args.step == "3.0":
            await pipeline.step_3_0_data_enhancement()
        elif args.step == "all":
            # 执行完整流水线
            logger.info("=== 执行完整流水线 ===")
            logger.info(f"配置: 火山API={args.use_volcano}, 多线程={use_multithread}, 断点恢复={args.resume}")
            
            # 步骤1.1：文本预处理
            if not pipeline.step_status["1.1"]["completed"] or args.no_resume:
                await pipeline.step_1_1_text_preprocessing(args.input_dir)
            else:
                logger.info("步骤1.1已完成，跳过")
            
            # 步骤1.2：AI文本处理
            if not pipeline.step_status["1.2"]["completed"] or args.no_resume:
                await pipeline.step_1_2_ai_processing(
                    batch_size=args.batch_size,
                    use_multithread=use_multithread
                )
            else:
                logger.info("步骤1.2已完成，跳过")
            
            # 步骤1.3：文本质量评估
            if not pipeline.step_status["1.3"]["completed"] or args.no_resume:
                await pipeline.step_1_3_text_quality_judgment(
                    model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
                )
            else:
                logger.info("步骤1.3已完成，跳过")
            
            # 步骤2.1：分类问题生成
            if not pipeline.step_status["2.1"]["completed"] or args.no_resume:
                await pipeline.step_2_1_classified_question_generation(
                    model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
                )
            else:
                logger.info("步骤2.1已完成，跳过")
            
            # 步骤2.2：问题格式转换
            if not pipeline.step_status["2.2"]["completed"] or args.no_resume:
                await pipeline.step_2_2_question_format_conversion(
                    model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
                )
            else:
                logger.info("步骤2.2已完成，跳过")
            
            # 步骤2.3：问题质量评估
            if not pipeline.step_status["2.3"]["completed"] or args.no_resume:
                await pipeline.step_2_3_question_quality_evaluation(
                    quality_threshold=args.quality_threshold,
                    model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
                )
            else:
                logger.info("步骤2.3已完成，跳过")
            
            # 步骤2.4：答案生成
            if not pipeline.step_status["2.4"]["completed"] or args.no_resume:
                await pipeline.step_2_4_answer_generation(
                    model_name=args.model, batch_size=args.batch_size, gpu_devices=args.gpu_devices
                )
            else:
                logger.info("步骤2.4已完成，跳过")
            
            # 步骤3.0：数据增强
            if not pipeline.step_status["3.0"]["completed"] or args.no_resume:
                await pipeline.step_3_0_data_enhancement()
            else:
                logger.info("步骤3.0已完成，跳过")
            
            # 生成最终报告
            pipeline.generate_final_report()
            logger.info("=== 完整流水线执行完成 ===")
        
        # 显示当前状态
        pipeline.show_status()
        
    except Exception as e:
        logger.error(f"执行步骤 {args.step} 失败: {e}")
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())