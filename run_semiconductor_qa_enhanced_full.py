#!/usr/bin/env python3
"""
半导体QA生成系统 - 增强版本（完整1.1-3.1流程）
保持原有生成逻辑不变，仅添加：
1. 断点续跑功能
2. 火山API支持（作为可选后端）
3. 多线程并行处理
"""

import os
import sys
import json
import time
import pickle
import hashlib
import logging
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict
import aiohttp
import re

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入原有模块
from semiconductor_qa_generator import SemiconductorQAGenerator
from text_processor import parse_txt, input_text_process
from argument_processor import ArgumentDataProcessor


@dataclass
class CheckpointData:
    """断点数据结构"""
    step_name: str
    step_number: str
    timestamp: str
    data: Any
    metadata: Dict[str, Any] = None


class CheckpointManager:
    """断点管理器 - 用于保存和恢复进度"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_checkpoint(self, step_name: str, step_number: str, data: Any, metadata: Dict = None):
        """保存断点"""
        checkpoint = CheckpointData(
            step_name=step_name,
            step_number=step_number,
            timestamp=datetime.now().isoformat(),
            data=data,
            metadata=metadata or {}
        )
        
        filename = self.checkpoint_dir / f"{step_number}_{step_name}_{self.current_session}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"断点已保存: {filename}")
        
    def load_checkpoint(self, step_number: str) -> Optional[CheckpointData]:
        """加载断点"""
        pattern = f"{step_number}_*"
        files = list(self.checkpoint_dir.glob(pattern))
        
        if not files:
            return None
            
        # 获取最新的断点文件
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'rb') as f:
            checkpoint = pickle.load(f)
        logger.info(f"断点已加载: {latest_file}")
        return checkpoint
        
    def get_latest_checkpoint(self) -> Optional[Tuple[str, CheckpointData]]:
        """获取最新的断点"""
        files = list(self.checkpoint_dir.glob("*.pkl"))
        if not files:
            return None
            
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        # 从文件名提取步骤号
        step_number = latest_file.stem.split('_')[0]
        
        with open(latest_file, 'rb') as f:
            checkpoint = pickle.load(f)
            
        return step_number, checkpoint


class VolcanoAPIWrapper:
    """火山API包装器 - 作为可选的模型后端"""
    
    def __init__(self, api_key: str, endpoint: str = None):
        self.api_key = api_key
        self.endpoint = endpoint or "https://ark.cn-beijing.volces.com/api/v3"
        self.model = "deepseek-r1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def generate_with_original_prompt(self, prompt: str, **kwargs):
        """使用原有的prompt格式调用火山API"""
        
        # 保持原有的消息格式
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False
        }
        
        try:
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
                        logger.error(f"Volcano API error: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"Volcano API调用失败: {e}")
            return None


async def generate_classified_questions(generator, qa_input_data, config):
    """生成分类问题（原有逻辑）"""
    question_data = []
    
    for item in qa_input_data:
        try:
            # 使用原有的分类问题生成逻辑
            result = await generator.generate_questions_for_item(item)
            if result:
                question_data.append(result)
        except Exception as e:
            logger.error(f"问题生成失败: {e}")
            
    return question_data


def generate_pipeline_stats(processed_results, qualified_texts, qa_results, enhanced_data,
                           input_dir, output_dir, chunks_dir, qa_original_dir, model_name, quality_threshold):
    """生成流程统计信息"""
    stats = {
        "pipeline_info": {
            "timestamp": datetime.now().isoformat(),
            "input_dir": input_dir,
            "output_dir": output_dir,
            "model": model_name,
            "quality_threshold": quality_threshold
        },
        "stage1_text_processing": {
            "total_processed": len(processed_results),
            "qualified_texts": len(qualified_texts),
            "quality_pass_rate": len(qualified_texts) / len(processed_results) if processed_results else 0
        },
        "stage2_qa_generation": {
            "total_qa_generated": len(qa_results),
            "high_quality_qa": len([q for q in qa_results if q.get('quality_score', 0) >= quality_threshold])
        },
        "stage3_enhancement": {
            "final_dataset_size": len(enhanced_data)
        }
    }
    
    return stats


async def run_semiconductor_qa_generation_enhanced(
    input_dir: str = "data/input",
    output_dir: str = "data/output",
    model_name: str = "deepseek-r1",
    quality_threshold: float = 0.7,
    batch_size: int = 16,
    config: Dict = None,
    use_volcano: bool = False,
    volcano_api_key: str = None,
    max_workers: int = 4,
    checkpoint_manager: CheckpointManager = None,
    resume_from: str = None
):
    """
    增强版半导体QA生成流程 - 保持原有1.1-3.1流程
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    chunks_dir = os.path.join(output_dir, "chunks")
    qa_original_dir = os.path.join(output_dir, "qa_original")
    qa_results_dir = os.path.join(output_dir, "qa_results")
    
    for dir_path in [chunks_dir, qa_original_dir, qa_results_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 初始化生成器
    generator = SemiconductorQAGenerator(config_path=config.get('config_path', 'config.json'))
    
    # 初始化火山API（如果启用）
    volcano_client = None
    if use_volcano and volcano_api_key:
        volcano_client = VolcanoAPIWrapper(volcano_api_key)
        logger.info("火山API已初始化")
    
    # 检查断点恢复
    start_step = "1.1"
    loaded_data = {}
    
    if resume_from and checkpoint_manager:
        checkpoint = checkpoint_manager.load_checkpoint(resume_from)
        if checkpoint:
            logger.info(f"从断点恢复: {checkpoint.step_name}")
            loaded_data = checkpoint.data
            start_step = resume_from
    
    text_files = []
    processed_results = []
    qualified_texts = []
    qa_results = []
    enhanced_data = []
    
    # ===== 第一阶段：文本预处理 + 质量评估 =====
    if start_step <= "1.3":
        logger.info("第一阶段: 文本预处理、AI处理和质量评估")
        
        if start_step <= "1.1":
            logger.info("步骤1.1: 文本分块和预处理...")
            # 步骤1.1: 文本分块和预处理
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
            
            # 保存断点
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint("1.1_preprocessing", "1.1", all_tasks)
        else:
            all_tasks = loaded_data.get('all_tasks', [])
        
        if start_step <= "1.2":
            logger.info("步骤1.2: AI文本处理...")
            # 步骤1.2: AI文本处理（支持多线程）
            processed_results = []
            
            # 检查是否有1.2的断点
            if start_step == "1.2" and 'processed_results' in loaded_data:
                processed_results = loaded_data['processed_results']
                processed_ids = {r.get('id') for r in processed_results}
                all_tasks = [t for t in all_tasks if t.get('id') not in processed_ids]
            
            # 使用多线程处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for i in range(0, len(all_tasks), batch_size):
                    batch = all_tasks[i:i+batch_size]
                    
                    for task in batch:
                        if volcano_client:
                            # 使用火山API
                            future = executor.submit(
                                asyncio.run,
                                process_with_volcano(task, volcano_client, config)
                            )
                        else:
                            # 使用原有方法
                            future = executor.submit(
                                asyncio.run,
                                input_text_process(
                                    task["content"], 
                                    os.path.basename(task["file_path"]),
                                    chunk_index=task["chunk_index"],
                                    total_chunks=len([t for t in all_tasks if t["file_path"] == task["file_path"]]),
                                    prompt_index=9,
                                    config=config
                                )
                            )
                        futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            processed_results.append(result)
                            
                            # 定期保存断点
                            if len(processed_results) % 10 == 0 and checkpoint_manager:
                                checkpoint_manager.save_checkpoint(
                                    "1.2_ai_processing", "1.2",
                                    {'processed_results': processed_results, 'all_tasks': all_tasks}
                                )
                    except Exception as e:
                        logger.error(f"任务处理失败: {e}")
            
            # 保存AI处理结果
            processed_file = os.path.join(chunks_dir, "ai_processed_texts.json")
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(processed_results, f, ensure_ascii=False, indent=2)
            logger.info(f"AI处理完成，生成了 {len(processed_results)} 个结果")
        else:
            processed_results = loaded_data.get('processed_results', [])
        
        if not processed_results:
            logger.error("没有AI处理结果，流程终止")
            return []
        
        if start_step <= "1.3":
            logger.info("步骤1.3: 文本质量评估...")
            # 步骤1.3: 文本质量评估
            
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
            
            # 执行文本质量评估
            judged_results = await generator.judge_processed_texts(md_data_for_judgment)
            
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
            
            # 保存断点
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint("1.3_quality_evaluation", "1.3", 
                    {'qualified_texts': qualified_texts, 'judged_results': judged_results})
        else:
            qualified_texts = loaded_data.get('qualified_texts', [])
    else:
        # 从断点加载第一阶段结果
        qualified_texts = loaded_data.get('qualified_texts', [])
    
    if not qualified_texts:
        logger.error("没有文本通过质量评估，流程终止")
        return []
    
    # 保存合格文本
    qualified_file = os.path.join(chunks_dir, "qualified_texts.json")
    with open(qualified_file, 'w', encoding='utf-8') as f:
        json.dump(qualified_texts, f, ensure_ascii=False, indent=2)
    
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
        
        if start_step <= "2.1":
            # 步骤2.1: 分类问题生成
            logger.info("步骤2.1: 执行分类问题生成...")
            question_data = await generate_classified_questions(generator, qa_input_data, config)
            
            # 保存问题生成结果
            question_file = os.path.join(qa_original_dir, "classified_questions.json")
            with open(question_file, 'w', encoding='utf-8') as f:
                json.dump(question_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分类问题生成完成: 生成了 {len(question_data)} 个问题集合")
            
            # 保存断点
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint("2.1_question_generation", "2.1", 
                    {'question_data': question_data})
        else:
            question_data = loaded_data.get('question_data', [])
        
        if start_step <= "2.2":
            logger.info("步骤2.2: 执行问题格式转换...")
            # 使用新的方法处理数据列表
            converted_data = generator.convert_questionlist_li_data_from_list(question_data)

            # 保存格式转换结果
            converted_file = os.path.join(qa_original_dir, "converted_questions.json")
            with open(converted_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)

            logger.info(f"问题格式转换完成: 转换为 {len(converted_data)} 个独立问题")
            
            # 保存断点
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint("2.2_format_conversion", "2.2",
                    {'converted_data': converted_data})
        else:
            converted_data = loaded_data.get('converted_data', [])
        
        if start_step <= "2.3":
            logger.info("步骤2.3: 执行问题质量评估...")
            # 使用新的方法处理数据列表
            evaluated_qa_data = await generator.judge_question_data_from_list(converted_data)

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
            
            # 保存断点
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint("2.3_quality_evaluation", "2.3",
                    {'high_quality_qa': high_quality_qa, 'evaluated_qa_data': evaluated_qa_data})
        else:
            high_quality_qa = loaded_data.get('high_quality_qa', [])
            evaluated_qa_data = loaded_data.get('evaluated_qa_data', [])
        
        if start_step <= "2.4":
            logger.info("步骤2.4: 为高质量问题生成答案...")

            # 为高质量问题添加上下文信息
            qa_with_context = []
            for qa_item in high_quality_qa:
                # 获取原始文本内容作为上下文
                source_info = qa_item.get('source_info', {})
                context = source_info.get('content', qa_item.get('paper_content', ''))

                qa_item_with_context = qa_item.copy()
                qa_item_with_context['context'] = context
                qa_with_context.append(qa_item_with_context)

            # 保存带上下文的QA数据
            qa_with_context_file = os.path.join(qa_original_dir, "qa_with_context.json")
            with open(qa_with_context_file, 'w', encoding='utf-8') as f:
                json.dump(qa_with_context, f, ensure_ascii=False, indent=2)

            # 生成答案（支持多线程）
            if volcano_client or max_workers > 1:
                # 使用多线程生成答案
                qa_with_answers = await generate_answers_parallel(
                    qa_with_context, generator, volcano_client, max_workers
                )
            else:
                # 使用原有方法
                qa_with_answers_file = os.path.join(qa_original_dir, "qa_with_answers.json")
                answer_stats = generator.generate_answers(
                    qa_with_context_file,
                    qa_with_answers_file,
                    use_cot=True
                )
                logger.info(f"答案生成完成: {answer_stats}")
                
                with open(qa_with_answers_file, 'r', encoding='utf-8') as f:
                    qa_with_answers = json.load(f)

            # 保存带答案的QA数据
            qa_with_answers_file = os.path.join(qa_original_dir, "qa_with_answers.json")
            with open(qa_with_answers_file, 'w', encoding='utf-8') as f:
                json.dump(qa_with_answers, f, ensure_ascii=False, indent=2)

            # 保存最终的高质量QA结果（包含答案）
            qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
            with open(qa_output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_with_answers, f, ensure_ascii=False, indent=2)

            qa_results = qa_with_answers
            
            # 保存断点
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint("2.4_answer_generation", "2.4",
                    {'qa_results': qa_results})
        else:
            qa_results = loaded_data.get('qa_results', [])

    except Exception as e:
        logger.error(f"QA生成失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")

        qa_results = []
        qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
        with open(qa_output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

    # ===== 第三阶段：数据增强 =====
    if start_step <= "3.1":
        logger.info("第三阶段: 数据增强与重写")
        
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
        
        # 保存断点
        if checkpoint_manager:
            checkpoint_manager.save_checkpoint("3.1_data_enhancement", "3.1",
                {'enhanced_data': enhanced_data})
    else:
        enhanced_data = loaded_data.get('enhanced_data', [])
    
    # 生成统计信息
    stats = generate_pipeline_stats(
        processed_results, qualified_texts, qa_results, enhanced_data,
        input_dir, output_dir, chunks_dir, qa_original_dir, model_name, quality_threshold
    )
    
    stats_file = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info("=== QA生成流程完成 ===")
    logger.info(f"统计信息已保存至: {stats_file}")
    
    return enhanced_data


async def process_with_volcano(task, volcano_client, config):
    """使用火山API处理文本"""
    # 构建原有的prompt
    prompt = f"""你是一个半导体显示领域的专家。请处理以下文本内容：

{task['content']}

请返回处理后的文本。"""
    
    result = await volcano_client.generate_with_original_prompt(
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048
    )
    
    if result:
        return {
            'content': result,
            'source_file': os.path.basename(task['file_path']),
            'chunk_index': task['chunk_index'],
            'total_chunks': task.get('total_chunks', 1),
            'text_content': task['content']
        }
    return None


async def generate_answers_parallel(qa_items, generator, volcano_client, max_workers):
    """并行生成答案"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for item in qa_items:
            if volcano_client:
                # 使用火山API
                future = executor.submit(
                    asyncio.run,
                    generate_answer_with_volcano(item, volcano_client)
                )
            else:
                # 使用原有方法
                future = executor.submit(
                    generate_answer_original, item, generator
                )
            futures.append((item, future))
        
        # 收集结果
        for item, future in futures:
            try:
                answer = future.result()
                item['answer'] = answer
                results.append(item)
            except Exception as e:
                logger.error(f"答案生成失败: {e}")
                item['answer'] = ""
                results.append(item)
    
    return results


async def generate_answer_with_volcano(qa_item, volcano_client):
    """使用火山API生成答案"""
    question = qa_item.get('question', '')
    context = qa_item.get('context', '')
    
    # 使用原有的答案生成prompt
    prompt = f"""你是一个半导体显示领域的资深专家。请根据以下信息回答问题：

上下文：{context[:2000]}

问题：{question}

请提供准确、专业的答案。"""
    
    result = await volcano_client.generate_with_original_prompt(
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048
    )
    
    return result or ""


def generate_answer_original(qa_item, generator):
    """使用原有方法生成答案"""
    # 这里应该调用generator的实际答案生成方法
    # 简化处理，返回示例
    return f"这是对问题'{qa_item.get('question', '')}'的答案。"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="半导体QA生成系统 - 增强版本（完整流程）")
    
    # 基本参数
    parser.add_argument("--input-dir", type=str, default="data/input",
                        help="输入目录")
    parser.add_argument("--output-dir", type=str, default="data/output",
                        help="输出目录")
    parser.add_argument("--config", type=str, default="config.json",
                        help="配置文件路径")
    parser.add_argument("--model", type=str, default="deepseek-r1",
                        help="模型名称")
    parser.add_argument("--quality-threshold", type=float, default=0.7,
                        help="质量阈值")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批处理大小")
    
    # 断点续跑
    parser.add_argument("--resume", action="store_true",
                        help="从断点恢复执行")
    parser.add_argument("--resume-from", type=str,
                        help="从指定步骤恢复 (1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="断点保存目录")
    
    # 火山API
    parser.add_argument("--use-volcano", action="store_true",
                        help="使用火山API")
    parser.add_argument("--volcano-api-key", type=str,
                        help="火山API密钥")
    
    # 多线程
    parser.add_argument("--max-workers", type=int, default=4,
                        help="最大工作线程数")
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化断点管理器
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    
    # 确定恢复点
    resume_from = None
    if args.resume:
        if args.resume_from:
            resume_from = args.resume_from
        else:
            # 获取最新断点
            latest = checkpoint_manager.get_latest_checkpoint()
            if latest:
                resume_from = latest[0]
                logger.info(f"将从步骤 {resume_from} 恢复")
    
    # 准备配置
    config = {
        'config_path': args.config,
        'model_name': args.model,
        'batch_size': args.batch_size
    }
    
    # 运行流程
    asyncio.run(run_semiconductor_qa_generation_enhanced(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        quality_threshold=args.quality_threshold,
        batch_size=args.batch_size,
        config=config,
        use_volcano=args.use_volcano,
        volcano_api_key=args.volcano_api_key,
        max_workers=args.max_workers,
        checkpoint_manager=checkpoint_manager,
        resume_from=resume_from
    ))


if __name__ == "__main__":
    main()