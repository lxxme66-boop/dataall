#!/usr/bin/env python3
"""
半导体QA生成系统 - 统一入口脚本
按照正确的流程调用各个模块：
1. text_processor.py - 文本预处理
2. text_main_batch_inference_enhanced.py - 文本召回与批量推理（新增）
3. clean_text_data.py - 数据清洗（新增）
4. semiconductor_qa_generator.py - 核心QA生成
5. 质量检查 - 独立的质量评估步骤（新增）
6. argument_data.py - 数据增强与重写
7. 最终输出整理 - 生成统计报告（新增）
"""
from TextGeneration.Datageneration import parse_txt, input_text_process,merge_chunk_responses
import asyncio
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pickle
from TextGeneration.Datageneration import parse_txt
from enhanced_file_processor import process_text_chunk
# Change line 24 in run_semiconductor_qa.py from:
from enhanced_file_processor import process_text_chunk

import semiconductor_qa_generator
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入各个模块
from text_processor import TextProcessor
from semiconductor_qa_generator import run_semiconductor_qa_generation

# 尝试导入argument_data模块
try:
    from argument_data import ArgumentDataProcessor
    ARGUMENT_DATA_AVAILABLE = True
except ImportError:
    ARGUMENT_DATA_AVAILABLE = False
    logger.warning("数据增强模块不可用（缺少volcenginesdkarkruntime）")
    
    # 创建一个mock类
    class ArgumentDataProcessor:
        """Mock ArgumentDataProcessor class"""
        def __init__(self):
            pass
        
        async def process_qa_data(self, *args, **kwargs):
            logger.warning("数据增强功能不可用，跳过此步骤")
            return args[0] if args else []

# 导入新增的模块
try:
    from text_main_batch_inference_enhanced import process_folders
    TEXT_RETRIEVAL_AVAILABLE = True
except ImportError:
    TEXT_RETRIEVAL_AVAILABLE = False
    logger.warning("文本召回模块不可用")

try:
    from clean_text_data import clean_data
    DATA_CLEANING_AVAILABLE = True
except ImportError:
    DATA_CLEANING_AVAILABLE = False
    logger.warning("数据清洗模块不可用")

try:
    from TextQA.enhanced_quality_checker import TextQAQualityIntegrator
    QUALITY_CHECK_AVAILABLE = True
except ImportError:
    QUALITY_CHECK_AVAILABLE = False
    logger.warning("增强质量检查模块不可用")


# async def run_complete_pipeline(
#     config: dict,  # 添加 config 作为第一个参数
#     input_dir: str = "data/texts",
#     output_dir: str = "data/qa_results",
#     model_name: str = "qwq_32",
#     batch_size: int = 2,
#     gpu_devices: str = "0,1",
#     enable_full_steps: bool = False  # 新增参数：是否启用完整7步骤
# ):
#     """运行完整的QA生成流程"""
    
#     logger.info("=== 开始半导体QA生成流程 ===")
#     logger.info(f"模式: {'完整7步骤' if enable_full_steps else '精简3步骤'}")
    
#         # 确保使用配置中的路径（如果存在）
#     if 'paths' in config and 'text_dir' in config['paths']:
#         input_dir = config['paths']['text_dir']
#     if 'paths' in config and 'output_dir' in config['paths']:
#         output_dir = config['paths']['output_dir']
    
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
#     preprocessed_file = None
#     text_files = []
    
#     # 步骤1: 文本预处理
#     all_tasks = []
#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith('.txt'):
#                 file_path = os.path.join(root, file)
#                 text_files.append(file_path)  # 记录文件
                
#                 # 调用修复后的 parse_txt
#                 file_tasks = await parse_txt(file_path, index=9, config=config)
                
#                 if file_tasks:
#                     logger.info(f"为文件 {file} 创建了 {len(file_tasks)} 个处理任务")
#                     all_tasks.extend(file_tasks)
    
#     # 处理任务
#     results = []
#     for i in range(0, len(all_tasks), batch_size):
#         batch = all_tasks[i:i+batch_size]
#         batch_results = await asyncio.gather(
#             *(process_text_chunk(task) for task in batch),
#             return_exceptions=True
#         )
        
#         # 处理结果
#         for result in batch_results:
#             if not isinstance(result, Exception) and result.get("success"):
#                 results.append(result)
    
#     # 保存预处理结果
#     if results:
#         preprocessed_file = os.path.join(output_dir, "preprocessed_texts.json")
#         with open(preprocessed_file, 'w', encoding='utf-8') as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
#         processed_texts = results
#     else:
#         logger.error("没有生成有效结果，跳过后续步骤")
#         return []
    
    
#     # 步骤2: 文本召回与批量推理（新增，可选）
#     if enable_full_steps and TEXT_RETRIEVAL_AVAILABLE:
#         logger.info("步骤2: 文本召回与批量推理")
#         try:
#             # 创建临时文件夹
#             temp_folder = os.path.join(output_dir, "temp")
#             os.makedirs(temp_folder, exist_ok=True)
            
#             # 运行文本召回
#             retrieval_results = await process_folders(
#                 folders=[input_dir],
#                 txt_path=input_dir,
#                 temporary_folder=temp_folder,
#                 index=43,  # 半导体领域的索引
#                 maximum_tasks=20,
#                 selected_task_number=500,
#                 storage_folder=output_dir,
#                 read_hist=False
#             )
            
#             # 保存召回结果
#             retrieval_file = os.path.join(output_dir, "retrieval_results.pkl")
#             with open(retrieval_file, 'wb') as f:
#                 pickle.dump(retrieval_results, f)
            
#             logger.info(f"文本召回完成，生成 {len(retrieval_results)} 个召回结果")
#         except Exception as e:
#             logger.warning(f"文本召回步骤失败: {e}，继续执行后续步骤")
    
#     # 步骤3: 数据清洗（新增，可选）
#     if enable_full_steps and DATA_CLEANING_AVAILABLE:
#         logger.info("步骤3: 数据清洗")
#         try:
#             # 如果有召回结果，使用召回结果；否则使用预处理结果
#             input_for_cleaning = retrieval_file if 'retrieval_file' in locals() else preprocessed_file
            
#             # 运行数据清洗
#             cleaned_data = await clean_data(
#                 input_file=input_for_cleaning,
#                 output_dir=output_dir
#             )
            
#             # 保存清洗后的数据
#             cleaned_file = os.path.join(output_dir, "cleaned_texts.json")
#             with open(cleaned_file, 'w', encoding='utf-8') as f:
#                 json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
#             # 更新预处理文件路径
#             preprocessed_file = cleaned_file
#             logger.info(f"数据清洗完成，清洗后数据保存至: {cleaned_file}")
#         except Exception as e:
#             logger.warning(f"数据清洗步骤失败: {e}，使用原始预处理数据")
    
#     # 步骤4: 使用semiconductor_qa_generator生成QA（原步骤2）
#     logger.info(f"步骤{4 if enable_full_steps else 2}: 核心QA生成")
    
#     # 准备输入文件列表
#     input_files = [preprocessed_file]
#     qa_output_file = os.path.join(output_dir, "qa_generated.json")
#     output_files = [qa_output_file]
    
#     # 运行QA生成
#     qa_results = await run_semiconductor_qa_generation(
#         raw_folders=input_files,
#         save_paths=output_files,
#         model_name=model_name,
#         batch_size=batch_size,
#         gpu_devices=gpu_devices
#     )
    
#     logger.info(f"QA生成完成，生成了 {len(qa_results)} 个QA对")
    
#     # 步骤5: 独立的质量检查（新增，可选）
#     if enable_full_steps and QUALITY_CHECK_AVAILABLE:
#         logger.info("步骤5: 独立质量检查")
#         try:
#             # 加载配置
#             config_path = "config.json"
#             if os.path.exists(config_path):
#                 with open(config_path, 'r', encoding='utf-8') as f:
#                     config = json.load(f)
#             else:
#                 config = {
#                     "quality_control": {
#                         "enhanced_quality_check": {
#                             "quality_threshold": 0.7
#                         }
#                     }
#                 }
            
#             # 初始化质量检查器
#             quality_checker = TextQAQualityIntegrator(config)
            
#             # 运行质量检查
#             quality_report = await quality_checker.enhanced_quality_check(
#                 qa_file_path=qa_output_file,
#                 output_dir=output_dir,
#                 quality_threshold=config['quality_control']['enhanced_quality_check']['quality_threshold']
#             )
            
#             # 保存质量报告
#             quality_report_file = os.path.join(output_dir, "quality_report.json")
#             with open(quality_report_file, 'w', encoding='utf-8') as f:
#                 json.dump(quality_report, f, ensure_ascii=False, indent=2)
            
#             logger.info(f"质量检查完成，通过率: {quality_report.get('pass_rate', 0):.2%}")
#             logger.info(f"质量报告保存至: {quality_report_file}")
#         except Exception as e:
#             logger.warning(f"质量检查步骤失败: {e}，继续执行后续步骤")
    
#     # 步骤6: 数据增强与重写（原步骤3）
#     logger.info(f"步骤{6 if enable_full_steps else 3}: 数据增强与重写")
    
#     # 初始化数据增强处理器
#     argument_processor = ArgumentDataProcessor()
    
#     # 加载生成的QA数据
#     with open(qa_output_file, 'r', encoding='utf-8') as f:
#         qa_data = json.load(f)
    
#     # 进行数据增强
#     enhanced_data = await argument_processor.enhance_qa_data(qa_data)
    
#     # 保存最终结果
#     final_output_file = os.path.join(output_dir, "final_qa_dataset.json")
#     with open(final_output_file, 'w', encoding='utf-8') as f:
#         json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
#     logger.info(f"数据增强完成，最终数据集保存至: {final_output_file}")
    
#     # 步骤7: 最终输出整理（新增）
#     if enable_full_steps:
#         logger.info("步骤7: 最终输出整理")
        
#         # 生成详细统计信息
#         stats = {
#             "pipeline_info": {
#                 "mode": "完整7步骤" if enable_full_steps else "精简3步骤",
#                 "input_directory": input_dir,
#                 "output_directory": output_dir,
#                 "model_used": model_name,
#                 "gpu_devices": gpu_devices,
#                 "batch_size": batch_size
#             },
#             "processing_stats": {
#                 "total_texts_processed": len(processed_texts),
#                 "total_qa_generated": len(qa_results),
#                 "total_qa_enhanced": len(enhanced_data),
#                 "files_processed": len(text_files)
#             }
#         }
        
#         # 添加质量检查统计（如果有）
#         if 'quality_report' in locals():
#             stats["quality_stats"] = {
#                 "total_qa_pairs": quality_report.get('total_qa_pairs', 0),
#                 "passed_qa_pairs": quality_report.get('passed_qa_pairs', 0),
#                 "pass_rate": quality_report.get('pass_rate', 0),
#                 "meets_threshold": quality_report.get('meets_threshold', False)
#             }
        
#         # 生成问题类型分布统计
#         question_types = {}
#         for item in enhanced_data:
#             if 'question_type' in item:
#                 q_type = item['question_type']
#                 question_types[q_type] = question_types.get(q_type, 0) + 1
        
#         stats["question_distribution"] = question_types
        
#         # 保存统计信息
#         stats_file = os.path.join(output_dir, "pipeline_stats.json")
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             json.dump(stats, f, ensure_ascii=False, indent=2)
        
#         # 生成摘要报告
#         summary_report = f"""
# === 半导体QA生成流程完成 ===

# 处理模式: {stats['pipeline_info']['mode']}
# 输入目录: {stats['pipeline_info']['input_directory']}
# 输出目录: {stats['pipeline_info']['output_directory']}

# 处理统计:
# - 处理文件数: {stats['processing_stats']['files_processed']}
# - 处理文本段落数: {stats['processing_stats']['total_texts_processed']}
# - 生成QA对数: {stats['processing_stats']['total_qa_generated']}
# - 增强后QA对数: {stats['processing_stats']['total_qa_enhanced']}
# """
        
#         if 'quality_stats' in stats:
#             summary_report += f"""
# 质量检查:
# - 通过率: {stats['quality_stats']['pass_rate']:.2%}
# - 是否达标: {'是' if stats['quality_stats']['meets_threshold'] else '否'}
# """
        
#         if question_types:
#             summary_report += "\n问题类型分布:"
#             for q_type, count in question_types.items():
#                 summary_report += f"\n- {q_type}: {count}"
        
#         summary_report += f"""

# 输出文件:
# - 最终数据集: {final_output_file}
# - 统计信息: {stats_file}
# """
        
#         # 保存摘要报告
#         summary_file = os.path.join(output_dir, "summary_report.txt")
#         with open(summary_file, 'w', encoding='utf-8') as f:
#             f.write(summary_report)
        
#         logger.info(summary_report)
#         logger.info(f"摘要报告保存至: {summary_file}")
#     else:
#         # 精简模式下的简单统计
#         stats = {
#             "total_texts_processed": len(processed_texts),
#             "total_qa_generated": len(qa_results),
#             "total_qa_enhanced": len(enhanced_data),
#             "input_directory": input_dir,
#             "output_directory": output_dir,
#             "model_used": model_name
#         }
        
#         stats_file = os.path.join(output_dir, "pipeline_stats.json")
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             json.dump(stats, f, ensure_ascii=False, indent=2)
    
#     logger.info("=== QA生成流程完成 ===")
#     logger.info(f"统计信息已保存至: {stats_file}")
    
#     return enhanced_data
import asyncio
import json
import os
from datetime import datetime
# 原有的导入保持不变
from semiconductor_qa_generator import SemiconductorQAGenerator  # 保持原有导入
# ... 其他导入

# =============================================================================
# 主要修改：替换原有的 run_complete_pipeline 函数
# =============================================================================

async def run_complete_pipeline(
    config: dict,
    input_dir: str = "data/texts",
    output_dir: str = "data/qa_results", 
    model_name: str = "qwq_32",
    batch_size: int = 2,
    gpu_devices: str = "0,1",
    quality_threshold: float = 0.7,  # 新增：问题质量阈值
    enable_full_steps: bool = False  # 新增：是否启用完整步骤
):
    """运行完整的QA生成流程 - 优化版本
    
    Args:
        config: 配置字典
        input_dir: 输入文本目录
        output_dir: 输出结果目录
        model_name: 使用的模型名称
        batch_size: 批处理大小
        gpu_devices: GPU设备
        quality_threshold: 问题质量阈值 (0.7表示只保留质量评分>=0.7的问题)
    """
    
    logger.info("=== 开始半导体QA生成流程 ===")
    logger.info(f"质量阈值: {quality_threshold}")
    
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
    
    # 初始化QA生成器 - 修复参数传递问题
    logger.info(f"DEBUG: 初始化前 USE_VLLM_HTTP={os.environ.get('USE_VLLM_HTTP')}")
    generator = SemiconductorQAGenerator(
        batch_size=batch_size,
        gpu_devices=gpu_devices
        # 不传递model_name参数，因为__init__不接受这个参数
    )
    # 修复：确保必要的属性被初始化
    if not hasattr(generator, 'stats'):
        generator.stats = {}

    if not hasattr(generator, 'stats'):
        generator.stats = {
            "generated_questions": 0,
            "total_questions": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }
    # 通过属性设置模型名称
    generator.model_name = model_name
    logger.info(f"设置模型名称为: {model_name}")
    
    text_files = []
    
    # ===== 第一阶段：文本预处理 + 质量评估 =====
    logger.info("第一阶段: 文本预处理、AI处理和质量评估")
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
    
    # 步骤1.2: AI文本处理
    logger.info("步骤1.2: AI文本处理...")
    processed_results = []
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i+batch_size]
        
        batch_results = await asyncio.gather(
            *(input_text_process(
                task["content"], 
                os.path.basename(task["file_path"]),
                chunk_index=task["chunk_index"],
                total_chunks=len([t for t in all_tasks if t["file_path"] == task["file_path"]]),
                prompt_index=9,
                config=config
            ) for task in batch),
            return_exceptions=True
        )
        
        for result in batch_results:
            if not isinstance(result, Exception) and result is not None:
                processed_results.append(result)
            else:
                logger.error(f"任务处理失败: {result}")
    
    # 保存AI处理结果
    processed_file = os.path.join(chunks_dir, "ai_processed_texts.json")
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
    logger.info(f"AI处理完成，生成了 {len(processed_results)} 个结果")
    
    if not processed_results:
        logger.error("没有AI处理结果，流程终止")
        return []
    
    # 步骤1.3: 文本质量评估
    logger.info("步骤1.3: 文本质量评估...")
    
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
    
    if not qualified_texts:
        logger.error("没有文本通过质量评估，流程终止")
        return []
    
    # 保存合格文本
    qualified_file = os.path.join(chunks_dir, "qualified_texts.json")
    with open(qualified_file, 'w', encoding='utf-8') as f:
        json.dump(qualified_texts, f, ensure_ascii=False, indent=2)
    
    # ===== 第二阶段：QA生成（3个步骤）=====
    logger.info("第二阶段: QA生成（问题生成 → 格式转换 → 质量评估）")
    
    try:
        # 准备QA生成的输入数据
        qa_input_data = []
        for text in qualified_texts:
            qa_input_data.append({
                "paper_name": f"{text['source_file']}_chunk_{text['chunk_index']}",
                "md_content": text['content'],
                "source_info": text
            })
        
        # 步骤2.1: 分类问题生成
        logger.info("步骤2.1: 执行分类问题生成...")
        question_data = await generate_classified_questions(generator, qa_input_data, config)
        
        # 保存问题生成结果
        question_file = os.path.join(qa_original_dir, "classified_questions.json")
        with open(question_file, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分类问题生成完成: 生成了 {len(question_data)} 个问题集合")
        
        # 步骤2.2: 问题格式转换
#         logger.info("步骤2.2: 执行问题格式转换...")
#         converted_data = await generator.convert_questionlist_li_data(question_data)
        
#         # 保存格式转换结果
#         converted_file = os.path.join(qa_original_dir, "converted_questions.json")
#         with open(converted_file, 'w', encoding='utf-8') as f:
#             json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
#         logger.info(f"问题格式转换完成: 转换为 {len(converted_data)} 个独立问题")
        logger.info("步骤2.2: 执行问题格式转换...")
        # 修改：使用新的方法处理数据列表
        converted_data = generator.convert_questionlist_li_data_from_list(question_data)

        # 保存格式转换结果
        converted_file = os.path.join(qa_original_dir, "converted_questions.json")
        with open(converted_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)

        logger.info(f"问题格式转换完成: 转换为 {len(converted_data)} 个独立问题")
        
        # 步骤2.3: 问题质量评估和筛选
#         logger.info("步骤2.3: 执行问题质量评估...")
#         evaluated_qa_data = await generator.judge_question_data(converted_data)
        
#         # 保存评估结果
#         evaluated_file = os.path.join(qa_original_dir, "evaluated_qa_data.json")
#         with open(evaluated_file, 'w', encoding='utf-8') as f:
#             json.dump(evaluated_qa_data, f, ensure_ascii=False, indent=2)
        
#         # 根据质量阈值筛选高质量问题
#         high_quality_qa = []
#         for qa_item in evaluated_qa_data:
#             quality_score = qa_item.get('quality_score', 0)
#             if quality_score >= quality_threshold:
#                 high_quality_qa.append(qa_item)
        
#         logger.info(f"问题质量评估完成: {len(evaluated_qa_data)} -> {len(high_quality_qa)} 高质量问题")
        logger.info("步骤2.3: 执行问题质量评估...")
        # 修改：使用新的方法处理数据列表
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
    
        
#         # 步骤2.4: 答案生成（新增）
#         logger.info("步骤2.4: 为高质量问题生成答案...")
        
#         # 为高质量问题添加上下文信息
#         qa_with_context = []
#         for qa_item in high_quality_qa:
#             # 获取原始文本内容作为上下文
#             source_info = qa_item.get('source_info', {})
#             context = source_info.get('content', qa_item.get('paper_content', ''))
            
#             qa_item_with_context = qa_item.copy()
#             qa_item_with_context['context'] = context
#             qa_with_context.append(qa_item_with_context)
        
#         # 保存带上下文的QA数据
#         qa_with_context_file = os.path.join(qa_original_dir, "qa_with_context.json")
#         with open(qa_with_context_file, 'w', encoding='utf-8') as f:
#             json.dump(qa_with_context, f, ensure_ascii=False, indent=2)
        
#         # 生成答案
#         qa_with_answers_file = os.path.join(qa_original_dir, "qa_with_answers.json")
#         answer_stats = generator.generate_answers(
#             qa_with_context_file,
#             qa_with_answers_file,
#             use_cot=True  # 使用Chain of Thought方式
#         )
        
#         logger.info(f"答案生成完成: {answer_stats}")
        
#         # 读取带答案的QA数据
#         with open(qa_with_answers_file, 'r', encoding='utf-8') as f:
#             qa_with_answers = json.load(f)
        
#         # 保存最终的高质量QA结果（包含答案）
#         qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
#         with open(qa_output_file, 'w', encoding='utf-8') as f:
#             json.dump(qa_with_answers, f, ensure_ascii=False, indent=2)
        
#         qa_results = qa_with_answers
        
#     except Exception as e:
#         logger.error(f"QA生成失败: {e}")
#         import traceback
#         logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        
#         qa_results = []
#         qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
#         with open(qa_output_file, 'w', encoding='utf-8') as f:
#             json.dump([], f)
        # 步骤2.4: 答案生成（保持原有逻辑）
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

        # 生成答案
        qa_with_answers_file = os.path.join(qa_original_dir, "qa_with_answers.json")
        answer_stats = generator.generate_answers(
            qa_with_context_file,
            qa_with_answers_file,
            use_cot=True  # 使用Chain of Thought方式
        )

        logger.info(f"答案生成完成: {answer_stats}")

        # 读取带答案的QA数据
        with open(qa_with_answers_file, 'r', encoding='utf-8') as f:
            qa_with_answers = json.load(f)

        # 保存最终的高质量QA结果（包含答案）
        qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
        with open(qa_output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_with_answers, f, ensure_ascii=False, indent=2)

        qa_results = qa_with_answers

    except Exception as e:
        logger.error(f"QA生成失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")

        qa_results = []
        qa_output_file = os.path.join(qa_results_dir, "qa_generated.json")
        with open(qa_output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

    # 其余代码保持不变...
    # ===== 第三阶段：数据增强 =====
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
        #enhanced_data = await argument_processor.enhance_qa_data(high_quality_for_enhancement)enhance_qa_data_with_quality_driven_strategy
        enhanced_data = await argument_processor.enhance_qa_data_with_quality_driven_strategy(high_quality_for_enhancement)
    else:
        logger.warning("未找到qa_original数据，使用当前qa_results")
        enhanced_data = await argument_processor.enhance_qa_data_with_quality_driven_strategy(qa_results)
    
    # 保存最终结果
    final_output_file = os.path.join(qa_results_dir, "final_qa_dataset.json")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据增强完成，最终数据集保存至: {final_output_file}")
    
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


# =============================================================================
# 新增函数：分类问题生成相关
# =============================================================================

async def generate_classified_questions(generator, input_data: list, config: dict):
    """生成带分类的问题
    
    Args:
        generator: QA生成器实例
        input_data: 输入数据列表
        config: 配置字典
        
    Returns:
        list: 分类问题数据
    """
    
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
                "与传统的IGZO薄膜相比，为什么SiNx覆盖下的IGZO薄膜其电阻率降低，而SiOx覆盖下的IGZO薄膜其电阻率反而升高呢？",
                "在转移特性曲线中漏极电压升高时漏记电流会突然减低，这是什么原因？"
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
                # 计算该类型应生成的问题数量（基于总问题数3个）
                num_questions = max(1, int(6 * type_info["ratio"]))
                
                type_questions = await generate_questions_by_type(
                    generator, 
                    data_item["md_content"], 
                    question_type,
                    type_info,
                    num_questions,
                    config
                )
                
                item_questions["questions"][question_type] = type_questions
            
            classified_questions.append(item_questions)
            logger.info(f"为 {data_item['paper_name']} 生成了分类问题")
            
        except Exception as e:
            logger.error(f"为 {data_item['paper_name']} 生成分类问题失败: {e}")
            continue
    
    return classified_questions


# async def generate_questions_by_type(generator:SemiconductorQAGenerator, content: str, question_type: str, type_info: dict, num_questions: int, config: dict):
async def generate_questions_by_type(
    generator: SemiconductorQAGenerator, 
    content: str, 
    question_type: str, 
    type_info: dict, 
    num_questions: int,  # 期望生成的问题数量
    config: dict
):
    """为特定类型生成问题
    
    Args:
        generator: QA生成器实例
        content: 文本内容
        question_type: 问题类型 (factual, comparative, reasoning, open)
        type_info: 类型信息(包含description和examples)
        num_questions: 期望生成的问题数量
        config: 配置字典
    """
#     try:
#         # 调用修改后的模型生成函数
#         response = await call_model_for_question_generation(generator, content, question_type, type_info, config)
        
#         # 解析生成的问题
#         questions = parse_generated_questions(response, question_type)
        
#         logger.info(f"生成{question_type}类型问题: 期望{num_questions}个，实际生成{len(questions)}个")
        
#         return questions[:num_questions]  # 确保不超过预期数量
        
#     except Exception as e:
#         logger.error(f"生成{question_type}类型问题失败: {e}")
#         return []
#     try:
#             # 调用修改后的模型生成函数，返回问题列表
#             question_list = await call_model_for_question_generation(
#                 generator, content, question_type, type_info, config
#             )

#             # 记录实际生成数量
#             actual_count = len(question_list)
#             logger.info(f"生成{question_type}类型问题: 期望{num_questions}个，实际生成{actual_count}个")

#             return question_list[:num_questions]  # 确保不超过预期数量
#     except Exception as e:
#         logger.error(f"生成{question_type}类型问题失败: {e}")
#         return []
    try:
        # 调用模型生成函数 - 确保传递所有必需参数
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
        
        # 优先选择长度适中的问题（避免过长或过短）  #有优化点
        selected_questions = []
        for question in all_questions:
            # 确保问题是字符串
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
        
        return selected_questions[:num_questions]  # 确保不超过预期数量
        
    except Exception as e:
        logger.error(f"生成{question_type}类型问题失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return []
# async def call_model_for_question_generation(generator:SemiconductorQAGenerator, content: str, question_type: str, type_info: dict, config: dict):
#     """调用模型生成问题的实际实现
    
#     使用SemiconductorQAGenerator的generate_question_data方法，通过add_prompt参数传递问题类型信息
#     """
#     import tempfile
    
#     try:
#         # 添加调试：输入信息
#         logger.info(f"=== 开始生成 {question_type} 类型问题 ===")
#         logger.info(f"输入内容长度: {len(content)} 字符")
#         logger.info(f"内容预览: {content[:200]}...")
#         # 构建额外的提示信息
#         add_prompt = f"""
# ### 特定问题类型要求：
# 本次需要生成的是【{question_type.upper()}】类型的问题。

# **类型说明**：{type_info['description']}

# **参考示例**：
# {chr(10).join(f"{i+1}. {example}" for i, example in enumerate(type_info['examples'][:3]))}

# **生成要求**：
# 1. 生成2-3个符合【{question_type}】类型特征的问题
# 2. 问题要体现{type_info['description']}的特点
# 3. 确保问题的专业性和技术深度
# 4. 问题必须基于给定的学术内容，有明确的答案依据
# """     
#         #调试
#         logger.debug(f"额外提示内容: {add_prompt}")
#         # 创建临时输入文件
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_input:
#             temp_input_path = temp_input.name
#             # 写入适合generate_question_data的格式
#             temp_data = {
#                 "paper_name": f"classified_{question_type}_generation",
#                 "paper_content": content,
#                 "stats": 1  # 标记为适合生成问题的文本
#             }
#             temp_input.write(json.dumps(temp_data, ensure_ascii=False) + '\n')
        
#         # 创建临时输出文件
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_output:
#             temp_output_path = temp_output.name
#         # generator = SemiconductorQAGenerator(
#         #     batch_size=2,
#         #     gpu_devices="0,1"
#         #     # 不传递model_name参数，因为__init__不接受这个参数
#         # )
#         try:
#             # 调用generate_question_data方法，传入add_prompt
#             print(f"Generator type: {type(generator)}")
#             print(f"Has generate_question_data: {hasattr(generator, 'generate_question_data')}")
#             print(f"Method type: {type(getattr(generator, 'generate_question_data', None))}")
#             #stats = generator.generate_question_data(temp_input_path, temp_output_path, add_prompt=add_prompt)
#             stats = generator.generate_question_data(temp_input_path, temp_output_path, add_prompt=add_prompt)
#                         # 添加调试：最终结果
#             logger.info(f"=== {question_type} 类型问题生成完成 ===")
#             logger.info(f"共生成 {len(generated_questions)} 个问题")
#             # 直接返回问题列表而不是拼接字符串
#             return generated_questions
# #             # 添加调试：查看生成统计
# #             logger.info(f"生成统计: {stats}")
# #             # 读取生成的结果
# #             generated_questions = []
# #             if os.path.exists(temp_output_path):
# #                 with open(temp_output_path, 'r', encoding='utf-8') as f:
# #                     for line in f:
# #                         try:
# #                             result_data = json.loads(line)
# #                             # 【修复：正确获取问题列表】
# #                             question_list = result_data.get('question_list', [])
# #                             if question_list:  # 确保问题列表不为空
# #                                 generated_questions.extend(question_list)
# #                             else:
# #                                 logger.warning(f"No questions found in result: {result_data.keys()}")
# #                         except Exception as e:
# #                             logger.error(f"解析输出结果失败: {e}")
# #                             logger.error(f"Problem line: {line}")
# #             else:
# #                 logger.error(f"Output file not found: {temp_output_path}")

# #             # 添加调试：最终结果
# #             logger.info(f"=== {question_type} 类型问题生成完成 ===")
# #             logger.info(f"共生成 {len(generated_questions)} 个问题")
# #             # 将问题列表转换为文本格式
# #             if generated_questions:
# #                 response = '\n'.join(generated_questions)
# #             else:
# #                 response = ""
                
#         finally:
#             # 清理临时文件
#             try:
#                 os.unlink(temp_input_path)
#                 os.unlink(temp_output_path)
#             except:
#                 pass
            
#             # 恢复原始模板
#             # The original code had this line, but it's not directly related to the prompt_template change.
#             # Keeping it as is, as it might be part of a larger context.
#             # if original_template is not None:
#             #     generator.prompt_template = original_template
        
#         return response
        
#     except Exception as e:
#         logger.error(f"模型调用失败: {e}")
#         import traceback
#         logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
#         return ""

async def call_model_for_question_generation(generator: SemiconductorQAGenerator, content: str, question_type: str, type_info: dict, config: dict):
    """调用模型生成问题的实际实现
    
    使用SemiconductorQAGenerator的generate_question_data方法，通过add_prompt参数传递问题类型信息
    """
    import tempfile
    import os
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:  # 添加 try 块开始
        logger.info(f"=== 开始生成 {question_type} 类型问题 ===")
        logger.info(f"输入内容长度: {len(content)} 字符")
        logger.info(f"内容预览: {content[:200]}...")
        
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
        logger.debug(f"额外提示内容: {add_prompt}")
        
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
    
    except Exception as e:  # 添加 except 块结束 try
        logger.error(f"模型调用失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return []

    
def parse_generated_questions(response: str, question_type: str):
    """解析模型生成的问题"""
    if not response:
        return []
    
    # 按行分割并清理
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    questions = []
    for line in lines:
        # 【修复：直接返回问题文本，而不是字典格式】
        # 因为 generate_question_data 期望的是字符串列表
        cleaned_line = line.lstrip('0123456789.- ').strip()
        if cleaned_line and ('?' in cleaned_line or '？' in cleaned_line):
            questions.append(cleaned_line)  # 直接添加问题文本
    
    return questions

def generate_pipeline_stats(processed_results, qualified_texts, qa_results, enhanced_data, 
                          input_dir, output_dir, chunks_dir, qa_original_dir, model_name, quality_threshold):
    """生成流水线统计信息"""
    
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
            "total_input_texts": len(processed_results),
            "qualified_after_judgment": len(qualified_texts),
            "qualification_rate": f"{len(qualified_texts)/len(processed_results)*100:.1f}%" if processed_results else "0%",
            "total_qa_generated": len(qa_results),
            "total_qa_enhanced": len(enhanced_data)
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



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="半导体QA生成系统")
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
    parser.add_argument("--enable-full-steps", action="store_true",
                        help="启用完整7步骤流程（默认为精简3步骤）")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（可选，用于vLLM HTTP等配置）")
    
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
            
            # 从配置文件中获取处理参数（如果命令行没有指定，则使用配置文件的值）
            if args.batch_size == 32 and 'processing' in config:  # 使用默认值时才从配置文件读取
                args.batch_size = config['processing'].get('batch_size', args.batch_size)
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    # 运行异步流程
    asyncio.run(run_complete_pipeline(
        config=config,  # 第一个参数
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        gpu_devices=args.gpu_devices,
        enable_full_steps=args.enable_full_steps
    ))


if __name__ == "__main__":
    main()