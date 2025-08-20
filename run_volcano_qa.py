#!/usr/bin/env python3
"""
半导体QA生成系统 - 火山API版本
使用火山引擎API (deepseek-r1) 进行大规模QA生成
支持多线程处理，目标：处理4000个文档，生成20000条数据
"""

import os
import sys
import json
import time
import logging
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import pickle

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入必要的模块
from TextGeneration.Datageneration import parse_txt, input_text_process, merge_chunk_responses
from LocalModels.volcano_client import VolcanoAPIClient, VolcanoModelManager
from semiconductor_qa_generator import run_semiconductor_qa_generation
from enhanced_file_processor import process_text_chunk

# 尝试导入可选模块
try:
    from argument_data import ArgumentDataProcessor
    ARGUMENT_DATA_AVAILABLE = True
except ImportError:
    ARGUMENT_DATA_AVAILABLE = False
    
    class ArgumentDataProcessor:
        """Mock ArgumentDataProcessor class"""
        def __init__(self):
            pass
        
        async def process_qa_data(self, *args, **kwargs):
            return args[0] if args else []

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volcano_qa_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VolcanoQAProcessor:
    """火山API QA处理器 - 支持多线程大规模处理"""
    
    def __init__(self, config_path: str = "config_volcano.json"):
        """初始化处理器"""
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 初始化火山API客户端
        self.volcano_client = VolcanoAPIClient(self.config)
        self.model_manager = VolcanoModelManager(self.config)
        
        # 处理配置
        self.parallel_config = self.config.get('parallel_processing', {})
        self.large_scale_config = self.config.get('large_scale_processing', {})
        
        # 多线程配置
        self.max_workers = self.parallel_config.get('max_workers', 16)
        self.batch_size = self.parallel_config.get('batch_size', 100)
        
        # 大规模处理配置
        self.target_qa_count = self.large_scale_config.get('target_qa_count', 20000)
        self.max_text_files = self.large_scale_config.get('max_text_files', 4000)
        self.chunk_size = self.large_scale_config.get('chunk_size', 2000)
        self.questions_per_chunk = self.large_scale_config.get('questions_per_chunk', 5)
        
        # 统计信息
        self.stats = {
            'start_time': time.time(),
            'processed_files': 0,
            'processed_chunks': 0,
            'generated_qa_pairs': 0,
            'failed_chunks': 0,
            'total_tokens': 0
        }
        
        # 进度跟踪
        self.progress_lock = threading.Lock()
        
        logger.info(f"Volcano QA处理器初始化完成")
        logger.info(f"目标: 处理 {self.max_text_files} 个文档, 生成 {self.target_qa_count} 条QA数据")
        logger.info(f"并行配置: {self.max_workers} 个工作线程, 批大小 {self.batch_size}")
    
    def process_single_file(self, file_path: Path) -> List[Dict]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            生成的QA列表
        """
        try:
            logger.info(f"处理文件: {file_path}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 文本分块
            chunks = self._split_text(content)
            
            # 处理每个块
            qa_list = []
            for chunk in chunks:
                qa_pairs = self._process_chunk(chunk)
                qa_list.extend(qa_pairs)
            
            with self.progress_lock:
                self.stats['processed_files'] += 1
                self.stats['processed_chunks'] += len(chunks)
                self.stats['generated_qa_pairs'] += len(qa_list)
            
            logger.info(f"文件 {file_path.name} 处理完成，生成 {len(qa_list)} 个QA对")
            return qa_list
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")
            with self.progress_lock:
                self.stats['failed_chunks'] += 1
            return []
    
    def _split_text(self, text: str) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 原始文本
            
        Returns:
            文本块列表
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _process_chunk(self, chunk: str) -> List[Dict]:
        """
        处理单个文本块，生成QA
        
        Args:
            chunk: 文本块
            
        Returns:
            QA对列表
        """
        qa_pairs = []
        
        try:
            # Step 1: 文本理解和优化
            optimized_text = self._optimize_text(chunk)
            
            # Step 2: 生成问题
            questions = self._generate_questions(optimized_text)
            
            # Step 3: 生成答案
            for question in questions:
                answer = self._generate_answer(optimized_text, question)
                if answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'context': optimized_text[:500],  # 保存部分上下文
                        'timestamp': datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"处理文本块失败: {e}")
        
        return qa_pairs
    
    def _optimize_text(self, text: str) -> str:
        """使用AI优化文本"""
        prompt = f"""请优化以下半导体技术文本，使其更加清晰和专业：

文本：
{text[:1500]}

要求：
1. 保持技术准确性
2. 改善表达清晰度
3. 保留所有重要信息

优化后的文本："""
        
        result = self.volcano_client.generate(prompt, temperature=0.3, max_tokens=2000)
        return result if result else text
    
    def _generate_questions(self, text: str) -> List[str]:
        """生成问题列表"""
        prompt = f"""基于以下半导体技术文本，生成{self.questions_per_chunk}个高质量的问题。

文本：
{text[:1500]}

要求：
1. 问题类型多样化（事实型、比较型、推理型、开放型）
2. 问题要有深度和专业性
3. 每个问题独立成行

问题列表："""
        
        result = self.volcano_client.generate(prompt, temperature=0.7, max_tokens=1000)
        
        if result:
            # 解析问题列表
            questions = []
            for line in result.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # 移除编号
                    if line[0].isdigit() and '.' in line[:3]:
                        line = line.split('.', 1)[1].strip()
                    if line:
                        questions.append(line)
            return questions[:self.questions_per_chunk]
        
        return []
    
    def _generate_answer(self, context: str, question: str) -> str:
        """生成答案"""
        prompt = f"""基于以下半导体技术文本，回答问题。使用Chain of Thought方式，提供详细专业的答案。

文本：
{context[:1500]}

问题：{question}

请按以下格式回答：
1. 首先理解问题的核心
2. 分析相关的技术要点
3. 给出详细的答案
4. 总结关键信息

答案："""
        
        result = self.volcano_client.generate(prompt, temperature=0.5, max_tokens=1500)
        return result if result else ""
    
    def process_files_parallel(self, input_dir: str, output_dir: str):
        """
        并行处理多个文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有文本文件
        input_path = Path(input_dir)
        text_files = list(input_path.glob("**/*.txt"))[:self.max_text_files]
        
        logger.info(f"找到 {len(text_files)} 个文本文件")
        
        # 结果收集
        all_qa_pairs = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path
                for file_path in text_files
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    qa_pairs = future.result()
                    all_qa_pairs.extend(qa_pairs)
                    completed += 1
                    
                    # 打印进度
                    if completed % 10 == 0:
                        self._print_progress(completed, len(text_files))
                    
                    # 检查是否达到目标
                    if len(all_qa_pairs) >= self.target_qa_count:
                        logger.info(f"已达到目标QA数量: {len(all_qa_pairs)}")
                        break
                        
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时出错: {e}")
        
        # 保存结果
        self._save_results(all_qa_pairs, output_path)
        
        # 打印最终统计
        self._print_final_stats()
    
    def _print_progress(self, completed: int, total: int):
        """打印进度"""
        with self.progress_lock:
            progress = (completed / total) * 100
            logger.info(f"进度: {completed}/{total} ({progress:.1f}%) - "
                       f"已生成 {self.stats['generated_qa_pairs']} 个QA对")
    
    def _save_results(self, qa_pairs: List[Dict], output_path: Path):
        """保存结果"""
        # 保存为JSON
        json_path = output_path / "volcano_qa_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {json_path}")
        
        # 保存统计信息
        stats_path = output_path / "volcano_processing_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已保存到: {stats_path}")
        
        # 保存为pickle（用于后续处理）
        pickle_path = output_path / "volcano_qa_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(qa_pairs, f)
        logger.info(f"Pickle文件已保存到: {pickle_path}")
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        elapsed_time = time.time() - self.stats['start_time']
        
        logger.info("=" * 50)
        logger.info("处理完成 - 最终统计:")
        logger.info(f"  处理文件数: {self.stats['processed_files']}")
        logger.info(f"  处理文本块数: {self.stats['processed_chunks']}")
        logger.info(f"  生成QA对数: {self.stats['generated_qa_pairs']}")
        logger.info(f"  失败块数: {self.stats['failed_chunks']}")
        logger.info(f"  总耗时: {elapsed_time:.2f} 秒")
        logger.info(f"  平均速度: {self.stats['generated_qa_pairs'] / elapsed_time:.2f} QA/秒")
        
        # 获取API统计
        api_stats = self.volcano_client.get_stats()
        logger.info(f"  API调用次数: {api_stats['total_requests']}")
        logger.info(f"  API成功率: {api_stats['success_rate']:.2%}")
        logger.info(f"  总Token数: {api_stats['total_tokens']}")
        logger.info("=" * 50)
    
    async def process_with_enhancement(self, input_dir: str, output_dir: str):
        """
        带数据增强的处理流程
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        # Step 1: 并行处理文件生成QA
        logger.info("Step 1: 生成原始QA数据...")
        self.process_files_parallel(input_dir, output_dir)
        
        # Step 2: 数据增强（如果可用）
        if ARGUMENT_DATA_AVAILABLE:
            logger.info("Step 2: 执行数据增强...")
            
            # 加载生成的QA数据
            qa_path = Path(output_dir) / "volcano_qa_results.json"
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            # 执行增强
            processor = ArgumentDataProcessor()
            enhanced_data = await processor.process_qa_data(qa_data)
            
            # 保存增强结果
            enhanced_path = Path(output_dir) / "volcano_qa_enhanced.json"
            with open(enhanced_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"增强后的数据已保存到: {enhanced_path}")
        else:
            logger.info("数据增强模块不可用，跳过增强步骤")
    
    def close(self):
        """关闭处理器"""
        self.volcano_client.close()
        logger.info("Volcano QA处理器已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用火山API生成半导体QA数据集")
    parser.add_argument("--input-dir", type=str, default="data/texts",
                       help="输入文本目录")
    parser.add_argument("--output-dir", type=str, default="data/volcano_output",
                       help="输出目录")
    parser.add_argument("--config", type=str, default="config_volcano.json",
                       help="配置文件路径")
    parser.add_argument("--enable-enhancement", action="store_true",
                       help="启用数据增强")
    parser.add_argument("--max-files", type=int, default=4000,
                       help="最大处理文件数")
    parser.add_argument("--target-qa", type=int, default=20000,
                       help="目标QA数量")
    parser.add_argument("--workers", type=int, default=16,
                       help="并行工作线程数")
    
    args = parser.parse_args()
    
    # 更新配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 更新配置参数
    if 'large_scale_processing' not in config:
        config['large_scale_processing'] = {}
    config['large_scale_processing']['max_text_files'] = args.max_files
    config['large_scale_processing']['target_qa_count'] = args.target_qa
    
    if 'parallel_processing' not in config:
        config['parallel_processing'] = {}
    config['parallel_processing']['max_workers'] = args.workers
    
    # 保存更新后的配置
    temp_config_path = "temp_volcano_config.json"
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建处理器
    processor = VolcanoQAProcessor(temp_config_path)
    
    try:
        if args.enable_enhancement:
            # 异步运行带增强的流程
            asyncio.run(processor.process_with_enhancement(args.input_dir, args.output_dir))
        else:
            # 运行基础流程
            processor.process_files_parallel(args.input_dir, args.output_dir)
    finally:
        processor.close()
        # 清理临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    main()