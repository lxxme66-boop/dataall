#!/usr/bin/env python3
"""
DeepSeek QA生成系统 - 基于semiconductor_qa流程
保持原有的处理流程，使用DeepSeek-R1模型替代火山API
支持多线程并行处理
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 导入DeepSeek API客户端
from deepseek_api_client import DeepSeekAPIClient, DeepSeekQAProcessor

# 导入现有模块
try:
    from text_processor import TextProcessor
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    TEXT_PROCESSOR_AVAILABLE = False
    print("警告: text_processor模块不可用")

try:
    from semiconductor_qa_generator import SemiconductorQAGenerator
    QA_GENERATOR_AVAILABLE = True
except ImportError:
    QA_GENERATOR_AVAILABLE = False
    print("警告: semiconductor_qa_generator模块不可用")

try:
    from argument_data import ArgumentDataProcessor
    ARGUMENT_DATA_AVAILABLE = True
except ImportError:
    ARGUMENT_DATA_AVAILABLE = False
    print("警告: argument_data模块不可用")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepSeekSemiconductorQAProcessor:
    """使用DeepSeek API的半导体QA处理器，保持原有流程"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-r1",
        max_workers: int = 16,
        batch_size: int = 10
    ):
        """
        初始化处理器
        
        Args:
            api_key: DeepSeek API密钥
            model: 使用的模型
            max_workers: 最大并发工作数
            batch_size: 批处理大小
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.api_client = None
        self.qa_processor = None
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_files': 0,
            'total_chunks': 0,
            'total_questions': 0,
            'total_qa_pairs': 0,
            'step_times': {}
        }
    
    async def initialize(self):
        """初始化API客户端和处理器"""
        self.api_client = DeepSeekAPIClient(
            api_key=self.api_key,
            model=self.model,
            max_concurrent_calls=self.max_workers
        )
        
        self.qa_processor = DeepSeekQAProcessor(
            api_client=self.api_client,
            max_workers=self.max_workers,
            batch_size=self.batch_size
        )
        
        logger.info(f"初始化DeepSeek API客户端 (模型: {self.model})")
        return self
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        await self.api_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.api_client:
            await self.api_client.__aexit__(exc_type, exc_val, exc_tb)
    
    # ========== 步骤 1.1: 文本预处理 ==========
    async def step_1_1_text_preprocessing(
        self,
        input_dir: str,
        max_files: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        步骤1.1: 文本预处理
        读取和预处理文本文件，分块处理
        """
        logger.info("=== 步骤 1.1: 文本预处理 ===")
        step_start = time.time()
        
        input_path = Path(input_dir)
        text_files = list(input_path.glob("*.txt"))[:max_files]
        
        if not text_files:
            logger.error(f"在 {input_dir} 中未找到文本文件")
            return []
        
        logger.info(f"找到 {len(text_files)} 个文本文件")
        self.stats['total_files'] = len(text_files)
        
        all_chunks = []
        
        # 使用线程池并行读取文件
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(text_files))) as executor:
            futures = {}
            for file_path in text_files:
                future = executor.submit(self._read_and_chunk_file, file_path)
                futures[future] = file_path
            
            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="读取文件"):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    file_path = futures[future]
                    logger.error(f"处理文件 {file_path} 失败: {e}")
        
        self.stats['total_chunks'] = len(all_chunks)
        self.stats['step_times']['1.1_preprocessing'] = time.time() - step_start
        
        logger.info(f"生成了 {len(all_chunks)} 个文本块")
        return all_chunks
    
    def _read_and_chunk_file(self, file_path: Path, chunk_size: int = 2000) -> List[Dict[str, Any]]:
        """读取文件并分块"""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分块处理
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                if len(chunk.strip()) > 100:  # 忽略太短的块
                    chunks.append({
                        'file': str(file_path),
                        'chunk_id': i // chunk_size,
                        'content': chunk,
                        'start_pos': i,
                        'end_pos': min(i + chunk_size, len(content))
                    })
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {e}")
        
        return chunks
    
    # ========== 步骤 1.2: 文本召回与批量推理 ==========
    async def step_1_2_batch_inference(
        self,
        chunks: List[Dict[str, Any]],
        questions_per_chunk: int = 5
    ) -> List[Dict[str, Any]]:
        """
        步骤1.2: 文本召回与批量推理
        使用DeepSeek API生成问题
        """
        logger.info("=== 步骤 1.2: 文本召回与批量推理 ===")
        step_start = time.time()
        
        all_questions = []
        
        # 分批处理chunks
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            logger.info(f"处理批次 {i//self.batch_size + 1}/{(len(chunks) + self.batch_size - 1)//self.batch_size}")
            
            # 并行生成问题
            tasks = []
            for chunk in batch:
                task = self._generate_questions_for_chunk(chunk, questions_per_chunk)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集结果
            for chunk, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"生成问题失败: {result}")
                    continue
                
                for q in result:
                    all_questions.append({
                        'question': q,
                        'source_file': chunk['file'],
                        'chunk_id': chunk['chunk_id'],
                        'context': chunk['content']
                    })
        
        self.stats['total_questions'] = len(all_questions)
        self.stats['step_times']['1.2_batch_inference'] = time.time() - step_start
        
        logger.info(f"生成了 {len(all_questions)} 个问题")
        return all_questions
    
    async def _generate_questions_for_chunk(
        self,
        chunk: Dict[str, Any],
        num_questions: int
    ) -> List[str]:
        """为单个文本块生成问题"""
        prompt = f"""请基于以下文本内容生成{num_questions}个高质量的问题。

要求：
1. 问题要有深度和价值
2. 问题类型多样化
3. 问题必须能从文本中找到答案
4. 只返回问题列表，每行一个问题

文本内容：
{chunk['content'][:2000]}

问题列表："""
        
        try:
            response = await self.api_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            # 解析问题列表
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # 移除编号
                    import re
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if line:
                        questions.append(line)
            
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"生成问题失败: {e}")
            return []
    
    # ========== 步骤 1.3: 数据清洗 ==========
    async def step_1_3_data_cleaning(
        self,
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        步骤1.3: 数据清洗
        清洗和过滤低质量问题
        """
        logger.info("=== 步骤 1.3: 数据清洗 ===")
        step_start = time.time()
        
        cleaned_questions = []
        
        for q in questions:
            # 基本清洗规则
            question_text = q['question'].strip()
            
            # 过滤规则
            if len(question_text) < 10:  # 太短
                continue
            if len(question_text) > 500:  # 太长
                continue
            if question_text.count('?') > 3:  # 太多问号
                continue
            if any(word in question_text.lower() for word in ['请问', '以下', '上述', '如下']):
                # 移除这些词
                for word in ['请问', '以下', '上述', '如下']:
                    question_text = question_text.replace(word, '')
                question_text = question_text.strip()
            
            if question_text:
                q['question'] = question_text
                cleaned_questions.append(q)
        
        self.stats['step_times']['1.3_data_cleaning'] = time.time() - step_start
        
        logger.info(f"清洗后保留 {len(cleaned_questions)} 个问题")
        return cleaned_questions
    
    # ========== 步骤 1.4: 核心QA生成 ==========
    async def step_1_4_qa_generation(
        self,
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        步骤1.4: 核心QA生成
        为问题生成答案
        """
        logger.info("=== 步骤 1.4: 核心QA生成 ===")
        step_start = time.time()
        
        qa_pairs = []
        
        # 分批处理问题
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i + self.batch_size]
            logger.info(f"生成答案批次 {i//self.batch_size + 1}/{(len(questions) + self.batch_size - 1)//self.batch_size}")
            
            # 并行生成答案
            tasks = []
            for q in batch:
                task = self._generate_answer(q['question'], q['context'])
                tasks.append(task)
            
            batch_answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 组合QA对
            for q, answer in zip(batch, batch_answers):
                if isinstance(answer, Exception):
                    logger.error(f"生成答案失败: {answer}")
                    continue
                
                if answer and len(answer.strip()) > 10:
                    qa_pairs.append({
                        'question': q['question'],
                        'answer': answer.strip(),
                        'source_file': q['source_file'],
                        'chunk_id': q['chunk_id'],
                        'metadata': {
                            'model': self.model,
                            'timestamp': time.time()
                        }
                    })
        
        self.stats['total_qa_pairs'] = len(qa_pairs)
        self.stats['step_times']['1.4_qa_generation'] = time.time() - step_start
        
        logger.info(f"生成了 {len(qa_pairs)} 个完整的QA对")
        return qa_pairs
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """生成答案"""
        prompt = f"""请基于以下上下文回答问题。

上下文：
{context[:2000]}

问题：{question}

要求：
1. 答案要准确、详细
2. 如果上下文中没有足够信息，请明确说明
3. 答案要有条理，可以分点说明

答案："""
        
        try:
            answer = await self.api_client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000
            )
            return answer.strip()
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return ""
    
    # ========== 步骤 1.5: 质量检查 ==========
    async def step_1_5_quality_check(
        self,
        qa_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        步骤1.5: 质量检查
        检查QA对的质量
        """
        logger.info("=== 步骤 1.5: 质量检查 ===")
        step_start = time.time()
        
        high_quality_pairs = []
        
        for qa in qa_pairs:
            # 基本质量检查
            q_len = len(qa['question'])
            a_len = len(qa['answer'])
            
            # 质量标准
            if q_len < 10 or q_len > 500:
                continue
            if a_len < 20 or a_len > 2000:
                continue
            if qa['question'].lower() in qa['answer'].lower():
                # 答案不应该重复问题
                continue
            
            # 计算质量分数（简化版）
            quality_score = 0
            if 20 <= q_len <= 100:
                quality_score += 1
            if 50 <= a_len <= 500:
                quality_score += 1
            if '？' in qa['question'] or '?' in qa['question']:
                quality_score += 1
            
            qa['quality_score'] = quality_score
            
            if quality_score >= 2:
                high_quality_pairs.append(qa)
        
        self.stats['step_times']['1.5_quality_check'] = time.time() - step_start
        
        logger.info(f"质量检查后保留 {len(high_quality_pairs)} 个高质量QA对")
        return high_quality_pairs
    
    # ========== 步骤 1.6: 数据增强与重写 ==========
    async def step_1_6_data_augmentation(
        self,
        qa_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        步骤1.6: 数据增强与重写
        增强QA对的多样性
        """
        logger.info("=== 步骤 1.6: 数据增强与重写 ===")
        step_start = time.time()
        
        augmented_pairs = []
        
        # 选择部分QA对进行增强
        pairs_to_augment = qa_pairs[:min(len(qa_pairs) // 2, 100)]
        
        for qa in pairs_to_augment:
            # 生成变体问题
            variants = await self._generate_question_variants(qa['question'])
            
            for variant in variants:
                if variant and variant != qa['question']:
                    augmented_pairs.append({
                        'question': variant,
                        'answer': qa['answer'],
                        'source_file': qa['source_file'],
                        'chunk_id': qa['chunk_id'],
                        'metadata': {
                            **qa.get('metadata', {}),
                            'augmented': True,
                            'original_question': qa['question']
                        }
                    })
        
        # 合并原始和增强的QA对
        all_pairs = qa_pairs + augmented_pairs
        
        self.stats['step_times']['1.6_augmentation'] = time.time() - step_start
        
        logger.info(f"数据增强后共有 {len(all_pairs)} 个QA对")
        return all_pairs
    
    async def _generate_question_variants(self, question: str) -> List[str]:
        """生成问题变体"""
        prompt = f"""请为以下问题生成2个不同表述但意思相同的变体：

原始问题：{question}

要求：
1. 保持原意不变
2. 使用不同的表述方式
3. 每个变体独立一行

变体问题："""
        
        try:
            response = await self.api_client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=200
            )
            
            variants = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # 移除编号
                    import re
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if line:
                        variants.append(line)
            
            return variants[:2]
            
        except Exception as e:
            logger.error(f"生成问题变体失败: {e}")
            return []
    
    # ========== 步骤 1.7: 最终输出整理 ==========
    async def step_1_7_final_output(
        self,
        qa_pairs: List[Dict[str, Any]],
        output_dir: str
    ) -> str:
        """
        步骤1.7: 最终输出整理
        保存结果和生成统计报告
        """
        logger.info("=== 步骤 1.7: 最终输出整理 ===")
        step_start = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存QA数据
        timestamp = int(time.time())
        output_file = output_path / f"deepseek_qa_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        self.stats['end_time'] = time.time()
        self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
        self.stats['step_times']['1.7_final_output'] = time.time() - step_start
        
        stats_file = output_path / f"deepseek_qa_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        self._print_statistics()
        
        logger.info(f"结果已保存到: {output_file}")
        return str(output_file)
    
    def _print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("DeepSeek QA生成完成！")
        print("="*60)
        print(f"处理文件数: {self.stats['total_files']}")
        print(f"生成文本块: {self.stats['total_chunks']}")
        print(f"生成问题数: {self.stats['total_questions']}")
        print(f"完整QA对数: {self.stats['total_qa_pairs']}")
        print(f"总耗时: {self.stats.get('total_time', 0):.2f} 秒")
        
        if self.stats.get('total_time', 0) > 0:
            print(f"平均速度: {self.stats['total_qa_pairs'] / self.stats['total_time']:.2f} QA/秒")
        
        print("\n各步骤耗时:")
        for step, time_spent in self.stats.get('step_times', {}).items():
            print(f"  {step}: {time_spent:.2f} 秒")
        
        print("="*60)
    
    async def run_complete_pipeline(
        self,
        input_dir: str,
        output_dir: str,
        max_files: int = 5000,
        target_qa_count: int = 20000
    ):
        """
        运行完整的QA生成流程
        
        Args:
            input_dir: 输入文本目录
            output_dir: 输出目录
            max_files: 最大处理文件数
            target_qa_count: 目标QA数量
        """
        self.stats['start_time'] = time.time()
        
        logger.info("=== 开始DeepSeek QA生成流程 ===")
        logger.info(f"使用模型: {self.model}")
        logger.info(f"最大并发数: {self.max_workers}")
        logger.info(f"批处理大小: {self.batch_size}")
        
        # 步骤1.1: 文本预处理
        chunks = await self.step_1_1_text_preprocessing(input_dir, max_files)
        if not chunks:
            logger.error("没有可处理的文本块")
            return
        
        # 计算每个块需要生成的问题数
        questions_per_chunk = max(1, target_qa_count // len(chunks))
        questions_per_chunk = min(questions_per_chunk, 10)  # 限制每块最多10个问题
        
        # 步骤1.2: 文本召回与批量推理
        questions = await self.step_1_2_batch_inference(chunks, questions_per_chunk)
        if not questions:
            logger.error("没有生成任何问题")
            return
        
        # 步骤1.3: 数据清洗
        cleaned_questions = await self.step_1_3_data_cleaning(questions)
        
        # 步骤1.4: 核心QA生成
        qa_pairs = await self.step_1_4_qa_generation(cleaned_questions)
        
        # 步骤1.5: 质量检查
        high_quality_pairs = await self.step_1_5_quality_check(qa_pairs)
        
        # 步骤1.6: 数据增强与重写
        augmented_pairs = await self.step_1_6_data_augmentation(high_quality_pairs)
        
        # 限制到目标数量
        if len(augmented_pairs) > target_qa_count:
            augmented_pairs = augmented_pairs[:target_qa_count]
            logger.info(f"限制QA对数量到 {target_qa_count}")
        
        # 步骤1.7: 最终输出整理
        output_file = await self.step_1_7_final_output(augmented_pairs, output_dir)
        
        return output_file


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DeepSeek QA生成系统")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/texts",
        help="输入文本目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="输出目录"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5000,
        help="最大处理文件数"
    )
    parser.add_argument(
        "--target-qa",
        type=int,
        default=20000,
        help="目标QA数量"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DeepSeek API密钥（也可通过DEEPSEEK_API_KEY环境变量设置）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-r1",
        help="使用的模型名称"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="最大并发工作数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="批处理大小"
    )
    
    args = parser.parse_args()
    
    # 检查API密钥
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 请设置DeepSeek API密钥")
        print("使用 --api-key 参数或设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)
    
    # 创建处理器并运行
    async with DeepSeekSemiconductorQAProcessor(
        api_key=api_key,
        model=args.model,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    ) as processor:
        await processor.run_complete_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_files=args.max_files,
            target_qa_count=args.target_qa
        )


if __name__ == "__main__":
    asyncio.run(main())