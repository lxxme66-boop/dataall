#!/usr/bin/env python3
"""
大规模QA生成脚本
支持处理2万+条数据和5000个文本文件
集成火山API调用Qwen3-235B-A22B-Instruct模型
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

# 导入增强的并行处理器
from enhanced_parallel_processor import (
    EnhancedParallelProcessor,
    VolcanoAPIClient,
    LargeScaleQAProcessor
)

# 导入现有模块（复用已有功能）
try:
    from semiconductor_qa_generator import SemiconductorQAGenerator
    from text_processor import TextProcessor
    from argument_data import ArgumentDataProcessor
    HAS_EXISTING_MODULES = True
except ImportError:
    HAS_EXISTING_MODULES = False
    print("警告: 某些现有模块不可用，将使用简化版本")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedLargeScaleProcessor:
    """集成的大规模处理器 - 结合新旧功能"""
    
    def __init__(self, config_path: str = "config_volcano.json"):
        """
        初始化集成处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 初始化并行处理器
        parallel_config = self.config['parallel_processing']
        self.parallel_processor = EnhancedParallelProcessor(
            max_workers=parallel_config['max_workers'],
            batch_size=parallel_config['batch_size'],
            use_multiprocessing=parallel_config['use_multiprocessing'],
            max_concurrent_api_calls=parallel_config['max_concurrent_api_calls'],
            api_timeout=parallel_config['api_timeout']
        )
        
        # 初始化API客户端
        self.api_client = None
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_files_processed': 0,
            'total_chunks_created': 0,
            'total_questions_generated': 0,
            'total_qa_pairs': 0,
            'failed_items': []
        }
    
    async def initialize_api_client(self):
        """初始化API客户端"""
        api_config = self.config['api']
        
        # 尝试从环境变量获取API密钥
        api_key = os.environ.get('VOLCANO_API_KEY') or api_config.get('volcano_api_key')
        endpoint_id = os.environ.get('VOLCANO_ENDPOINT_ID') or api_config.get('volcano_endpoint_id')
        
        self.api_client = VolcanoAPIClient(
            api_key=api_key,
            endpoint_id=endpoint_id,
            region=api_config.get('volcano_region', 'cn-beijing')
        )
        
        # 测试连接
        if api_key:
            logger.info("使用火山API (Qwen3-235B-A22B-Instruct)")
        else:
            logger.info("未配置火山API密钥，将使用本地vLLM服务")
        
        return self.api_client
    
    async def process_large_scale_texts(
        self,
        input_dir: str,
        output_dir: str,
        max_files: int = 5000,
        target_qa_count: int = 20000
    ):
        """
        处理大规模文本文件
        
        Args:
            input_dir: 输入文本目录
            output_dir: 输出目录
            max_files: 最大处理文件数
            target_qa_count: 目标QA数量
        """
        self.stats['start_time'] = time.time()
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取文本文件
        input_path = Path(input_dir)
        text_files = list(input_path.glob("*.txt"))[:max_files]
        
        if not text_files:
            logger.error(f"在 {input_dir} 中未找到文本文件")
            return
        
        logger.info(f"找到 {len(text_files)} 个文本文件，开始处理...")
        self.stats['total_files_processed'] = len(text_files)
        
        # 初始化API客户端
        async with await self.initialize_api_client() as api_client:
            # 创建处理器
            processor = LargeScaleQAProcessor(
                parallel_processor=self.parallel_processor,
                api_client=api_client,
                output_dir=output_dir
            )
            
            # 分批处理文件（避免内存溢出）
            batch_size = 100  # 每批处理100个文件
            all_qa_pairs = []
            
            for i in range(0, len(text_files), batch_size):
                batch_files = text_files[i:i+batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}/{(len(text_files) + batch_size - 1)//batch_size}")
                
                # 处理当前批次
                batch_stats = await processor.process_texts(
                    text_files=[str(f) for f in batch_files],
                    target_qa_count=target_qa_count // ((len(text_files) + batch_size - 1) // batch_size)
                )
                
                # 更新统计
                self.stats['total_chunks_created'] += batch_stats['total_chunks']
                self.stats['total_questions_generated'] += batch_stats['total_questions']
                self.stats['total_qa_pairs'] += batch_stats['total_qa_pairs']
                
                # 读取生成的QA对
                with open(batch_stats['output_file'], 'r', encoding='utf-8') as f:
                    batch_qa = json.load(f)
                    all_qa_pairs.extend(batch_qa)
                
                # 检查是否已达到目标数量
                if len(all_qa_pairs) >= target_qa_count:
                    logger.info(f"已达到目标QA数量 {target_qa_count}")
                    all_qa_pairs = all_qa_pairs[:target_qa_count]
                    break
            
            # 保存最终结果
            final_output = output_path / f"final_qa_dataset_{int(time.time())}.json"
            with open(final_output, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            
            self.stats['end_time'] = time.time()
            self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
            
            # 保存统计信息
            stats_file = output_path / "processing_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            
            # 打印最终统计
            self._print_final_stats()
            
            return all_qa_pairs
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        print("\n" + "="*60)
        print("大规模QA生成完成！")
        print("="*60)
        print(f"处理文件数: {self.stats['total_files_processed']}")
        print(f"生成文本块: {self.stats['total_chunks_created']}")
        print(f"生成问题数: {self.stats['total_questions_generated']}")
        print(f"完整QA对数: {self.stats['total_qa_pairs']}")
        print(f"总耗时: {self.stats['total_time']:.2f} 秒")
        print(f"平均速度: {self.stats['total_qa_pairs'] / self.stats['total_time']:.2f} QA/秒")
        print("="*60)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="大规模QA生成系统")
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
        "--config",
        type=str,
        default="config_volcano.json",
        help="配置文件路径"
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
        "--use-volcano",
        action="store_true",
        help="使用火山API（需要设置VOLCANO_API_KEY环境变量）"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量（如果指定使用火山API）
    if args.use_volcano:
        if not os.environ.get('VOLCANO_API_KEY'):
            print("错误: 使用火山API需要设置VOLCANO_API_KEY环境变量")
            print("请运行: export VOLCANO_API_KEY='your_api_key'")
            sys.exit(1)
    
    # 创建处理器
    processor = IntegratedLargeScaleProcessor(config_path=args.config)
    
    # 运行处理
    await processor.process_large_scale_texts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        target_qa_count=args.target_qa
    )


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())