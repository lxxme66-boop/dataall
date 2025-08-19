#!/usr/bin/env python3
"""
文本处理器 - 专门处理txt文件
"""

import asyncio
import time
import pickle as pkl
import argparse
import json
import os
import logging
from typing import List, Dict, Optional
from TextGeneration.Datageneration import parse_txt, input_text_process
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import text processing functions
try:
    from TextGeneration.Datageneration import parse_txt, input_text_process
    logger.info("Successfully imported text processing functions")
except ImportError as e:
    logger.error(f"Failed to import text processing functions: {e}")
    raise


class TextProcessor:
    """文本处理器类"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化文本处理器"""
        self.config = self._load_config(config_path)
        self.text_dir = self.config.get('paths', {}).get('text_dir', 'data/texts')
        self.output_dir = self.config.get('paths', {}).get('text_output_dir', 'data/qa_results/texts')
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    async def process_single_txt(self, file_path: str, index: int = 9) -> List[Dict]:
        """处理单个txt文件"""
        logger.info(f"Processing txt file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        if not file_path.endswith('.txt'):
            logger.error(f"Not a txt file: {file_path}")
            return []
        
        try:
            # 使用parse_txt函数处理文件
            tasks = await parse_txt(file_path, index=index)
            
            if not tasks:
                logger.warning(f"No tasks generated for {file_path}")
                return []
            
            # 执行所有任务
            results = await asyncio.gather(*tasks)
            
            # 过滤掉None结果
            results = [r for r in results if r is not None]
            
            logger.info(f"Generated {len(results)} results for {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    async def process_txt_folder(self, folder_path: Optional[str] = None, 
                                batch_size: int = 2,
                                max_files: Optional[int] = None) -> Dict:
        """处理文件夹中的所有txt文件"""
        if folder_path is None:
            folder_path = self.text_dir
        
        logger.info(f"Processing txt files from: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return {"success": False, "error": "Folder not found"}
        
        # 获取所有txt文件
        txt_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        
        if not txt_files:
            logger.warning(f"No txt files found in {folder_path}")
            return {"success": False, "error": "No txt files found"}
        
        logger.info(f"Found {len(txt_files)} txt files")
        
        # 限制处理文件数量
        if max_files:
            txt_files = txt_files[:max_files]
            logger.info(f"Processing only first {max_files} files")
        
        # 批量处理文件
        all_results = []
        for i in range(0, len(txt_files), batch_size):
            batch_files = txt_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_files)} files")
            
            batch_tasks = []
            for file_path in batch_files:
                task = self.process_single_txt(file_path)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            # 展平结果
            for file_results in batch_results:
                all_results.extend(file_results)
        
        # 保存结果
        output_file = os.path.join(self.output_dir, "text_qa_results.json")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        
        # 生成统计信息
        stats = {
            "total_files": len(txt_files),
            "total_results": len(all_results),
            "average_results_per_file": len(all_results) / len(txt_files) if txt_files else 0
        }
        
        return {
            "success": True,
            "stats": stats,
            "output_file": output_file,
            "results": all_results
        }


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文本处理器 - 处理txt文件")
    parser.add_argument("--input", type=str, help="输入txt文件或文件夹路径")
    parser.add_argument("--output", type=str, help="输出目录")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--batch-size", type=int, default=100, help="批处理大小")
    parser.add_argument("--max-files", type=int, help="最大处理文件数")
    parser.add_argument("--index", type=int, default=9, help="提示词索引")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = TextProcessor(args.config)
    
    # 设置输出目录
    if args.output:
        processor.output_dir = args.output
    
    # 处理输入
    if args.input:
        if os.path.isfile(args.input) and args.input.endswith('.txt'):
            # 处理单个文件
            results = await processor.process_single_txt(args.input, index=args.index)
            print(f"处理完成: 生成 {len(results)} 个结果")
        elif os.path.isdir(args.input):
            # 处理文件夹
            result = await processor.process_txt_folder(
                args.input, 
                batch_size=args.batch_size,
                max_files=args.max_files
            )
            if result['success']:
                print(f"处理完成: {result['stats']}")
            else:
                print(f"处理失败: {result['error']}")
        else:
            print(f"错误: 输入必须是txt文件或文件夹")
    else:
        # 使用默认文本目录
        result = await processor.process_txt_folder(
            batch_size=args.batch_size,
            max_files=args.max_files
        )
        if result['success']:
            print(f"处理完成: {result['stats']}")
        else:
            print(f"处理失败: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())