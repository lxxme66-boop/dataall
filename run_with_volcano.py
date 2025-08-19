#!/usr/bin/env python3
"""
使用火山引擎API运行QA生成系统
支持多线程优化和大规模数据处理
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_qa_generator import (
    OptimizedQAGenerator,
    VolcanoConfig,
    OptimizationConfig
)

# 设置日志
def setup_logging(log_file: str = None):
    """设置日志配置"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def load_config(config_file: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_text_chunks(input_dir: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """准备文本块"""
    chunks = []
    chunk_id = 0
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 文本分块
                    for i in range(0, len(content), chunk_size - chunk_overlap):
                        chunk_text = content[i:i + chunk_size]
                        if len(chunk_text.strip()) > 100:  # 忽略太短的块
                            chunks.append({
                                'id': chunk_id,
                                'file': file,
                                'file_path': file_path,
                                'text': chunk_text,
                                'start_pos': i,
                                'end_pos': i + len(chunk_text)
                            })
                            chunk_id += 1
                    
                    logging.info(f"文件 {file} 分割为 {len([c for c in chunks if c['file'] == file])} 个块")
                    
                except Exception as e:
                    logging.error(f"处理文件 {file_path} 失败: {e}")
    
    return chunks

async def process_with_volcano(
    chunks: List[Dict[str, Any]],
    volcano_config: VolcanoConfig,
    opt_config: OptimizationConfig,
    output_dir: str
) -> Dict[str, Any]:
    """使用火山引擎API处理文本块"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 初始化生成器
    generator = OptimizedQAGenerator(volcano_config, opt_config)
    
    # 按文件分组处理
    file_chunks = {}
    for chunk in chunks:
        file_name = chunk['file']
        if file_name not in file_chunks:
            file_chunks[file_name] = []
        file_chunks[file_name].append(chunk)
    
    all_results = []
    file_stats = {}
    
    for file_name, file_chunk_list in file_chunks.items():
        logging.info(f"开始处理文件: {file_name} ({len(file_chunk_list)} 个块)")
        
        # 提取文本内容
        texts = [chunk['text'] for chunk in file_chunk_list]
        
        # 批处理
        if opt_config.use_async:
            results = await generator.process_batch_async(texts)
        else:
            results = generator.process_batch_sync(texts)
        
        # 合并元数据
        for i, result in enumerate(results):
            result['source_file'] = file_name
            result['chunk_info'] = {
                'id': file_chunk_list[i]['id'],
                'start_pos': file_chunk_list[i]['start_pos'],
                'end_pos': file_chunk_list[i]['end_pos']
            }
            all_results.append(result)
        
        # 保存单个文件的结果
        file_output = os.path.join(output_dir, f"{Path(file_name).stem}_qa_{timestamp}.json")
        with open(file_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 统计信息
        file_stats[file_name] = {
            'chunks': len(file_chunk_list),
            'qa_pairs': sum(len(r.get('qa_pairs', [])) for r in results),
            'successful': len([r for r in results if not r.get('error')]),
            'failed': len([r for r in results if r.get('error')])
        }
        
        logging.info(f"文件 {file_name} 处理完成: {file_stats[file_name]}")
    
    # 保存汇总结果
    summary = {
        'timestamp': timestamp,
        'total_chunks': len(chunks),
        'total_files': len(file_chunks),
        'total_qa_pairs': sum(len(r.get('qa_pairs', [])) for r in all_results),
        'file_stats': file_stats,
        'generator_stats': generator.stats
    }
    
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 保存所有QA对
    all_qa_file = os.path.join(output_dir, f"all_qa_pairs_{timestamp}.json")
    all_qa_pairs = []
    for result in all_results:
        for qa in result.get('qa_pairs', []):
            qa['source_file'] = result.get('source_file')
            all_qa_pairs.append(qa)
    
    with open(all_qa_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
    
    # 清理资源
    generator.cleanup()
    
    return summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用火山引擎API生成QA数据集")
    
    # 基本参数
    parser.add_argument("--config", default="config_volcano.json", help="配置文件路径")
    parser.add_argument("--input-dir", help="输入文本目录（覆盖配置文件）")
    parser.add_argument("--output-dir", help="输出目录（覆盖配置文件）")
    
    # 火山引擎API参数
    parser.add_argument("--api-key", help="火山引擎API密钥（覆盖配置文件）")
    parser.add_argument("--endpoint-id", help="火山引擎端点ID（覆盖配置文件）")
    parser.add_argument("--model", help="模型名称（覆盖配置文件）")
    
    # 优化参数
    parser.add_argument("--max-workers", type=int, help="最大并发数")
    parser.add_argument("--batch-size", type=int, help="批处理大小")
    parser.add_argument("--use-async", action="store_true", help="使用异步模式")
    parser.add_argument("--use-multiprocess", action="store_true", help="使用多进程")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    
    # 处理参数
    parser.add_argument("--chunk-size", type=int, help="文本分块大小")
    parser.add_argument("--chunk-overlap", type=int, help="分块重叠大小")
    parser.add_argument("--quality-threshold", type=float, help="质量阈值")
    
    # 其他参数
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要处理的文件")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.input_dir:
        config['paths']['input_dir'] = args.input_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.api_key:
        config['volcano_api']['api_key'] = args.api_key
    if args.endpoint_id:
        config['volcano_api']['endpoint_id'] = args.endpoint_id
    if args.model:
        config['volcano_api']['model'] = args.model
    
    # 设置日志
    log_file = config.get('logging', {}).get('file')
    logger = setup_logging(log_file)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查API密钥
    if config['volcano_api']['api_key'] == "YOUR_VOLCANO_API_KEY":
        logger.error("请在配置文件中设置火山引擎API密钥，或使用 --api-key 参数")
        sys.exit(1)
    
    # 准备文本块
    input_dir = config['paths']['input_dir']
    chunk_size = args.chunk_size or config['optimization']['chunk_size']
    chunk_overlap = args.chunk_overlap or config['optimization']['chunk_overlap']
    
    logger.info(f"从 {input_dir} 加载文本文件...")
    chunks = prepare_text_chunks(input_dir, chunk_size, chunk_overlap)
    
    if not chunks:
        logger.error(f"在 {input_dir} 中没有找到文本文件")
        sys.exit(1)
    
    logger.info(f"准备了 {len(chunks)} 个文本块")
    
    if args.dry_run:
        # 仅显示统计信息
        file_stats = {}
        for chunk in chunks:
            file_name = chunk['file']
            if file_name not in file_stats:
                file_stats[file_name] = 0
            file_stats[file_name] += 1
        
        logger.info("将要处理的文件：")
        for file_name, count in file_stats.items():
            logger.info(f"  - {file_name}: {count} 个块")
        return
    
    # 创建配置对象
    volcano_config = VolcanoConfig(
        api_key=config['volcano_api']['api_key'],
        endpoint_id=config['volcano_api']['endpoint_id'],
        region=config['volcano_api'].get('region', 'cn-beijing'),
        model=config['volcano_api'].get('model', 'doubao-pro-32k'),
        max_tokens=config['model_params']['max_tokens'],
        temperature=config['model_params']['temperature'],
        top_p=config['model_params']['top_p'],
        timeout=config['volcano_api']['timeout']
    )
    
    opt_config = OptimizationConfig(
        max_workers=args.max_workers or config['optimization']['max_workers'],
        use_multiprocess=args.use_multiprocess or config['optimization']['use_multiprocess'],
        use_async=args.use_async or config['optimization']['use_async'],
        batch_size=args.batch_size or config['optimization']['batch_size'],
        queue_size=config['optimization']['queue_size'],
        enable_cache=not args.no_cache and config['optimization']['enable_cache'],
        cache_dir=config['optimization']['cache_dir'],
        rate_limit=config['optimization']['rate_limit'],
        rate_limit_window=config['optimization']['rate_limit_window'],
        max_retries=config['volcano_api']['max_retries'],
        retry_delay=config['volcano_api']['retry_delay']
    )
    
    # 运行处理
    output_dir = config['paths']['output_dir']
    logger.info(f"开始处理，结果将保存到 {output_dir}")
    
    start_time = time.time()
    
    try:
        # 运行异步处理
        if opt_config.use_async:
            summary = asyncio.run(
                process_with_volcano(chunks, volcano_config, opt_config, output_dir)
            )
        else:
            # 创建生成器并同步处理
            generator = OptimizedQAGenerator(volcano_config, opt_config)
            texts = [chunk['text'] for chunk in chunks]
            results = generator.process_batch_sync(texts)
            
            # 保存结果
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"qa_results_{timestamp}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            summary = {
                'total_chunks': len(chunks),
                'total_qa_pairs': sum(len(r.get('qa_pairs', [])) for r in results),
                'stats': generator.stats
            }
            
            generator.cleanup()
        
        elapsed_time = time.time() - start_time
        
        # 打印最终统计
        logger.info("="*60)
        logger.info("处理完成！")
        logger.info(f"总耗时: {elapsed_time:.2f} 秒")
        logger.info(f"处理块数: {summary.get('total_chunks', 0)}")
        logger.info(f"生成QA对: {summary.get('total_qa_pairs', 0)}")
        if summary.get('total_chunks', 0) > 0:
            logger.info(f"平均速度: {summary['total_chunks']/elapsed_time:.2f} 块/秒")
        logger.info(f"结果保存在: {output_dir}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()