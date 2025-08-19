#!/usr/bin/env python3
"""
增强文件处理器 - 智能识别和处理PDF/TXT文件
支持从不同目录加载不同类型的文件
"""
import re
import os
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

# Import text filtering functions
from TextGeneration.text_filter import drop, is_to_drop

logger = logging.getLogger(__name__)
async def process_text_chunk(task: dict):
    """处理单个文本块 - 核心文本处理逻辑"""
    try:
        logger.info(f"处理文本块: {task['file_path']} 分块{task['chunk_index']}")
        
        # 1. 获取文本内容
        content = task.get("content", "")
        
        # 2. 基础文本清洗
        cleaned_text = clean_text(content)
        
        # 3. 关键信息提取
        key_phrases = extract_key_phrases(cleaned_text)
        
        # 4. 文本规范化
        normalized_text = normalize_text(cleaned_text)
        
        # 5. 返回处理结果
        return {
            "file_path": task["file_path"],
            "chunk_index": task["chunk_index"],
            "original_length": len(content),
            "processed_text": normalized_text,
            "key_phrases": key_phrases,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"处理文本块失败: {task.get('file_path', '未知文件')} 分块{task.get('chunk_index', '?')} - {str(e)}")
        return {
            "file_path": task.get("file_path", ""),
            "chunk_index": task.get("chunk_index", -1),
            "error": str(e),
            "success": False
        }

def clean_text(text: str) -> str:
    """基础文本清洗"""
    # 移除多余空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 移除特殊字符（保留常见标点）
    text = re.sub(r'[^\w\s,.?!;:\-()\[\]{}]', '', text)
    
    # 标准化引号
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    
    return text

def extract_key_phrases(text: str) -> list:
    """提取关键短语（简化版）"""
    # 实际应用中应使用更复杂的方法
    words = text.split()
    if len(words) > 10:
        # 提取名词和专有名词
        return [word for word in words if word.istitle() or word.isupper()][:5]
    return []

def normalize_text(text: str) -> str:
    """文本规范化"""
    # 确保首字母大写
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    
    # 确保句末有标点
    if text and text[-1] not in {'.', '!', '?', ';', ':'}:
        text += '.'
    
    return text
class EnhancedFileProcessor:
    """
    增强型文件处理器
    支持从不同目录加载不同类型的文件
    """
    
    def __init__(self, config_path: str = "config.json"):
        """初始化处理器"""
        self.config = self._load_config(config_path)
        self.pdf_dir = self.config.get('paths', {}).get('pdf_dir', 'data/pdfs')
        self.text_dir = self.config.get('paths', {}).get('text_dir', 'data/texts')
        self.output_dir = self.config.get('paths', {}).get('output_dir', 'data/output')
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    async def process_pdf_file(self, file_path: str) -> Dict:
        """处理PDF文件"""
        logger.info(f"Processing PDF: {file_path}")
        
        # 这里可以集成PDF处理逻辑
        # 暂时返回模拟数据
        return {
            'file_path': file_path,
            'file_type': 'pdf',
            'content': f"PDF content from {file_path}",
            'metadata': {
                'pages': 10,
                'processed_at': datetime.now().isoformat()
            }
        }
    
    # async def process_txt_file(self, file_path: str) -> Dict:
    #     """处理文本文件"""
    #     logger.info(f"Processing text file: {file_path}")
        
    #     try:
    #         with open(file_path, 'r', encoding='utf-8') as f:
    #             content = f.read()
            
    #         # 应用文本过滤
    #         filtered_content = drop(content)
            
    #         # 记录过滤信息
    #         original_lines = content.count('\n') + 1
    #         filtered_lines = filtered_content.count('\n') + 1
    #         filter_ratio = 1 - (filtered_lines / original_lines) if original_lines > 0 else 0
            
    #         return {
    #             'file_path': file_path,
    #             'file_type': 'txt',
    #             'content': filtered_content,
    #             'metadata': {
    #                 'original_length': len(content),
    #                 'filtered_length': len(filtered_content),
    #                 'original_lines': original_lines,
    #                 'filtered_lines': filtered_lines,
    #                 'filter_ratio': filter_ratio,
    #                 'processed_at': datetime.now().isoformat()
    #             }
    #         }
    #     except Exception as e:
    #         logger.error(f"Error processing {file_path}: {e}")
    #         return {
    #             'file_path': file_path,
    #             'file_type': 'txt',
    #             'content': '',
    #             'error': str(e)
    #         }
    async def process_txt_file(file_path: str, prompt_index: int, config: dict) -> List[asyncio.Task]:
        """处理单个文本文件并返回任务列表"""
        logger.info(f"处理文本文件: {file_path}")
        
        # 确保文件存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return []
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"文件读取完成，大小: {len(content)/1024:.1f}KB")
            
            # 分块处理
            chunks = extract_text_chunks(content)
            logger.info(f"共生成 {len(chunks)} 个文本分块")
            
            # 创建任务
            tasks = []
            source_file = os.path.basename(file_path)
            
            for i, chunk in enumerate(chunks):
                task = asyncio.create_task(
                    input_text_process(
                        chunk, 
                        source_file,
                        chunk_index=i,
                        total_chunks=len(chunks),
                        prompt_index=prompt_index,
                        config=config
                    )
                )
                tasks.append(task)
            
            logger.info(f"为文件 {source_file} 创建了 {len(tasks)} 个处理任务")
            return tasks
        
        except Exception as e:
            logger.error(f"处理文件失败: {file_path} - {e}")
            logger.exception("详细错误:")
            return []
        
    def get_files_from_directory(self, directory: str, extensions: List[str]) -> List[str]:
        """从目录获取指定扩展名的文件"""
        files = []
        
        if not os.path.exists(directory):
            logger.warning(f"目录不存在: {directory}")
            return files
        
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
        
        return files
    
    async def process_pdf_directory(self, pdf_dir: Optional[str] = None) -> List[Dict]:
        """处理PDF目录"""
        if pdf_dir is None:
            pdf_dir = self.pdf_dir
        
        pdf_files = self.get_files_from_directory(pdf_dir, ['.pdf'])
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        tasks = []
        for pdf_file in pdf_files:
            task = self.process_pdf_file(pdf_file)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def process_text_directory(self, text_dir: Optional[str] = None) -> List[Dict]:
        """处理文本目录"""
        if text_dir is None:
            text_dir = self.text_dir
        
        txt_files = self.get_files_from_directory(text_dir, ['.txt'])
        logger.info(f"Found {len(txt_files)} text files in {text_dir}")
        
        tasks = []
        for txt_file in txt_files:
            task = self.process_txt_file(txt_file)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def prepare_for_retrieval(self, pdf_results: List[Dict], txt_results: List[Dict]) -> List[Dict]:
        """准备用于检索的数据"""
        all_results = []
        
        # 处理PDF结果
        for result in pdf_results:
            if 'error' not in result:
                all_results.append({
                    'source_file': result['file_path'],
                    'file_type': 'pdf',
                    'content': result['content'],
                    'metadata': result.get('metadata', {})
                })
        
        # 处理文本结果
        for result in txt_results:
            if 'error' not in result:
                all_results.append({
                    'source_file': result['file_path'],
                    'file_type': 'txt',
                    'content': result['content'],
                    'metadata': result.get('metadata', {})
                })
        
        logger.info(f"Prepared {len(all_results)} documents for retrieval")
        return all_results
    
    async def process_directory(self, directory: str) -> Tuple[List[Dict], List[Dict]]:
        """
        处理目录，智能识别PDF和文本文件
        
        Args:
            directory: 要处理的目录路径
            
        Returns:
            Tuple[pdf_results, txt_results]
        """
        pdf_results = []
        txt_results = []
        
        # 如果是PDF目录
        if 'pdf' in directory.lower():
            # 处理PDF目录
            pdf_results = await self.process_pdf_directory(directory)
            
            # 检查是否有对应的文本目录
            text_dir = directory.replace('pdf', 'text').replace('PDF', 'text')
            if os.path.exists(text_dir):
                logger.info(f"Also found text directory: {text_dir}")
                txt_results = await self.process_text_directory(text_dir)
        
        # 如果是文本目录
        elif 'text' in directory.lower() or 'txt' in directory.lower():
            # 处理文本目录
            txt_results = await self.process_text_directory(directory)
            
            # 检查是否有对应的PDF目录
            pdf_dir = directory.replace('text', 'pdf').replace('txt', 'pdf')
            if os.path.exists(pdf_dir):
                logger.info(f"Also found PDF directory: {pdf_dir}")
                pdf_results = await self.process_pdf_directory(pdf_dir)
        
        # 通用目录处理
        else:
            # 同时检查PDF和文本文件
            pdf_files = self.get_files_from_directory(directory, ['.pdf'])
            txt_files = self.get_files_from_directory(directory, ['.txt'])
            
            if pdf_files:
                logger.info(f"Found {len(pdf_files)} PDF files")
                # 检查PDF目录
                pdf_subdir = os.path.join(directory, 'pdfs')
                if os.path.exists(pdf_subdir):
                    pdf_results = await self.process_pdf_directory(pdf_subdir)
                else:
                    pdf_results = await self.process_pdf_directory(directory)
            
            if txt_files:
                logger.info(f"Found {len(txt_files)} text files")
                # 检查文本目录
                text_subdir = os.path.join(directory, 'texts')
                if os.path.exists(text_subdir):
                    txt_results = await self.process_text_directory(text_subdir)
                else:
                    txt_results = await self.process_text_directory(directory)
        
        return pdf_results, txt_results


# 示例用法
async def main():
    processor = EnhancedFileProcessor()
    
    # 处理特定目录
    pdf_results, txt_results = await processor.process_directory("/workspace/data")
    
    # 准备检索数据
    retrieval_data = processor.prepare_for_retrieval(pdf_results, txt_results)
    
    # 保存结果
    output_file = os.path.join(processor.output_dir, "processed_data.json")
    os.makedirs(processor.output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(pdf_results)} PDF files and {len(txt_results)} text files")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试目录
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/texts", exist_ok=True)
    
    # 运行示例
    asyncio.run(main())