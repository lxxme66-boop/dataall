# -*- coding: utf-8 -*-
"""
增强文档处理器 - 整合智能分块功能
使用RecursiveCharacterTextSplitter进行语义感知的文档分块
"""
import os
import json
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from typing import Literal, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("Warning: langchain not installed. Install with: pip install langchain")
    RecursiveCharacterTextSplitter = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DocType = Literal["txt", "md", "json", "pdf"]


@dataclass
class ChunkConfig:
    """分块配置"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: List[str] = None
    keep_separator: bool = True
    
    def __post_init__(self):
        if self.separators is None:
            # 针对中文和英文的智能分隔符
            self.separators = [
                "\n\n\n",  # 多个空行
                "\n\n",    # 段落分隔
                "\n",      # 行分隔
                "。\n",    # 中文句号+换行
                "。",      # 中文句号
                ".\n",     # 英文句号+换行
                ". ",      # 英文句号+空格
                "；",      # 中文分号
                ";",       # 英文分号
                "，",      # 中文逗号
                ", ",      # 英文逗号
                " ",       # 空格
                ""         # 字符
            ]


@dataclass
class DocumentChunk:
    """文档块数据结构"""
    chunk_id: str
    content: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "chunk_id": self.chunk_id,
            "chunk": self.content,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata
        }


class EnhancedDocumentProcessor:
    """增强文档处理器"""
    
    def __init__(self, chunk_config: Optional[ChunkConfig] = None):
        """
        初始化文档处理器
        
        Args:
            chunk_config: 分块配置
        """
        self.chunk_config = chunk_config or ChunkConfig()
        self._init_text_splitter()
        
        # 统计信息
        self.stats = {
            "total_files_processed": 0,
            "total_chunks_created": 0,
            "total_characters_processed": 0,
            "processing_errors": 0
        }
    
    def _init_text_splitter(self):
        """初始化文本分割器"""
        if RecursiveCharacterTextSplitter is None:
            logger.warning("RecursiveCharacterTextSplitter not available, using simple splitter")
            self.text_splitter = None
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_config.chunk_size,
                chunk_overlap=self.chunk_config.chunk_overlap,
                separators=self.chunk_config.separators,
                keep_separator=self.chunk_config.keep_separator
            )
    
    def read_file_names(self, data_dir: str, ext: str = ".md") -> List[str]:
        """
        读取目录中的文件名
        
        Args:
            data_dir: 数据目录
            ext: 文件扩展名
            
        Returns:
            文件名列表
        """
        if not os.path.exists(data_dir):
            logger.error(f"Directory not found: {data_dir}")
            return []
            
        file_names = []
        for filename in os.listdir(data_dir):
            # 过滤掉临时文件
            if filename.startswith("~$"):
                continue
            if filename.endswith(ext):
                file_names.append(filename)
        
        logger.info(f"Found {len(file_names)} {ext} files in {data_dir}")
        return sorted(file_names)
    
    def _generate_chunk_id(self, source: str, chunk_index: int, content: str) -> str:
        """生成唯一的块ID"""
        # 使用源文件名、块索引和内容的前50个字符生成唯一ID
        id_source = f"{source}_{chunk_index}_{content[:50]}"
        return hashlib.md5(id_source.encode()).hexdigest()
    
    def _simple_chunk(self, text: str, chunk_size: int) -> List[str]:
        """简单的文本分块（备用方案）"""
        chunks = []
        for i in range(0, len(text), chunk_size - self.chunk_config.chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def get_doc_chunks(self, file_path: Path, doctype: DocType = "txt", 
                      chunk_size: Optional[int] = None) -> List[DocumentChunk]:
        """
        获取文档的分块
        
        Args:
            file_path: 文件路径
            doctype: 文档类型
            chunk_size: 分块大小（可选，覆盖默认配置）
            
        Returns:
            文档块列表
        """
        chunk_size = chunk_size or self.chunk_config.chunk_size
        
        try:
            # 读取文件内容
            if doctype == "json":
                with open(file_path, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                text = data.get("text", str(data))
            elif doctype in ["txt", "md"]:
                with open(file_path, 'r', encoding="utf-8") as f:
                    text = f.read()
            elif doctype == "pdf":
                # PDF处理需要额外的库
                logger.warning("PDF processing not implemented yet")
                return []
            else:
                raise ValueError(f"Unsupported document type: {doctype}")
            
            # 更新统计
            self.stats["total_characters_processed"] += len(text)
            
            # 分块处理
            if self.text_splitter:
                # 使用LangChain的智能分块
                chunks = self.text_splitter.create_documents([text])
                chunk_texts = [chunk.page_content for chunk in chunks]
            else:
                # 使用简单分块
                chunk_texts = self._simple_chunk(text, chunk_size)
            
            # 创建DocumentChunk对象
            doc_chunks = []
            total_chunks = len(chunk_texts)
            source_name = file_path.name
            
            for idx, chunk_text in enumerate(chunk_texts):
                chunk_id = self._generate_chunk_id(source_name, idx, chunk_text)
                
                doc_chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source=source_name,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    metadata={
                        "file_path": str(file_path),
                        "doc_type": doctype,
                        "chunk_size": len(chunk_text),
                        "created_at": datetime.now().isoformat()
                    }
                )
                doc_chunks.append(doc_chunk)
            
            self.stats["total_chunks_created"] += len(doc_chunks)
            return doc_chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats["processing_errors"] += 1
            return []
    
    def get_chunks(self, data_path: Path, doctype: DocType = "txt", 
                  chunk_size: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
        """
        获取文件或目录的所有分块
        
        Args:
            data_path: 数据路径（文件或目录）
            doctype: 文档类型
            chunk_size: 分块大小
            
        Returns:
            (分块列表, 文件名)
        """
        chunks = []
        file_paths = []
        
        if data_path.is_dir():
            # 处理目录
            file_paths = list(data_path.rglob(f'**/*.{doctype}'))
            filename = data_path.name
        else:
            # 处理单个文件
            file_paths = [data_path]
            filename = data_path.name
        
        # 使用线程池并行处理
        futures = []
        with tqdm(total=len(file_paths), desc="Processing files", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=4) as executor:
                for file_path in file_paths:
                    future = executor.submit(
                        self.get_doc_chunks, file_path, doctype, chunk_size
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        doc_chunks = future.result()
                        # 转换为字典格式
                        chunk_dicts = [chunk.to_dict() for chunk in doc_chunks]
                        chunks.extend(chunk_dicts)
                        pbar.set_postfix({'chunks': len(chunks)})
                    except Exception as e:
                        logger.error(f"Error in future: {e}")
                    finally:
                        pbar.update(1)
        
        self.stats["total_files_processed"] += len(file_paths)
        return chunks, filename
    
    def build_or_load_chunks(self, datapath: Path, doctype: str, 
                           chunk_size: Optional[int] = None,
                           cache_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        """
        构建或加载分块（支持缓存）
        
        Args:
            datapath: 数据路径
            doctype: 文档类型
            chunk_size: 分块大小
            cache_path: 缓存路径
            
        Returns:
            (分块列表, 文件名)
        """
        # 生成缓存键
        cache_key = f"{datapath}_{doctype}_{chunk_size or self.chunk_config.chunk_size}"
        cache_file = None
        
        if cache_path:
            cache_file = Path(cache_path) / f"{hashlib.md5(cache_key.encode()).hexdigest()}.json"
            
            # 尝试从缓存加载
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    logger.info(f"Loaded {len(cached_data['chunks'])} chunks from cache")
                    return cached_data['chunks'], cached_data['filename']
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
        
        # 构建新的分块
        chunks, filename = self.get_chunks(datapath, doctype, chunk_size)
        
        # 保存到缓存
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'chunks': chunks,
                        'filename': filename,
                        'metadata': {
                            'created_at': datetime.now().isoformat(),
                            'chunk_config': {
                                'chunk_size': chunk_size or self.chunk_config.chunk_size,
                                'chunk_overlap': self.chunk_config.chunk_overlap
                            }
                        }
                    }, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(chunks)} chunks to cache")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
        
        return chunks, filename
    
    def save_chunks(self, chunks: List[Dict[str, Any]], article_name: str, 
                   filename: str, mode: str = "append"):
        """
        保存分块到文件
        
        Args:
            chunks: 分块列表
            article_name: 文章名称
            filename: 输出文件名
            mode: 保存模式 ("append" 或 "overwrite")
        """
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        if mode == "append" and os.path.exists(filename):
            # 追加模式
            try:
                with open(filename, 'r', encoding="utf-8") as f:
                    existing = json.load(f)
            except:
                existing = {}
            
            # 合并数据
            if article_name in existing:
                existing[article_name].extend(chunks)
            else:
                existing[article_name] = chunks
            
            data_to_save = existing
        else:
            # 覆盖模式
            data_to_save = {article_name: chunks}
        
        # 保存数据
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks for '{article_name}' to {filename}")
    
    def process_directory(self, data_dir: str, output_path: str, 
                         doc_type: str = "md", chunk_size: Optional[int] = None,
                         start_idx: Optional[int] = None, end_idx: Optional[int] = None):
        """
        处理整个目录的文档
        
        Args:
            data_dir: 输入目录
            output_path: 输出路径
            doc_type: 文档类型
            chunk_size: 分块大小
            start_idx: 起始索引
            end_idx: 结束索引
        """
        # 检查输出文件是否已存在
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"{output_path} already exists. Loading existing data...")
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                logger.info(f"Found {len(existing_data)} articles in existing file")
            except Exception as e:
                logger.error(f"Error loading existing file: {e}")
                existing_data = {}
        else:
            existing_data = {}
        
        # 获取文件列表
        ext = f".{doc_type}" if not doc_type.startswith(".") else doc_type
        filenames = self.read_file_names(data_dir, ext)
        
        # 应用索引范围
        if start_idx is not None or end_idx is not None:
            start_idx = start_idx or 0
            end_idx = end_idx or len(filenames)
            filenames = filenames[start_idx:end_idx]
            logger.info(f"Processing files from index {start_idx} to {end_idx}")
        
        # 处理每个文件
        total_chunks = 0
        with tqdm(filenames, desc="Processing files") as pbar:
            for filename in pbar:
                # 跳过已处理的文件
                if filename in existing_data:
                    logger.debug(f"Skipping already processed file: {filename}")
                    continue
                
                file_path = Path(data_dir) / filename
                
                try:
                    # 获取文档分块
                    doc_chunks = self.get_doc_chunks(file_path, doc_type, chunk_size)
                    
                    if doc_chunks:
                        # 转换为字典格式
                        chunk_dicts = [chunk.to_dict() for chunk in doc_chunks]
                        
                        # 保存分块
                        self.save_chunks(chunk_dicts, filename, output_path, mode="append")
                        
                        total_chunks += len(chunk_dicts)
                        pbar.set_postfix({'total_chunks': total_chunks})
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    self.stats["processing_errors"] += 1
        
        logger.info(f"Processing completed. Total chunks created: {total_chunks}")
        logger.info(f"Processing statistics: {self.stats}")
        
        return total_chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            **self.stats,
            "chunk_config": {
                "chunk_size": self.chunk_config.chunk_size,
                "chunk_overlap": self.chunk_config.chunk_overlap,
                "separators_count": len(self.chunk_config.separators)
            }
        }


# 便捷函数
def process_documents(data_dir: str, output_path: str, 
                     doc_type: str = "md", chunk_size: int = 512,
                     chunk_overlap: int = 50) -> int:
    """
    处理文档的便捷函数
    
    Args:
        data_dir: 输入目录
        output_path: 输出路径
        doc_type: 文档类型
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        
    Returns:
        创建的分块总数
    """
    # 创建配置
    config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 创建处理器
    processor = EnhancedDocumentProcessor(config)
    
    # 处理目录
    total_chunks = processor.process_directory(
        data_dir=data_dir,
        output_path=output_path,
        doc_type=doc_type
    )
    
    # 打印统计信息
    stats = processor.get_statistics()
    print(f"\nProcessing Statistics:")
    print(f"- Total files processed: {stats['total_files_processed']}")
    print(f"- Total chunks created: {stats['total_chunks_created']}")
    print(f"- Total characters processed: {stats['total_characters_processed']:,}")
    print(f"- Processing errors: {stats['processing_errors']}")
    
    return total_chunks


# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Document Processor with Smart Chunking")
    parser.add_argument("--data_dir", type=str, required=True, help="Input directory containing documents")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for chunks")
    parser.add_argument("--doc_type", type=str, default="md", help="Document type (txt, md, json)")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size in characters")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap between chunks")
    parser.add_argument("--start_idx", type=int, help="Start index for file processing")
    parser.add_argument("--end_idx", type=int, help="End index for file processing")
    
    args = parser.parse_args()
    
    # 处理文档
    total_chunks = process_documents(
        data_dir=args.data_dir,
        output_path=args.output_path,
        doc_type=args.doc_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    print(f"\nTotal chunks created: {total_chunks}")