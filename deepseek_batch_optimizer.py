"""
DeepSeek-R1 批量推理优化器
通过批量处理和优化策略显著降低API调用成本
"""

import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib
import pickle
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 批量处理设置
    batch_size: int = 10  # 每批处理的文档数
    questions_per_batch: int = 40  # 每批生成的问题数
    
    # Token优化
    max_context_length: int = 500  # 减少上下文长度
    max_output_tokens: int = 800  # 限制输出长度
    
    # 缓存设置
    enable_cache: bool = True
    cache_dir: str = "cache/deepseek"
    
    # API设置
    api_key: str = ""
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    model_name: str = "deepseek-r1"
    temperature: float = 0.3  # 降低温度减少随机性
    
    # 成本优化
    enable_compression: bool = True  # 启用提示词压缩
    enable_deduplication: bool = True  # 启用去重
    reuse_context: bool = True  # 复用上下文


class DeepSeekBatchOptimizer:
    """DeepSeek批量推理优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, content: str, operation: str) -> str:
        """生成缓存键"""
        return hashlib.md5(f"{operation}:{content}".encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存加载结果"""
        if not self.config.enable_cache:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """保存结果到缓存"""
        if not self.config.enable_cache:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def compress_prompt(self, text: str) -> str:
        """压缩提示词，减少Token消耗"""
        if not self.config.enable_compression:
            return text
            
        # 移除多余空白
        text = ' '.join(text.split())
        
        # 截断过长文本
        if len(text) > self.config.max_context_length * 4:  # 估算字符数
            text = text[:self.config.max_context_length * 4]
            
        return text
    
    async def batch_generate_qa(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量生成问答对
        
        Args:
            documents: 文档列表，每个包含 'id' 和 'content'
            
        Returns:
            问答对列表
        """
        all_qa_pairs = []
        
        # 按批次处理文档
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i:i + self.config.batch_size]
            
            # 1. 批量生成问题（1次API调用替代10次）
            questions = await self._batch_generate_questions(batch_docs)
            
            # 2. 批量生成答案（1次API调用替代10次）
            qa_pairs = await self._batch_generate_answers(batch_docs, questions)
            
            all_qa_pairs.extend(qa_pairs)
            
            logger.info(f"处理批次 {i//self.config.batch_size + 1}, "
                       f"生成 {len(qa_pairs)} 个QA对, "
                       f"API调用: {self.api_calls}")
        
        return all_qa_pairs
    
    async def _batch_generate_questions(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """批量生成问题 - 单次API调用"""
        
        # 检查缓存
        cache_key = self._get_cache_key(str([d['id'] for d in documents]), "questions")
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info("使用缓存的问题")
            return cached_result
        
        # 构建批量提示词
        batch_prompt = self._build_batch_question_prompt(documents)
        
        # 调用API
        response = await self._call_deepseek_api(batch_prompt)
        self.api_calls += 1
        
        # 解析结果
        questions = self._parse_batch_questions(response, documents)
        
        # 保存缓存
        self._save_to_cache(cache_key, questions)
        
        return questions
    
    def _build_batch_question_prompt(self, documents: List[Dict[str, str]]) -> str:
        """构建批量问题生成提示词"""
        prompt_parts = [
            "批量生成问题任务。对以下每个文档生成4个高质量问题。",
            "输出格式：JSON数组，每个元素包含doc_id和questions。",
            "示例：[{\"doc_id\":\"1\",\"questions\":[\"Q1\",\"Q2\",\"Q3\",\"Q4\"]}]",
            "\n文档列表："
        ]
        
        for doc in documents:
            content = self.compress_prompt(doc['content'])
            prompt_parts.append(f"\nID:{doc['id']}\n内容:{content[:self.config.max_context_length]}")
        
        prompt_parts.append("\n请生成JSON格式的问题列表：")
        
        return '\n'.join(prompt_parts)
    
    async def _batch_generate_answers(self, documents: List[Dict[str, str]], 
                                     questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量生成答案 - 单次API调用"""
        
        # 构建批量提示词
        batch_prompt = self._build_batch_answer_prompt(documents, questions)
        
        # 调用API
        response = await self._call_deepseek_api(batch_prompt)
        self.api_calls += 1
        
        # 解析结果
        qa_pairs = self._parse_batch_answers(response, questions)
        
        return qa_pairs
    
    def _build_batch_answer_prompt(self, documents: List[Dict[str, str]], 
                                  questions: List[Dict[str, Any]]) -> str:
        """构建批量答案生成提示词"""
        prompt_parts = [
            "批量回答问题任务。基于文档内容回答所有问题。",
            "输出格式：JSON数组，每个元素包含question和answer。",
            "要求：答案简洁准确，每个答案不超过100字。",
            "\n问答任务："
        ]
        
        # 创建文档索引
        doc_dict = {doc['id']: doc['content'][:self.config.max_context_length] 
                   for doc in documents}
        
        qa_tasks = []
        for q_item in questions:
            doc_id = q_item['doc_id']
            context = self.compress_prompt(doc_dict.get(doc_id, ""))
            
            for question in q_item['questions']:
                qa_tasks.append({
                    "question": question,
                    "context": context[:200]  # 进一步压缩上下文
                })
        
        # 批量构建提示
        prompt_parts.append(json.dumps(qa_tasks, ensure_ascii=False))
        prompt_parts.append("\n请生成JSON格式的答案列表：")
        
        return '\n'.join(prompt_parts)
    
    async def _call_deepseek_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": "你是一个高效的问答生成助手。请严格按照JSON格式输出。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens
        }
        
        try:
            async with self.session.post(
                self.config.api_url,
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                
                # 统计Token使用
                if 'usage' in result:
                    self.total_input_tokens += result['usage'].get('prompt_tokens', 0)
                    self.total_output_tokens += result['usage'].get('completion_tokens', 0)
                
                return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return "{}"
    
    def _parse_batch_questions(self, response: str, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """解析批量问题响应"""
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            logger.error("解析问题失败")
        
        # 降级处理：为每个文档生成默认问题
        return [{"doc_id": doc['id'], "questions": ["这段文本的主要内容是什么？"]} 
                for doc in documents]
    
    def _parse_batch_answers(self, response: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """解析批量答案响应"""
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            logger.error("解析答案失败")
        
        # 降级处理
        qa_pairs = []
        for q_item in questions:
            for question in q_item['questions']:
                qa_pairs.append({
                    "question": question,
                    "answer": "需要根据上下文具体分析。"
                })
        return qa_pairs
    
    def calculate_cost(self) -> Dict[str, float]:
        """计算成本"""
        input_cost = (self.total_input_tokens / 1_000_000) * 4  # 4元/百万tokens
        output_cost = (self.total_output_tokens / 1_000_000) * 16  # 16元/百万tokens
        total_cost = input_cost + output_cost
        
        return {
            "api_calls": self.api_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost_rmb": input_cost,
            "output_cost_rmb": output_cost,
            "total_cost_rmb": total_cost,
            "cost_reduction": f"{(1 - self.api_calls/30000) * 100:.1f}%"  # 相比原方案的成本降低
        }


async def optimize_qa_generation(
    input_dir: str,
    output_dir: str,
    target_qa_count: int = 20000,
    max_files: int = 5000
):
    """
    优化的QA生成主函数
    
    预期优化效果：
    - API调用次数：从30,000次降至约1,000次（95%+减少）
    - Token消耗：通过压缩和批处理减少50-70%
    - 总成本：从1,500元降至约300-500元
    """
    
    # 配置优化参数
    config = OptimizationConfig(
        batch_size=10,  # 10个文档一批
        questions_per_batch=40,  # 每批40个问题
        max_context_length=500,  # 压缩上下文
        max_output_tokens=800,  # 限制输出
        enable_cache=True,
        enable_compression=True,
        api_key=os.environ.get("DEEPSEEK_API_KEY", "")
    )
    
    # 读取文档
    input_path = Path(input_dir)
    text_files = list(input_path.glob("*.txt"))[:max_files]
    
    documents = []
    for idx, file_path in enumerate(text_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                "id": str(idx),
                "content": content,
                "file": str(file_path)
            })
    
    logger.info(f"加载了 {len(documents)} 个文档")
    
    # 使用优化器生成QA
    async with DeepSeekBatchOptimizer(config) as optimizer:
        qa_pairs = await optimizer.batch_generate_qa(documents)
        
        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"optimized_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs[:target_qa_count], f, ensure_ascii=False, indent=2)
        
        # 输出成本统计
        cost_stats = optimizer.calculate_cost()
        logger.info("=" * 50)
        logger.info("优化后的成本统计：")
        logger.info(f"API调用次数: {cost_stats['api_calls']} (原方案: 30,000)")
        logger.info(f"输入Tokens: {cost_stats['input_tokens']:,}")
        logger.info(f"输出Tokens: {cost_stats['output_tokens']:,}")
        logger.info(f"输入成本: ¥{cost_stats['input_cost_rmb']:.2f}")
        logger.info(f"输出成本: ¥{cost_stats['output_cost_rmb']:.2f}")
        logger.info(f"总成本: ¥{cost_stats['total_cost_rmb']:.2f} (原方案: ¥1,500)")
        logger.info(f"成本降低: {cost_stats['cost_reduction']}")
        logger.info("=" * 50)
        
        # 保存成本报告
        report_file = output_path / f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(cost_stats, f, ensure_ascii=False, indent=2)
        
        return qa_pairs, cost_stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python deepseek_batch_optimizer.py <input_dir> <output_dir>")
        sys.exit(1)
    
    asyncio.run(optimize_qa_generation(
        input_dir=sys.argv[1],
        output_dir=sys.argv[2],
        target_qa_count=20000,
        max_files=5000
    ))