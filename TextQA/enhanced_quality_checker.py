import asyncio
import json
import pandas as pd
import logging
from openai import AsyncOpenAI
from asyncio import Semaphore, gather
from typing import List, Dict, Any, Optional, Tuple
import re
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedQualityChecker:
    """
    增强版QA质量检查器
    基于checkInfor/checkQuestion.py的双阶段验证思路
    """
    
    def __init__(self, 
                 api_key: str,
                 base_url: str,
                 model: str,
                 system_prompt: str,
                 parallel_core: int = 10,
                 activate_stream: bool = False):
        """
        初始化质量检查器
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型名称
            system_prompt: 系统提示词
            parallel_core: 并发核心数
            activate_stream: 是否使用流式输出
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )
        self.model = model
        self.system_prompt = system_prompt
        self.parallel_core = parallel_core
        self.activate_stream = activate_stream
        self.semaphore = Semaphore(parallel_core)
        
        # 质量检查相关的prompt模板
        self.quality_check_prompts = {
            'answer_generation': """请回答以下问题，返回内容直接给出答案，不需要给出思考过程：

问题：{question}

请提供准确、完整的答案。""",
            
            'answer_verification': """请判断模型回答的答案是否正确。

原始问题：{question}
标准答案：{standard_answer}
模型回答：{model_answer}

判断标准：
1. 如果是事实型问题，回答必须准确无误
2. 如果是分析型问题，回答要求逻辑清晰、要点完整
3. 不能包含与标准答案冲突的错误信息
4. 内容要覆盖问题的所有关键要点

如果回答正确请返回1，如果不正确请返回0
返回内容只能是数字："""
        }
    
    async def get_model_response(self, prompt: str) -> str:
        """
        获取模型响应
        
        Args:
            prompt: 输入提示词
            
        Returns:
            模型的响应内容
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.9,
                stream=self.activate_stream
            )
            
            if self.activate_stream:
                content = ""
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                return content
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"模型响应获取失败: {e}")
            return ""
    
    async def verify_qa_pair(self, qa_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证单个QA对的质量
        
        Args:
            qa_data: 包含question和answer的QA数据
            
        Returns:
            验证结果，包含原始数据和质量标签
        """
        question = qa_data.get('question', '')
        standard_answer = qa_data.get('answer', '')
        
        if not question or not standard_answer:
            logger.warning(f"QA对数据不完整: {qa_data}")
            return {**qa_data, 'quality_label': 0, 'verification_details': '数据不完整'}
        
        try:
            # 第一阶段：让模型回答问题
            answer_prompt = self.quality_check_prompts['answer_generation'].format(
                question=question
            )
            
            async with self.semaphore:
                model_answer = await self.get_model_response(answer_prompt)
            
            # 第二阶段：验证答案正确性
            verification_prompt = self.quality_check_prompts['answer_verification'].format(
                question=question,
                standard_answer=standard_answer,
                model_answer=model_answer
            )
            
            async with self.semaphore:
                verification_result = await self.get_model_response(verification_prompt)
            
            # 提取数字标签
            try:
                quality_label = int(re.search(r'\d+', verification_result.strip()).group())
            except (AttributeError, ValueError):
                logger.warning(f"无法解析验证结果: {verification_result}")
                quality_label = 0
            
            # 记录详细信息
            verification_details = {
                'model_answer': model_answer,
                'verification_response': verification_result,
                'question_length': len(question),
                'answer_length': len(standard_answer),
                'model_answer_length': len(model_answer)
            }
            
            return {
                **qa_data,
                'quality_label': quality_label,
                'verification_details': verification_details
            }
            
        except Exception as e:
            logger.error(f"QA对验证失败: {e}")
            return {**qa_data, 'quality_label': 0, 'verification_details': f'验证失败: {str(e)}'}
    
    async def batch_verify_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量验证QA对
        
        Args:
            qa_pairs: QA对列表
            
        Returns:
            验证结果列表
        """
        logger.info(f"开始批量验证 {len(qa_pairs)} 个QA对")
        
        # 创建验证任务
        tasks = []
        for qa_data in qa_pairs:
            task = asyncio.create_task(self.verify_qa_pair(qa_data))
            tasks.append(task)
        
        # 分批执行任务
        results = []
        for i in range(0, len(tasks), self.parallel_core):
            batch_tasks = tasks[i:i + self.parallel_core]
            batch_results = await gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"任务执行异常: {result}")
                    results.append({'quality_label': 0, 'verification_details': f'执行异常: {str(result)}'})
                else:
                    results.append(result)
            
            logger.info(f"完成批次 {i//self.parallel_core + 1}/{(len(tasks) + self.parallel_core - 1)//self.parallel_core}")
        
        return results
    
    async def check_qa_file_quality(self, 
                                   file_path: str, 
                                   output_path: Optional[str] = None,
                                   quality_threshold: float = 0.6) -> Dict[str, Any]:
        """
        检查QA文件的质量
        
        Args:
            file_path: QA文件路径（支持json, csv, xlsx）
            output_path: 输出文件路径
            quality_threshold: 质量阈值
            
        Returns:
            质量检查报告
        """
        logger.info(f"开始检查文件质量: {file_path}")
        
        # 读取数据
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 如果是嵌套结构，提取QA对
                qa_pairs = []
                for item in data:
                    if 'qa_pairs' in item:
                        for qa in item['qa_pairs']:
                            qa_pairs.append({
                                **qa,
                                'source_file': item.get('source_file', 'unknown')
                            })
                    else:
                        qa_pairs.append(item)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                qa_pairs = df.to_dict('records')
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                qa_pairs = df.to_dict('records')
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
                
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
            return {'error': f'文件读取失败: {str(e)}'}
        
        logger.info(f"成功读取 {len(qa_pairs)} 个QA对")
        
        # 批量验证
        verified_results = await self.batch_verify_qa_pairs(qa_pairs)
        
        # 统计分析
        total_count = len(verified_results)
        passed_count = sum(1 for result in verified_results if result.get('quality_label', 0) == 1)
        pass_rate = passed_count / total_count if total_count > 0 else 0
        
        # 按质量分类
        high_quality = [r for r in verified_results if r.get('quality_label', 0) == 1]
        low_quality = [r for r in verified_results if r.get('quality_label', 0) == 0]
        
        # 生成报告
        quality_report = {
            'file_path': file_path,
            'total_qa_pairs': total_count,
            'passed_qa_pairs': passed_count,
            'pass_rate': pass_rate,
            'quality_threshold': quality_threshold,
            'meets_threshold': pass_rate >= quality_threshold,
            'statistics': {
                'avg_question_length': sum(len(r.get('question', '')) for r in verified_results) / total_count if total_count > 0 else 0,
                'avg_answer_length': sum(len(r.get('answer', '')) for r in verified_results) / total_count if total_count > 0 else 0,
                'question_types_distribution': self._analyze_question_types(verified_results)
            },
            'verification_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 保存结果
        if output_path:
            # 保存详细结果
            detailed_output = output_path.replace('.json', '_detailed.json')
            with open(detailed_output, 'w', encoding='utf-8') as f:
                json.dump(verified_results, f, ensure_ascii=False, indent=2)
            
            # 保存高质量数据
            high_quality_output = output_path.replace('.json', '_high_quality.json')
            with open(high_quality_output, 'w', encoding='utf-8') as f:
                json.dump(high_quality, f, ensure_ascii=False, indent=2)
            
            # 保存质量报告
            report_output = output_path.replace('.json', '_quality_report.json')
            with open(report_output, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, ensure_ascii=False, indent=2)
            
            # 保存CSV格式（便于分析）
            df_results = pd.DataFrame(verified_results)
            csv_output = output_path.replace('.json', '_results.csv')
            df_results.to_csv(csv_output, index=False, encoding='utf-8')
            
            logger.info(f"结果已保存到: {output_path}")
        
        return quality_report
    
    def _analyze_question_types(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析问题类型分布"""
        type_counts = {}
        for qa in qa_pairs:
            q_type = qa.get('question_type', 'unknown')
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        return type_counts


# 与现有系统集成的适配器
class TextQAQualityIntegrator:
    """
    用于将增强质量检查器集成到text_qa_generation系统中
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化集成器
        
        Args:
            config: 配置字典，包含API和模型信息
        """
        self.config = config
        self.quality_checker = None
    
    def initialize_checker(self) -> EnhancedQualityChecker:
        """初始化质量检查器"""
        if self.quality_checker is None:
            # 从配置中提取参数
            api_config = self.config.get('api', {})
            model_config = self.config.get('models', {}).get('qa_generator_model', {})
            quality_config = self.config.get('quality_control', {})
            
            # 系统提示词
            system_prompt = """你是一位半导体和显示技术领域的资深专家，特别擅长IGZO、TFT、OLED等相关技术。
你需要准确回答技术问题，并能够判断答案的正确性和完整性。
请确保你的回答准确、专业、有深度。"""
            
            self.quality_checker = EnhancedQualityChecker(
                api_key=api_config.get('api_key', ''),
                base_url=api_config.get('ark_url', ''),
                model=model_config.get('path', ''),
                system_prompt=system_prompt,
                parallel_core=quality_config.get('parallel_core', 10),
                activate_stream=quality_config.get('activate_stream', False)
            )
        
        return self.quality_checker
    
    async def enhanced_quality_check(self, 
                                   qa_file_path: str,
                                   output_dir: str,
                                   quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        执行增强质量检查
        
        Args:
            qa_file_path: QA文件路径
            output_dir: 输出目录
            quality_threshold: 质量阈值
            
        Returns:
            质量检查报告
        """
        checker = self.initialize_checker()
        
        # 生成输出文件路径
        base_name = os.path.splitext(os.path.basename(qa_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_quality_checked.json")
        
        # 执行质量检查
        report = await checker.check_qa_file_quality(
            qa_file_path, 
            output_path, 
            quality_threshold
        )
        
        logger.info(f"质量检查完成: 通过率 {report['pass_rate']:.2%}")
        return report