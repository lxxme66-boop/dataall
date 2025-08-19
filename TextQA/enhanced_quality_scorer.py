#!/usr/bin/env python3
"""
增强版质量评分系统
借鉴半导体领域QA生成系统的评分机制
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """质量评分数据类"""
    completeness: float = 0.0  # 完整性
    complexity: float = 0.0     # 复杂性
    accuracy: float = 0.0       # 准确性
    reasoning: float = 0.0      # 推理性
    total: float = 0.0          # 总分
    is_suitable: bool = False   # 是否适合


class EnhancedQualityScorer:
    """增强版质量评分器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scoring_criteria = self._load_scoring_criteria()
        
    def _load_scoring_criteria(self) -> Dict[str, Any]:
        """加载评分标准"""
        return {
            "completeness": {
                "no_clear_question": 0,
                "has_main_question": 1,
                "has_interaction": 1
            },
            "complexity": {
                "undergraduate_level": 0,
                "graduate_level": 1,
                "expert_level": 1
            },
            "accuracy": {
                "significant_errors": -1,
                "some_correctness": 0,
                "minor_flaws": 0.5,
                "highly_correct": 0.5,
                "excellent": 1
            },
            "reasoning": {
                "no_reasoning": -1,
                "basic_reasoning": 0.5,
                "moderate_reasoning": 0.5,
                "significant_reasoning": 1,
                "exceptional_reasoning": 1
            }
        }
    
    async def evaluate_text_quality(self, text: str, domain: str = "general") -> QualityScore:
        """
        评估文本质量
        
        Args:
            text: 待评估的文本
            domain: 领域（general/semiconductor/optics/materials）
            
        Returns:
            QualityScore对象
        """
        score = QualityScore()
        
        # 1. 评估完整性
        score.completeness = await self._evaluate_completeness(text)
        
        # 2. 评估复杂性
        score.complexity = await self._evaluate_complexity(text, domain)
        
        # 3. 评估准确性
        score.accuracy = await self._evaluate_accuracy(text, domain)
        
        # 4. 评估推理性
        score.reasoning = await self._evaluate_reasoning(text)
        
        # 计算总分
        score.total = (score.completeness + score.complexity + 
                      score.accuracy + score.reasoning)
        
        # 判断是否适合生成推理问题
        score.is_suitable = (
            score.completeness > 0 and
            score.complexity > 0 and
            score.accuracy > 0 and
            score.reasoning >= 1
        )
        
        return score
    
    async def _evaluate_completeness(self, text: str) -> float:
        """评估文本完整性"""
        # 这里可以接入大模型进行评估
        # 暂时使用规则基础的评估
        
        score = 0.0
        
        # 检查是否有清晰的主要问题
        if self._has_clear_question(text):
            score += 1.0
        
        # 检查是否有互动和讨论
        if self._has_interaction(text):
            score += 1.0
            
        return score
    
    async def _evaluate_complexity(self, text: str, domain: str) -> float:
        """评估技术复杂性"""
        score = 0.0
        
        # 获取领域特定的技术术语
        domain_keywords = self._get_domain_keywords(domain)
        
        # 计算技术术语密度
        tech_density = self._calculate_tech_density(text, domain_keywords)
        
        if tech_density > 0.1:  # 研究生水平
            score += 1.0
        if tech_density > 0.2:  # 专家水平
            score += 1.0
            
        return score
    
    async def _evaluate_accuracy(self, text: str, domain: str) -> float:
        """评估技术准确性"""
        # 这里应该接入领域专家模型进行评估
        # 暂时返回默认值
        return 0.5
    
    async def _evaluate_reasoning(self, text: str) -> float:
        """评估推理深度"""
        score = 0.0
        
        # 检查推理标记词
        reasoning_markers = [
            "因此", "所以", "由于", "导致", "推断", "分析",
            "证明", "推理", "假设", "结论", "因果", "逻辑"
        ]
        
        marker_count = sum(1 for marker in reasoning_markers if marker in text)
        
        if marker_count >= 2:
            score += 0.5
        if marker_count >= 5:
            score += 0.5
        if self._has_reasoning_chain(text):
            score += 1.0
            
        return score
    
    def _has_clear_question(self, text: str) -> bool:
        """检查是否有清晰的问题"""
        question_markers = ["？", "?", "如何", "为什么", "什么是", "怎样"]
        return any(marker in text for marker in question_markers)
    
    def _has_interaction(self, text: str) -> bool:
        """检查是否有互动讨论"""
        interaction_markers = ["回应", "评论", "讨论", "修订", "批评", "反馈"]
        return any(marker in text for marker in interaction_markers)
    
    def _get_domain_keywords(self, domain: str) -> List[str]:
        """获取领域关键词"""
        domain_keywords = {
            "semiconductor": ["IGZO", "TFT", "OLED", "晶体管", "能带", "载流子"],
            "optics": ["光谱", "激光", "折射", "衍射", "干涉", "波长"],
            "materials": ["晶体", "聚合物", "复合材料", "表面", "界面", "结构"],
            "general": []
        }
        return domain_keywords.get(domain, [])
    
    def _calculate_tech_density(self, text: str, keywords: List[str]) -> float:
        """计算技术术语密度"""
        if not text or not keywords:
            return 0.0
        
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        return keyword_count / len(text.split())
    
    def _has_reasoning_chain(self, text: str) -> bool:
        """检查是否有推理链"""
        # 检查是否有"A导致B，B导致C"这样的推理链
        chain_patterns = [
            "导致.*进而",
            "因为.*所以.*因此",
            "首先.*其次.*最后",
            "由于.*从而.*最终"
        ]
        
        import re
        return any(re.search(pattern, text) for pattern in chain_patterns)
    
    async def evaluate_qa_pair(self, question: str, answer: str, 
                              source_text: str = "") -> Dict[str, Any]:
        """
        评估问答对质量
        
        Args:
            question: 问题
            answer: 答案
            source_text: 源文本（可选）
            
        Returns:
            评估结果字典
        """
        # 评估问题质量
        q_score = await self.evaluate_text_quality(question)
        
        # 评估答案质量
        a_score = await self.evaluate_text_quality(answer)
        
        # 评估问答匹配度
        match_score = self._evaluate_qa_match(question, answer)
        
        # 如果提供了源文本，评估相关性
        relevance_score = 0.0
        if source_text:
            relevance_score = self._evaluate_relevance(question, answer, source_text)
        
        return {
            "question_score": q_score.__dict__,
            "answer_score": a_score.__dict__,
            "match_score": match_score,
            "relevance_score": relevance_score,
            "overall_suitable": q_score.is_suitable and a_score.is_suitable,
            "recommendation": self._get_recommendation(q_score, a_score)
        }
    
    def _evaluate_qa_match(self, question: str, answer: str) -> float:
        """评估问答匹配度"""
        # 简单的关键词匹配
        q_keywords = set(question.split())
        a_keywords = set(answer.split())
        
        if not q_keywords:
            return 0.0
            
        overlap = len(q_keywords.intersection(a_keywords))
        return overlap / len(q_keywords)
    
    def _evaluate_relevance(self, question: str, answer: str, source: str) -> float:
        """评估与源文本的相关性"""
        # 检查问题和答案中的内容是否能在源文本中找到
        q_in_source = any(phrase in source for phrase in question.split("，"))
        a_in_source = any(phrase in source for phrase in answer.split("，"))
        
        if q_in_source and a_in_source:
            return 1.0
        elif q_in_source or a_in_source:
            return 0.5
        else:
            return 0.0
    
    def _get_recommendation(self, q_score: QualityScore, 
                          a_score: QualityScore) -> str:
        """获取改进建议"""
        recommendations = []
        
        if q_score.completeness < 1:
            recommendations.append("问题需要更加完整和清晰")
        if q_score.complexity < 1:
            recommendations.append("问题复杂度需要提升到研究生水平")
        if q_score.reasoning < 1:
            recommendations.append("问题需要包含更多推理要素")
        
        if a_score.accuracy < 0.5:
            recommendations.append("答案准确性需要提高")
        if a_score.reasoning < 1:
            recommendations.append("答案需要包含完整的推理过程")
        
        return "；".join(recommendations) if recommendations else "质量良好"


# 使用示例
async def demo():
    """演示如何使用增强版质量评分器"""
    config = {
        "domain": "semiconductor",
        "thresholds": {
            "min_total_score": 2.0,
            "min_reasoning_score": 1.0
        }
    }
    
    scorer = EnhancedQualityScorer(config)
    
    # 评估文本质量
    sample_text = """
    IGZO薄膜晶体管的电子迁移率为10-20 cm²/V·s，远高于传统非晶硅的0.5-1 cm²/V·s。
    这种高迁移率是由于IGZO材料的特殊能带结构，其中In的5s轨道形成了有效的电子传输通道。
    因此，IGZO TFT可以在较低的驱动电压下实现快速开关，进而提高显示器的刷新率和降低功耗。
    """
    
    score = await scorer.evaluate_text_quality(sample_text, "semiconductor")
    print(f"文本质量评分: {score}")
    
    # 评估问答对
    question = "为什么IGZO薄膜晶体管相比传统非晶硅具有更高的电子迁移率？"
    answer = "IGZO具有更高电子迁移率的原因是其特殊的能带结构，特别是In的5s轨道形成了有效的电子传输通道，使得电子能够更快速地移动。"
    
    qa_eval = await scorer.evaluate_qa_pair(question, answer, sample_text)
    print(f"问答对评估: {json.dumps(qa_eval, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    asyncio.run(demo())