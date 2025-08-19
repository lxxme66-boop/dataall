from typing import Dict, Any, Optional, List
from ..TextGeneration.prompts_conf import get_prompt, get_system_prompt


class PromptBuilder:
    """Prompt构建器，用于构建各种类型的prompt"""
    
    def __init__(self):
        """初始化Prompt构建器"""
        self.system_prompt = get_system_prompt()
    
    def build_rewrite_prompt(self, question: str, answer: str, 
                           template_type: str = "basic") -> str:
        """
        构建数据改写的prompt
        
        Args:
            question: 原始问题
            answer: 原始答案
            template_type: 模板类型 (basic, advanced, professional)
            
        Returns:
            构建好的prompt
        """
        if template_type == "advanced":
            template = """
基于以下问答对，生成一个高质量的改写版本，要求：

1. 保持核心内容和答案的准确性不变
2. 提升问题的表达质量和专业性
3. 增强推理过程的逻辑性和完整性
4. 适合研究生水平的学习使用

原始问题: {question}
原始答案: {answer}

请生成改写版本，以JSON格式返回：
{{
    "question": "改写后的问题（表达更自然、专业）",
    "answer": "改写后的答案（内容一致但表述更清晰）",
    "reasoning": "详细的推理过程",
    "background": "相关的背景知识",
    "difficulty_level": "问题难度等级（basic/intermediate/advanced）",
    "question_type": "问题类型（factual/reasoning/comparison/open_ended）"
}}
"""
        elif template_type == "professional":
            template = """
作为半导体和光学领域的专家，请基于以下问答对生成专业级别的改写版本：

原始问题: {question}
原始答案: {answer}

改写要求：
1. 使用专业术语和准确的科学表述
2. 增加相关的物理原理和机制解释
3. 提供详细的推理链条和科学依据
4. 确保内容适合作数据集

请以JSON格式返回：
{{
    "question": "专业改写的问题",
    "answer": "专业改写的答案", 
    "reasoning": "基于科学原理的详细推理过程",
    "related_concepts": ["相关概念1", "相关概念2"],
    "scientific_principles": "涉及的科学原理",
    "applications": "实际应用场景",
    "references": "相关理论或实验背景"
}}
"""
        else:  # basic
            template = """
基于以下问答对，生成一个新的改写版本，保持核心内容不变但表达方式更丰富：

原始问题: {question}
原始答案: {answer}

请生成：
1. 改写后的问题（保持原意但表达更自然）
2. 改写后的答案（内容一致但表述更清晰）
3. 相关的背景知识
4. 推理过程

以JSON格式返回，包含question, answer, background, reasoning字段。
"""
        
        return template.format(question=question, answer=answer)
    
#     def build_multimodal_prompt(self, text_content: str, image_path: str = "",
#                                prompt_id: int = 343) -> str:
#         """
#         构建多模态prompt
        
#         Args:
#             text_content: 文本内容
#             image_path: 图片路径
#             prompt_id: 使用的prompt ID
            
#         Returns:
#             构建好的prompt
#         """
#         base_prompt = get_prompt(prompt_id)
#         if not base_prompt:
#             # 使用默认的多模态prompt
#             base_prompt = """
# <image>
# 以下是论文中和图片相关的内容

# {text}

# 我需要你根据以上信息设计一个提示词，提示词必须是和图片相关的问题，问题是是客观题(选择题或者填空题)且和半导体相关或者光学相关
# 规则：
# 1. 需要为答案和推理过程提供足够的背景信息, 但不得直接摘录我提供的论文内容
# 2. 提示词需要围绕相关公理和推理过程展开
# 3. 题目需要有一定难度，不可以过分简单，面向人群是研究生人群
# 4. 问题必须要利用图片上的信息才能获得正确答案，如果发现只利用提示词就可以回答需要重新设计
# 5. 设计完成后判断，是否不依赖图片信息也能解答，如果可以重新设计问题

# 以json格式返回, 且只返回一个json对象。
# json的key为: question,img,answer,reasoning

# "question": 文字输入，如果是选择题,则为问题和选项的合并,例如: "根据以下问题和选项,回答问题......"
# 如果无法设计出问题,则在question中返回"无法设计出问题",answer和reasoning的值也为"无法设计出问题"
# """
        
#         return base_prompt.format(text=text_content)
    
    def build_quality_check_prompt(self, question: str, answer: str, reasoning: str,
                                 check_type: str = "comprehensive") -> str:
        """
        构建质量检查prompt
        
        Args:
            question: 问题
            answer: 答案  
            reasoning: 推理过程
            check_type: 检查类型 (reasoning, clarity, correctness, difficulty, comprehensive)
            
        Returns:
            构建好的prompt
        """
        if check_type == "reasoning":
            prompt_id = 36
        elif check_type == "clarity":
            prompt_id = 37
        elif check_type == "correctness":
            prompt_id = 39
        elif check_type == "difficulty":
            prompt_id = 40
        else:  # comprehensive
            return self.build_comprehensive_check_prompt(question, answer, reasoning)
        
        base_prompt = get_prompt(prompt_id)
        if base_prompt:
            return base_prompt.format(question=question, answer=answer, reasoning=reasoning)
        else:
            # 使用默认prompt
            return f"""
问题：{question}
答案：{answer}
推理过程：{reasoning}

请对以上QA对进行质量评估，从以下维度进行评分（1-5分）：
1. 问题的清晰度和完整性
2. 答案的准确性和完整性
3. 推理过程的逻辑性和严密性
4. 整体的专业性和难度适中性

请以JSON格式返回评估结果。
"""
    
    def build_comprehensive_check_prompt(self, question: str, answer: str, reasoning: str) -> str:
        """
        构建综合质量检查prompt
        
        Args:
            question: 问题
            answer: 答案
            reasoning: 推理过程
            
        Returns:
            构建好的prompt
        """
        return f"""
请对以下QA数据进行综合质量评估：

问题：{question}
答案：{answer}
推理过程：{reasoning}

评估维度：
1. 推理有效性：推理过程是否基于问题内容和科学原理
2. 问题清晰度：问题是否表述清楚，信息是否充分
3. 答案正确性：答案是否准确，推理是否正确
4. 难度适宜性：问题难度是否适中，是否需要适当推理

请对每个维度给出1-5分的评分，并提供具体的评价理由。

以JSON格式返回：
{{
    "reasoning_validity": {{"score": 分数, "reason": "评价理由"}},
    "question_clarity": {{"score": 分数, "reason": "评价理由"}},
    "answer_correctness": {{"score": 分数, "reason": "评价理由"}},
    "difficulty_level": {{"score": 分数, "reason": "评价理由"}},
    "overall_score": 总分,
    "suggestions": ["改进建议1", "改进建议2"]
}}
"""
    
    def build_domain_specific_prompt(self, content: str, domain: str = "semiconductor",
                                   task_type: str = "qa_generation") -> str:
        """
        构建领域特定的prompt
        
        Args:
            content: 内容
            domain: 领域 (semiconductor, optics, materials)
            task_type: 任务类型 (qa_generation, analysis, explanation)
            
        Returns:
            构建好的prompt
        """
        domain_contexts = {
            "semiconductor": "作为半导体物理和器件专家",
            "optics": "作为光学和光电子学专家", 
            "materials": "作为材料科学专家"
        }
        
        task_instructions = {
            "qa_generation": "请生成高质量的问答对",
            "analysis": "请进行深入的技术分析",
            "explanation": "请提供详细的原理解释"
        }
        
        domain_context = domain_contexts.get(domain, "作为技术专家")
        task_instruction = task_instructions.get(task_type, "请处理以下内容")
        
        return f"""
{domain_context}，{task_instruction}。

内容：
{content}

要求：
1. 确保技术准确性和专业性
2. 使用准确的专业术语
3. 提供充分的理论依据
4. 适合研究生及以上水平

请以JSON格式返回处理结果。
"""


def build_rewrite_prompt(question: str, answer: str, template_type: str = "basic") -> str:
    """
    构建改写prompt的便捷函数
    
    Args:
        question: 原始问题
        answer: 原始答案
        template_type: 模板类型
        
    Returns:
        构建好的prompt
    """
    builder = PromptBuilder()
    return builder.build_rewrite_prompt(question, answer, template_type)


def build_quality_check_prompt(question: str, answer: str, reasoning: str,
                             check_type: str = "comprehensive") -> str:
    """
    构建质量检查prompt的便捷函数
    
    Args:
        question: 问题
        answer: 答案
        reasoning: 推理过程
        check_type: 检查类型
        
    Returns:
        构建好的prompt
    """
    builder = PromptBuilder()
    return builder.build_quality_check_prompt(question, answer, reasoning, check_type)