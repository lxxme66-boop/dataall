"""
推理型问题生成模板
专门用于生成需要逻辑推理的高质量问题
"""

# 推理型问题模板
reasoning_question_templates = {
    "causal_chain": """
基于以下文本，生成展现完整因果链的问题。问题应该体现"机制A如何影响参数B，进而导致现象C"的逻辑结构。

要求：
1. 问题必须涉及多步推理
2. 每一步都要有明确的因果关系
3. 问题独立于文本，不使用"本文"等表述
4. 确保问题的科学严谨性

文本内容：
{text_content}

请生成3个因果链问题，格式如下：
[[1]] 第1个问题
[[2]] 第2个问题
[[3]] 第3个问题
""",

    "mechanism_analysis": """
基于以下技术文本，生成需要深入分析机制的问题。

要求：
1. 问题应该探究技术原理的深层机制
2. 需要结合多个知识点进行推理
3. 答案必须能从文本中推导出来
4. 避免简单的定义性问题

文本内容：
{text_content}

生成的问题应该让专家也需要思考才能回答。
""",

    "comparative_reasoning": """
基于文本内容，生成需要比较分析和推理的问题。

要求：
1. 问题涉及两个或多个对象的深度比较
2. 不仅比较表面特征，更要分析深层原因
3. 需要推理才能得出比较结论
4. 问题表述完整，不依赖原文

文本内容：
{text_content}
""",

    "problem_solving": """
基于技术文本，生成需要解决实际问题的推理题。

要求：
1. 问题描述一个具体的技术挑战
2. 解决方案需要综合运用文中的多个原理
3. 需要创造性地应用知识
4. 问题具有实际应用价值

文本内容：
{text_content}
"""
}

# 领域特定的推理问题模板
domain_specific_reasoning = {
    "semiconductor": {
        "device_optimization": """
针对半导体器件优化，生成需要深度推理的问题。

关注点：
- 器件性能与材料特性的关系
- 工艺参数对器件的影响机制
- 性能瓶颈的根本原因分析
- 优化策略的理论依据

文本内容：
{text_content}
""",
        "failure_analysis": """
生成关于器件失效机制分析的推理问题。

要求：
- 从现象推导失效原因
- 分析失效的物理机制
- 探讨预防措施的原理
- 多因素耦合效应

文本内容：
{text_content}
"""
    },
    
    "optics": {
        "optical_design": """
生成光学系统设计相关的推理问题。

关注：
- 光学原理的综合应用
- 系统性能的限制因素
- 优化设计的理论基础
- 实际应用中的权衡

文本内容：
{text_content}
""",
        "phenomenon_explanation": """
生成解释光学现象的深度推理问题。

要求：
- 从观察到的现象推导物理本质
- 多个光学原理的综合运用
- 定量分析与定性解释结合
- 理论与实验的对应关系

文本内容：
{text_content}
"""
    }
}

# 问题质量提升策略
quality_enhancement_rules = {
    "completeness": [
        "确保问题包含所有必要的背景信息",
        "明确指出问题的边界条件",
        "提供足够的技术参数"
    ],
    "independence": [
        "避免使用'本文'、'上述'、'该'等指代词",
        "将文中的特定案例泛化为一般性问题",
        "确保不了解原文的专家也能理解问题"
    ],
    "reasoning_depth": [
        "问题应该需要至少3步推理",
        "涉及多个概念的综合运用",
        "需要定量和定性分析结合"
    ],
    "scientific_rigor": [
        "使用准确的专业术语",
        "遵循领域内的标准表述",
        "确保逻辑链条的完整性"
    ]
}

# 问题验证检查点
validation_checklist = {
    "must_have": [
        "清晰的问题陈述",
        "完整的推理要求",
        "可验证的答案",
        "专业的表述"
    ],
    "must_not_have": [
        "文献引用",
        "自指表述",
        "模糊概念",
        "简单记忆型问题"
    ],
    "quality_indicators": [
        "多步推理",
        "因果关系",
        "定量分析",
        "创新应用"
    ]
}


def get_reasoning_prompt(template_type: str, domain: str = None) -> str:
    """
    获取推理型问题生成模板
    
    Args:
        template_type: 模板类型
        domain: 专业领域（可选）
        
    Returns:
        对应的prompt模板
    """
    if domain and domain in domain_specific_reasoning:
        domain_templates = domain_specific_reasoning[domain]
        if template_type in domain_templates:
            return domain_templates[template_type]
    
    return reasoning_question_templates.get(
        template_type, 
        reasoning_question_templates["causal_chain"]
    )


def validate_question(question: str) -> Dict[str, Any]:
    """
    验证生成的问题质量
    
    Args:
        question: 待验证的问题
        
    Returns:
        验证结果字典
    """
    results = {
        "is_valid": True,
        "issues": [],
        "suggestions": []
    }
    
    # 检查必须不包含的内容
    for forbidden in validation_checklist["must_not_have"]:
        if forbidden == "文献引用" and any(marker in question for marker in ["[1]", "[2]", "参考文献"]):
            results["is_valid"] = False
            results["issues"].append("包含文献引用")
            results["suggestions"].append("移除文献引用，使问题独立")
        
        if forbidden == "自指表述" and any(marker in question for marker in ["本文", "本研究", "该实验"]):
            results["is_valid"] = False
            results["issues"].append("包含自指表述")
            results["suggestions"].append("将特定指代改为通用表述")
    
    # 检查质量指标
    quality_score = 0
    for indicator in validation_checklist["quality_indicators"]:
        if indicator == "多步推理" and any(marker in question for marker in ["如何", "为什么", "分析"]):
            quality_score += 1
        if indicator == "因果关系" and any(marker in question for marker in ["导致", "影响", "原因"]):
            quality_score += 1
    
    results["quality_score"] = quality_score
    results["quality_level"] = "high" if quality_score >= 3 else "medium" if quality_score >= 2 else "low"
    
    return results