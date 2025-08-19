# import base64
# import argparse
# import os
# # 通过 pip install volcengine-python-sdk[ark] 安装方舟SDK
# #from volcenginesdkarkruntime import Ark, AsyncArk
# # 通过 pip install volcengine-python-sdk[ark] 安装方舟SDK
# try:
#     from volcenginesdkarkruntime import Ark, AsyncArk
#     HAS_VOLC_SDK = True
# except ImportError:
#     HAS_VOLC_SDK = False
#     Ark = None
#     AsyncArk = None
# import os
# import re
# #from Doubao.prompts_conf import system_prompt, user_prompts
# import os
# import re
# from TextGeneration.prompts_conf import system_prompt, user_prompts

# import asyncio
# import json
# from typing import List, Dict, Any

# ark_url = "http://localhost:8000/v1"
# api_key = "EMPTY" # my own api
# model = "/mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain"


# class ArgumentDataProcessor:
#     """数据增强处理器类"""
    
#     def __init__(self, api_key: str = api_key, model: str = model):
#         self.api_key = api_key
#         self.model = model
#         self.ark_url = ark_url
    
#     async def enhance_qa_data(self, qa_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """增强QA数据"""
#         enhanced_data = []
        
#         # 处理每个QA对
#         for idx, qa_item in enumerate(qa_data):
#             try:
#                 # 使用现有的modify_sft_response函数进行增强
#                 enhanced_item = await self._enhance_single_qa(qa_item, idx)
#                 enhanced_data.append(enhanced_item)
#             except Exception as e:
#                 print(f"Error enhancing QA item {idx}: {e}")
#                 enhanced_data.append(qa_item)  # 如果增强失败，保留原始数据
        
#         return enhanced_data
    
#     async def _enhance_single_qa(self, qa_item: Dict[str, Any], index: int) -> Dict[str, Any]:
#         """增强单个QA项"""
#         # 选择合适的prompt模板
#         prompt_template = user_prompts['rewrite_prompt']
        
#         # 调用现有的modify_sft_response函数
#         enhanced_response = await modify_sft_response([qa_item], 0, prompt_template)
        
#         if enhanced_response:
#             return enhanced_response
#         else:
#             return qa_item
# async def modify_sft_response(responses, index,prompt_template):

#     response = responses[index]
#     question = response.get('question', '')
#     answer = response.get('answer', '')
#     reasoning = response.get('reasoning', '')
#     lecture = response.get('lecture', '')
#     context = response.get('context', '')
#     imageDescription = response.get('imageDescription', '')
#     AnalysisResult = response.get('analysisResults', '')
#     RelatedKnowledge = response.get('relatedKnowledge', '')
#     prompt = prompt_template.format(
#         question=question,
#         answer=answer,
#         choices=choices,
#         reasoning=reasoning,
#         lecture=lecture,
#         context=context,
#         imageDescription=imageDescription,
#         analysisResults=AnalysisResult,
#         relatedKnowledge=RelatedKnowledge
#     )
#     ark = AsyncArk(api_key=api_key)
#     async def get_request(input_prompt):
#         try:
#             #print(f"Sending request for index {index} with prompt: {input_prompt}")
#             response = await ark.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": [
#                             {"type": "text", "text": system_prompt}

#                         ]
#                     },
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": input_prompt},
#                         ]
#                     },
#                 ],
#                 temperature=1,
#                 max_tokens=4096,
#                 top_p=0.7,
                
#                 stop=None


#             )
#             try:
#                 content = response.choices[0].message.content
#                 #print(content)

           
#                 content = eval(content)
#                 content["choices"] = choices
#                 content["question"] = question
#                 content["answer"] = answer
              
#                 content["lecture"] = lecture
#                 content["context"] = context
#             except Exception as e:
#                 print(f"Error in eval: {e}")
                
#                 #print(f"Using raw content: {content}")
#                 return None
#             return content
#         except Exception as e:
#             ##print(f"Error in request: {e}")
#             if 'ModelAccountTpmRateLimitExceeded' in str(e):
#                 print("Rate limit exceeded, waiting for 60 seconds...")
#                 await asyncio.sleep(60)
#                 return None
#             else:
#                 # interrupt the main thread
#                 print(f"input prompt: {input_prompt}")
                
#                 print(f"Error in request: {e}")
#                 return None
                
            
#     content = await get_request(prompt)
#     if content is None:
#         print(f"Response is None for index {index}, skipping...")
#         return None
#     content["image_path"] = image_path
#     print(content)
#     return content
# async def check_ori_response(responses,prompt_template,batch_size=1):
#     async def check_response(response,prompt_template):
        
#         vqa_data = {}
#         vqa_data["input"] = response["input"]
#         vqa_data["instruction"] = response["instruction"]
        
#         vqa_data["reasoning"] = response["reasoning"]
    
#         vqa_data["output"] = response["output"]
#         check_prompt = prompt_template.format(vqa_data=vqa_data)
#         ark = AsyncArk(api_key=api_key)
#         content = await ark.chat.completions.create(
#             model=model,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": [
#                         {"type": "text", "text": system_prompt}
#                     ]
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": check_prompt},
#                     ]
#                 },
#             ],
#             temperature=1,
#             max_tokens=4096,
#             top_p=0.7,
#             stop=None
#         )
#         try:
#             content = content.choices[0].message.content
#             content = eval(content)
#             # print(f"Response: {content}")
#             # print(f"input prompt: {response}")
#             return content, response
#         except Exception as e:
#             print(f"Error in eval: {e}")
#             return None
#     new_responses = []
#     tasks = []
#     for response in responses:
#         task = asyncio.create_task(check_response(response, prompt_template))
#         tasks.append(task)
#     for i in range(0, len(tasks), batch_size):
#         batches = tasks[i:i + batch_size]
#         batches = await asyncio.gather(*batches)
#         # Use asyncio.gather to run the tasks concurrently  
#         for t, data in enumerate(batches):
#             response, vqa_data = data
#             if response is None:
#                 print(f"Response is None, skipping...")
#                 continue
#             elif response["useful"]:
#                 new_responses.append(vqa_data)
#                 # print(vqa_data)
#             else:
#                 print(response)
#                 print(vqa_data)

        
#     return new_responses

# async def rephrase_response(responses, index,poolSize, CheckQestion):
#     response_index = 0
#     tasks = []
#     for i in range(len(responses)):
#         task = asyncio.create_task(modify_sft_response(responses, i, user_prompts[index]))
#         tasks.append(task)
#     # Use asyncio.gather to run the tasks concurrently
#     for i in range(0, len(tasks), poolSize):
#         batch_tasks = tasks[i:i + poolSize]
#         batch_responses = await asyncio.gather(*batch_tasks)
#         for response in batch_responses:
#             if response is not None:
#                 responses[response_index] = response
#                 response_index += 1
#                 print(f"Processed response {response_index}/{len(responses)}")
#     return responses


# if __name__ == "__main__":
#     import json
#     parser = argparse.ArgumentParser(description="Process some integers.")
#     parser.add_argument("--input_file", type=str, \
#                         default="/home/maxzhang/VLReasoningTCL/data/pdfs/version 9.1.json",\
#                          help="path to the input json file")

#     parser.add_argument("--output_file", type=str, default="/home/maxzhang/VLReasoningTCL/data/pdfs", help="path to the output folder")
#     parser.add_argument("--indexes", type=int, default = 21, help="indexes for rephrase prompt 21 for sft, 22 for cpt")
#     parser.add_argument("--poolSize", type=int, default=8, help="number of parallel tasks")
#     parser.add_argument("--CheckQestion",type=int, default=22, help="whether to check the question or not, if not the index will be -1, otherwise will be positive index")
    
#     args = parser.parse_args()
    
#     output_dir = args.output_file
#     indexes = args.indexes
#     poolsize = args.poolSize
#     input_file = args.input_file
#     with open(input_file,"r") as f:
#         input_file = json.load(f)

#     responses = asyncio.run(rephrase_response(input_file, indexes, poolSize=poolsize, CheckQestion=args.CheckQestion))
#     with open(os.path.join(output_dir, "rephrased_responses_qa.json"), "w", encoding='utf-8') as f:
#         import json
#         json.dump(responses, f, ensure_ascii=False, indent=4)
#     print(responses)
#     if args.CheckQestion != -1:
#         file_path = os.path.join(output_dir, f"rephrased_responses_qa.json")
#         responses = []
#         indexes = args.CheckQestion
#         with open(file_path, "r") as f:
#             responses = json.load(f)
#         responses = asyncio.run(check_ori_response(responses, user_prompts[indexes],batch_size=poolsize))
#         with open(os.path.join(output_dir, "checked_responses_qa.json"), "w", encoding='utf-8') as f:
#             import json
#             json.dump(responses, f, ensure_ascii=False, indent=4)


#!/usr/bin/env python3
"""
专家级文本QA数据增强处理器 - 基于质量评估的智能策略选择
融合半导体显示领域专家评估标准和专业改写要求
"""

import asyncio
import json
import os
import logging
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime
import httpx  # 添加httpx库

# 内置配置
SYSTEM_PROMPT = """你是半导体显示领域的专家，专门负责数据质量评估和增强。
你具备深厚的半导体工艺、显示技术、薄膜晶体管等领域知识。
请严格按照JSON格式返回结果，确保技术内容准确专业。"""

# 半导体关键词配置
SEMICONDUCTOR_KEYWORDS = {
    "materials": ["IGZO", "LTPS", "a-Si", "Oxide TFT", "OLED", "QD", "Mini LED", "Micro LED"],
    "processes": ["蒸镀", "光刻", "刻蚀", "退火", "CVD", "PVD", "离子注入", "CMP"],
    "defects": ["Mura", "亮点", "暗点", "串扰", "残影", "色偏", "均匀性"],
    "parameters": ["迁移率", "阈值电压", "开关比", "漏电流", "响应时间", "对比度", "色域"],
    "equipment": ["曝光机", "刻蚀机", "薄膜沉积", "检测设备", "贴合机", "激光退火"]
}

# 质量权重配置
QUALITY_WEIGHTS = {
    "reasoning_chain": 0.35,
    "technical_accuracy": 0.30,
    "domain_depth": 0.20,
    "application_value": 0.15
}

# API配置
API_CONFIG = {
    "temperature": 0.3,
    "max_tokens": 4096,
    "top_p": 0.9
}

# 日志模板
LOG_TEMPLATES = {
    "enhancement_start": "开始处理第 {index}/{total} 个QA: {question_preview}",
    "enhancement_success": "策略 {enhancement_type} 成功，质量评分: {quality_score:.2f}",
    "enhancement_failed": "策略 {enhancement_type} 失败: {error}"
}

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API配置 - 本地vllm服务
ARK_URL = "http://localhost:8000/v1"
API_KEY = ""  # 本地服务不需要密钥
MODEL = "qwen-vllm"  # 与启动的模型名称一致

class ExpertQualityAssessment:
    """专家级质量评估系统"""
    
    @staticmethod
    def get_quality_evaluation_prompt() -> str:
        """获取专家级质量评估提示词"""
        return """作为半导体显示领域的专业质量评估专家，请严格按以下标准评估问答对的质量。

### 评估维度
1. 思维链逻辑质量（权重35%）
   - 步骤完整性：推理步骤是否覆盖问题所有关键点？
   - 因果连贯性：前后步骤是否存在清晰因果关系？
   - 技术参数合理性：工艺参数是否符合物理规律？
   - 错误回溯机制：是否考虑可能故障点？
   - 答案是否源于文本

2. 技术准确度（权重30%）
   - 材料特性：材料描述是否符合物性？
   - 制程参数：工艺参数是否匹配行业标准？
   - 标准引用：是否准确引用SEMI/SID等国际标准？
   - 专利技术：技术方案是否规避近期专利？

3. 领域深度（权重20%）
   - 缺陷机理：是否分析根本原因？
   - 技术趋势：是否覆盖最新发展？
   - 工艺瓶颈：是否识别关键限制？

4. 应用价值（权重15%）
   - 工程可行性：方案是否具备量产实施条件？
   - 成本优化：是否量化成本效益？
   - 良率提升路径：是否提供可验证的改善方案？

### 综合评估标准
**high**: 无明显错误，内容准确、专业、逻辑清晰、紧扣主题，答案在文本中能找到，答案中包含的答案出处在文本适合直接用于SFT
**medium**: 存在轻微问题或可优化项，需修改后方可考虑使用
**low**: 存在严重缺陷，不适合用于SFT

### 待评估样本
问题: {question}
答案: {answer}
上下文: {context}

请严格按照JSON格式返回评估结果：
{{
    "quality_rating": {{
        "overall": "high/medium/low",
        "detailed_scores": {{
            "reasoning_chain": {{"score": "high/medium/low", "issues": ["具体问题1", "问题2"]}},
            "technical_accuracy": {{"score": "high/medium/low", "issues": []}},
            "domain_depth": {{"score": "high/medium/low", "issues": []}},
            "application_value": {{"score": "high/medium/low", "issues": []}}
        }}
    }},
    "improvement_suggestions": ["具体建议1", "建议2"],
    "numeric_score": 0.85,
    "sft_suitable": true
}}"""

class ExpertStandardsGuided:
    """专家标准指导的增强系统"""
    
    @staticmethod
    def get_expert_guided_enhancement_system() -> Dict[str, str]:
        """获取专家标准指导的增强模板系统"""
        return {
            "expert_rewrite": """你是半导体显示领域的资深专家。请按照以下专家标准重写问答对，确保输出直接达到高质量要求：

### 专家标准（必须全部满足）：
1. **思维链逻辑质量（20%权重）**
   - 推理步骤必须覆盖问题所有关键技术点
   - 每一步都有清晰的因果逻辑链条
   - 工艺参数必须符合物理规律（如LTPS退火温度<玻璃转化点）
   - 包含潜在故障点分析（如Mura缺陷需考虑设备精度因素）

2. **技术准确度（20%权重）**
   - 材料特性描述必须符合实际物性（如IGZO迁移率范围准确）
   - 制程参数匹配当前产线标准和能力
   - 准确引用SEMI/SID等国际标准
   - 技术方案规避近期专利冲突

3. **领域深度（5%权重）**
   - 深入分析缺陷根本机理（如亮暗点关联电致迁移机制）
   - 覆盖最新技术发展（如Micro LED巨量转移技术）
   - 识别关键工艺瓶颈（如QD-OLED喷墨打印精度限制）

4. **应用价值（10%权重）**
   - 方案具备量产实施可行性
   - 提供量化的成本效益分析
   - 给出可验证的良率提升路径

5. **答案完整性（20%权重）**
    -是否存在中断

6. **答案来源（25%权重）**
    -答案必须基于论文回答问题
    -答案必须在原文中有出处

原始问题: {question}
原始答案: {answer}
上下文: {context}

### 重写要求（按专家标准）：
- 问题：增加具体技术参数、工艺细节，使问题具有挑战性
- 答案：必须包含详细推理链、科学原理、量化数据、实际应用
- 禁止泛泛而谈，每个技术点都要有深度分析

请以JSON格式返回（确保达到专家级质量）：
{{
    "question": "按专家标准重写的专业问题",
    "answer": "按专家标准重写的深度技术答案",
    "expert_compliance": {{
        "reasoning_chain_quality": "详细推理过程说明",
        "technical_accuracy": "技术准确性保证",
        "domain_depth": "领域深度体现",
        "application_value": "应用价值说明"
    }},
    "quality_indicators": {{
        "technical_parameters_included": true,
        "scientific_principles_explained": true,
        "quantitative_analysis_provided": true,
        "industry_relevance_confirmed": true
    }},
    "expert_score_prediction": 0.9
}}""",

            "medium_quality_upgrade": """按专家标准将中等质量问答对升级到高质量：

### 专家升级标准：
1. 补充缺失的技术深度和推理链条
2. 增加量化分析和具体参数
3. 关联最新行业发展和标准
4. 提供工程实施的可行性分析
上下文: {context}
原始问题: {question}
原始答案: {answer}
当前质量问题: {quality_issues}

### 升级方向（确保满足专家标准）：
- 深化技术机理解释
- 补充定量分析内容
- 增加工艺参数细节
- 关联实际应用场景
- 基于上下文回答（严格执行）

请按JSON格式返回专家级升级版本：
{{
    "question": "升级后的专家级问题",
    "answer": "升级后的专家级答案",
    "upgrade_highlights": ["升级要点1", "升级要点2"],
    "expert_compliance_achieved": true,
    "predicted_expert_score": 0.85
}}""",

            "high_quality_diversification": """对已达专家标准的问答进行技术角度多样化：

### 多样化原则（保持专家级质量）：
1. 从不同技术维度提问（材料/工艺/设备/良率）
2. 保持相同的技术深度和专业性
3. 确保每个新问题都有深度技术答案
4. 关注不同的工程应用场景

高质量原问题: {question}
高质量原答案: {answer}
上下文: {context}
### 多样化方向：
- 材料特性角度：分析材料物性对性能的影响
- 工艺参数角度：探讨关键工艺对结果的影响  
- 设备精度角度：分析设备能力对良率的影响
- 成本优化角度：从成本效益角度分析技术选择

答案基于上下文回答优化，一定要在上下文中有出处（严格执行）

请生成一个保持专家级质量的相关问题：
{{
    "question": "新的专家级多样化问题",
    "answer": "对应的专家级深度答案", 
    "technical_angle": "技术角度说明",
    "expert_quality_maintained": true,
    "diversification_type": "材料/工艺/设备/成本"
}}"""
        }


class TextQAArgumentDataProcessor:
    """专家级文本QA数据增强处理器"""
    
    def __init__(self, api_key: str = API_KEY, model: str = MODEL):
        self.api_key = api_key
        self.model = model
        self.ark_url = ARK_URL
        
        # 质量评估器
        self.quality_assessor = ExpertQualityAssessment()
        
        # 专家标准指导系统
        self.expert_standards = ExpertStandardsGuided()
        
        # 重新定义策略配置：专家标准指导
        self.strategy_config = {
            "expert_guided": True,  # 标识这是专家指导系统
            "quality_thresholds": {
                "low": 0.6,      # 低于0.6需要完全按专家标准重写
                "medium": 0.85,   # 0.6-0.8需要按专家标准升级
                "high": 0.85      # 高于0.8进行专家级多样化
            }
        }
    
    def _get_enhancement_templates(self) -> Dict[str, str]:
        """获取专家标准指导的增强模板集合"""
        base_templates = {
            "quality_assessment": self.quality_assessor.get_quality_evaluation_prompt()
        }
        
        # 融合专家标准指导的模板
        expert_templates = self.expert_standards.get_expert_guided_enhancement_system()
        base_templates.update(expert_templates)
        
        return base_templates
    
    async def enhance_qa_data_with_quality_driven_strategy(self, qa_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于质量评估的智能策略增强"""
        logger.info(f"开始质量驱动增强 {len(qa_data)} 个QA对")
        enhanced_data = []
        
        for idx, qa_item in enumerate(qa_data):
            try:
                question_preview = qa_item.get('question', '')[:50] + "..." if len(qa_item.get('question', '')) > 50 else qa_item.get('question', '')
                logger.info(f"处理第 {idx+1}/{len(qa_data)} 个QA: {question_preview}")
                
                # 步骤1: 质量评估
                quality_result = await self._assess_quality(qa_item)
                if not quality_result:
                    enhanced_data.append(self._format_original_qa(qa_item))
                    continue
                
                overall_quality = quality_result.get("quality_rating", {}).get("overall", "low")
                numeric_score = quality_result.get("numeric_score", 0.5)
                
                logger.info(f"质量评估: {overall_quality} (分数: {numeric_score:.2f})")
                
                # 步骤2: 基于质量选择策略
                strategies = self._select_strategies_by_quality(overall_quality, numeric_score)
                
                # 步骤3: 执行增强策略
                enhanced_items = await self._execute_enhancement_strategies(qa_item, strategies, quality_result)
                enhanced_data.extend(enhanced_items)
                
                # 控制处理频率
                if (idx + 1) % 3 == 0:
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"增强第 {idx} 个QA对失败: {e}")
                enhanced_data.append(self._format_original_qa(qa_item))
        
        logger.info(f"质量驱动增强完成：原始 {len(qa_data)} 个 → 增强后 {len(enhanced_data)} 个")
        return enhanced_data
    
    def _select_strategies_by_quality(self, overall_quality: str, numeric_score: float) -> List[str]:
        """基于质量评估结果选择专家指导的增强策略"""
        if numeric_score < 0.6 or overall_quality == "low":
            strategies = ["expert_rewrite"]  # 低质量：按专家标准完全重写
            logger.info(f"低质量数据，采用专家标准重写策略")
        elif numeric_score < 0.8 or overall_quality == "medium":
            strategies = ["medium_quality_upgrade"]  # 中等质量：按专家标准升级
            logger.info(f"中等质量数据，采用专家标准升级策略") 
        else:
            strategies = ["high_quality_diversification"]  # 高质量：保持专家级标准的多样化
            logger.info(f"高质量数据，采用专家级多样化策略")
        
        return strategies
    
    async def _assess_quality(self, qa_item: Dict[str, Any]) -> Optional[Dict]:
        """评估QA对质量"""
        try:
            question = qa_item.get('question', '')
            answer = qa_item.get('answer', '')
            context = qa_item.get('context', qa_item.get('paper_content', ''))
            
            if not question or not answer:
                return None
            
            templates = self._get_enhancement_templates()
            prompt = templates["quality_assessment"].format(
                question=question, 
                answer=answer, 
                context=context[:1500]
            )
            
            result = await self._call_api(prompt, "quality_assessment")
            
            if result and isinstance(result, dict):
                # 确保数值评分存在
                if "numeric_score" not in result:
                    overall = result.get("quality_rating", {}).get("overall", "medium")
                    score_mapping = {"high": 0.85, "medium": 0.65, "low": 0.4}
                    result["numeric_score"] = score_mapping.get(overall, 0.5)
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return None
    
    async def _execute_enhancement_strategies(self, qa_item: Dict[str, Any], 
                                           strategies: List[str], 
                                           quality_result: Dict) -> List[Dict[str, Any]]:
        """执行增强策略"""
        enhanced_items = []
        
        question = qa_item.get('question', '')
        answer = qa_item.get('answer', '')
        context = qa_item.get('context', qa_item.get('paper_content', ''))
        
        templates = self._get_enhancement_templates()
        
        for strategy in strategies:
            try:
                if strategy not in templates:
                    logger.warning(f"未知策略: {strategy}")
                    continue
                
                # 根据当前质量问题调整增强参数
                quality_issues = self._extract_quality_issues(quality_result)
                
                prompt = templates[strategy].format(
                    question=question,
                    answer=answer, 
                    context=context[:2000],
                    quality_issues=quality_issues if strategy == "medium_quality_upgrade" else ""
                )
                
                enhancement_result = await self._call_api(prompt, strategy)
                
                if enhancement_result:
                    formatted_item = self._format_enhanced_qa_expert(
                        enhancement_result, qa_item, strategy, quality_result
                    )
                    enhanced_items.append(formatted_item)
                    
                    # 记录专家标准指导的成功
                    expert_score = enhancement_result.get('expert_score_prediction', enhancement_result.get('predicted_expert_score', 0.8))
                    logger.info(f"专家标准指导 {strategy} 成功，预期专家评分: {expert_score}")
                    
            except Exception as e:
                logger.error(f"策略 {strategy} 执行失败: {e}")
        
        # 如果没有成功增强，保留原始数据（带质量评估）
        if not enhanced_items:
            original_item = self._format_original_qa(qa_item)
            original_item["quality_assessment"] = quality_result
            enhanced_items.append(original_item)
        
        return enhanced_items
    
    def _extract_quality_issues(self, quality_result: Dict) -> str:
        """从质量评估结果中提取问题点"""
        try:
            detailed_scores = quality_result.get("quality_rating", {}).get("detailed_scores", {})
            issues = []
            
            for dimension, score_info in detailed_scores.items():
                if isinstance(score_info, dict):
                    dimension_issues = score_info.get("issues", [])
                    if dimension_issues:
                        issues.extend([f"{dimension}: {issue}" for issue in dimension_issues])
            
            improvement_suggestions = quality_result.get("improvement_suggestions", [])
            if improvement_suggestions:
                issues.extend([f"建议: {suggestion}" for suggestion in improvement_suggestions])
            
            return "; ".join(issues) if issues else "需要提升整体质量"
            
        except Exception as e:
            logger.warning(f"提取质量问题失败: {e}")
            return "需要提升整体质量"
    
    async def _call_api(self, prompt: str, operation_type: str) -> Optional[Dict]:
        """调用API"""
        try:
            # 构建请求数据
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": API_CONFIG["temperature"],
                "max_tokens": API_CONFIG["max_tokens"],
                "top_p": API_CONFIG["top_p"]
            }
            
            # 设置超时
            timeout = httpx.Timeout(60.0, connect=10.0)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ark_url}/chat/completions",
                    json=data,
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # 提取内容
                content = result['choices'][0]['message']['content'].strip()
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 尝试提取JSON内容
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    logger.warning(f"API返回非JSON格式: {content[:200]}...")
                    return None
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("API速率限制，等待60秒...")
                await asyncio.sleep(60)
                return await self._call_api(prompt, operation_type)
            else:
                logger.error(f"API调用失败 ({operation_type}): HTTP {e.response.status_code} - {e.response.text}")
                return None
        except Exception as e:
            logger.error(f"API调用失败 ({operation_type}): {e}")
            return None
    
    def _format_enhanced_qa_expert(self, enhancement_result: Dict, original_qa: Dict, 
                                  strategy: str, quality_result: Dict) -> Dict:
        """格式化专家增强后的QA数据"""
        formatted = {
            "question": enhancement_result.get("question", original_qa.get("question", "")),
            "answer": enhancement_result.get("answer", original_qa.get("answer", "")),
            "context": original_qa.get("context", original_qa.get("paper_content", "")),
            "source_file": original_qa.get("source_file", "unknown"),
            
            # 专家标准指导信息
            "expert_guided_enhancement": {
                "strategy": strategy,
                "guided_by_expert_standards": True,
                "quality_score": enhancement_result.get("quality_score", enhancement_result.get("expert_score_prediction", 0.8)),
                "enhancement_type": enhancement_result.get("enhancement_type", strategy),
                "enhanced_at": datetime.now().isoformat(),
                
                # 专家合规性检查
                "expert_compliance": enhancement_result.get("expert_compliance", {}),
                "quality_indicators": enhancement_result.get("quality_indicators", {}),
                "expert_score_prediction": enhancement_result.get("expert_score_prediction", 
                                                               enhancement_result.get("predicted_expert_score", 0.8)),
                
                # 传统专家级字段
                "reasoning": enhancement_result.get("reasoning", ""),
                "related_concepts": enhancement_result.get("related_concepts", []),
                "scientific_principles": enhancement_result.get("scientific_principles", ""),
                "applications": enhancement_result.get("applications", ""),
                "references": enhancement_result.get("references", ""),
                "upgrade_highlights": enhancement_result.get("upgrade_highlights", []),
                "technical_angle": enhancement_result.get("technical_angle", ""),
                "diversification_type": enhancement_result.get("diversification_type", "")
            },
            
            # 原始质量评估
            "original_quality_assessment": quality_result,
            
            # 原始数据备份
            "original_data": {
                "question": original_qa.get("question", ""),
                "answer": original_qa.get("answer", "")
            }
        }
        
        # 继承原始数据的其他字段
        for key, value in original_qa.items():
            if key not in formatted and key not in ["question", "answer"]:
                formatted[key] = value
        
        return formatted
    
    def _format_original_qa(self, qa_item: Dict) -> Dict:
        """格式化原始QA数据"""
        return {
            "question": qa_item.get("question", ""),
            "answer": qa_item.get("answer", ""),
            "context": qa_item.get("context", qa_item.get("paper_content", "")),
            "source_file": qa_item.get("source_file", "unknown"),
            "expert_guided_enhancement": {
                "strategy": "none",
                "guided_by_expert_standards": False,
                "quality_score": 0.6,
                "enhancement_type": "original",
                "enhanced_at": datetime.now().isoformat(),
                "note": "原始数据未经专家标准指导增强"
            }
        }
    
    async def generate_quality_report(self, qa_data: List[Dict[str, Any]]) -> Dict:
        """生成专家级质量报告"""
        logger.info("生成专家级质量报告...")
        
        quality_distribution = {"high": 0, "medium": 0, "low": 0}
        detailed_assessments = []
        numeric_scores = []
        
        for idx, qa_item in enumerate(qa_data):
            if "original_quality_assessment" in qa_item:
                # 已有质量评估
                assessment = qa_item["original_quality_assessment"]
            else:
                # 需要评估
                assessment = await self._assess_quality(qa_item)
                if not assessment:
                    continue
            
            overall_quality = assessment.get("quality_rating", {}).get("overall", "medium")
            numeric_score = assessment.get("numeric_score", 0.5)
            
            quality_distribution[overall_quality] += 1
            numeric_scores.append(numeric_score)
            detailed_assessments.append({
                "index": idx,
                "overall_quality": overall_quality,
                "numeric_score": numeric_score,
                "detailed_scores": assessment.get("quality_rating", {}).get("detailed_scores", {}),
                "improvement_suggestions": assessment.get("improvement_suggestions", []),
                "sft_suitable": assessment.get("sft_suitable", False)
            })
        
        avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
        sft_suitable_count = sum(1 for a in detailed_assessments if a.get("sft_suitable", False))
        
        return {
            "total_assessed": len(detailed_assessments),
            "quality_distribution": quality_distribution,
            "average_numeric_score": avg_score,
            "sft_suitable_count": sft_suitable_count,
            "sft_suitable_rate": sft_suitable_count / len(detailed_assessments) if detailed_assessments else 0,
            "detailed_assessments": detailed_assessments,
            "summary": {
                "excellence_rate": quality_distribution["high"] / len(detailed_assessments) if detailed_assessments else 0,
                "improvement_needed_rate": quality_distribution["low"] / len(detailed_assessments) if detailed_assessments else 0,
                "overall_assessment": "excellent" if avg_score >= 0.8 else "good" if avg_score >= 0.6 else "needs_improvement"
            },
            "generated_at": datetime.now().isoformat()
        }


# 兼容性接口
class ArgumentDataProcessor:
    def __init__(self, api_key: str = API_KEY, model: str = MODEL):
        self.processor = TextQAArgumentDataProcessor(api_key, model)
    
    async def enhance_qa_data(self, qa_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.processor.enhance_qa_data_with_quality_driven_strategy(qa_data)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="专家级文本QA数据增强处理")
    parser.add_argument("--input-file", type=str, required=True, help="输入QA数据文件")
    parser.add_argument("--output-file", type=str, required=True, help="输出增强后的QA数据文件")
    parser.add_argument("--quality-report", action="store_true", help="生成质量报告")
    
    args = parser.parse_args()
    
    # 加载输入数据
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    logger.info(f"加载了 {len(input_data)} 个QA对")
    
    # 初始化处理器
    processor = TextQAArgumentDataProcessor()
    
    # 执行专家级增强
    enhanced_data = await processor.enhance_qa_data_with_quality_driven_strategy(input_data)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"专家级增强结果已保存到: {args.output_file}")
    
    # 生成质量报告
    if args.quality_report:
        logger.info("生成专家级质量报告...")
        quality_report = await processor.generate_quality_report(enhanced_data)
        
        quality_report_file = args.output_file.replace('.json', '_expert_quality_report.json')
        with open(quality_report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"SFT适合率: {quality_report['sft_suitable_rate']:.2%}")
        logger.info(f"平均质量评分: {quality_report['average_numeric_score']:.3f}")
        logger.info(f"专家级质量报告已保存到: {quality_report_file}")


if __name__ == "__main__":
    asyncio.run(main())