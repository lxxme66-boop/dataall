# # -*- coding: utf-8 -*-
# # @Time : 2025/5/20 10:40 
# # @Author : dumingyu
# # @File : cot数据质量评估.py
# # @Software: PyCharm


# import os
# import pandas as pd
# from openai import OpenAI
# import datetime
# import json
# import multiprocessing
# import logging
# from tqdm import tqdm

# # 日志配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# time_label = str(datetime.datetime.today())[:10].replace('-', '')


# # 初始化客户端
# client = OpenAI(
#     base_url="http://8.130.143.102:81/v1",  # vLLM API 服务器的地址
#     api_key="EMPTY"  # 如果没有设置 API 密钥，可以使用任意字符串
# )




# quality_evaluation_prompt_v1 = {
#     "role": "system",
#     "content": """
# 作为半导体显示领域的专业质量评估专家，请严格按以下标准评估问答对的质量。评估分为核心维度，每个维度包含具体评估点和示例参考。

# ### 评估维度
# 1. 思维链逻辑质量（权重35%）
#    - 步骤完整性：推理步骤是否覆盖问题所有关键点？是否遗漏必要环节？
#    - 因果连贯性：前后步骤是否存在清晰因果关系？有无逻辑断裂？
#    - 技术参数合理性：工艺参数是否符合物理规律？（例：LTPS退火温度不能超过玻璃转化点）
#    - 错误回溯机制：是否考虑可能故障点？（例：分析Mura缺陷应包含设备精度因素）

# 2. 技术准确度（权重30%）
#    - 材料特性：材料描述是否符合物性？（例：IGZO迁移率范围是否正确）
#    - 制程参数：工艺参数是否匹配行业标准？（例：光刻精度是否满足当前产线能力）
#    - 标准引用：是否准确引用SEMI/SID等国际标准？
#    - 专利技术：技术方案是否规避近期专利？（自动核对2020-2024专利数据库）

# 3. 领域深度（权重20%）
#    - 缺陷机理：是否分析根本原因？（例：亮暗点应关联电致迁移机制）
#    - 技术趋势：是否覆盖最新发展？（例：需提及Micro LED巨量转移技术）
#    - 工艺瓶颈：是否识别关键限制？（例：指出QD-OLED的喷墨打印精度瓶颈）

# 4. 应用价值（权重15%）
#    - 工程可行性：方案是否具备量产实施条件？
#    - 成本优化：是否量化成本效益？（例：应计算采用MMG技术的材料节省）
#    - 良率提升路径：是否提供可验证的改善方案？

# ### 领域关键点（自动核对）
# | 要素类型       | 典型内容示例                  |
# |----------------|------------------------------|
# | 核心材料       | 氧化物TFT, QD材料, LTPO      |
# | 工艺痛点       | 蒸镀均匀性, 水氧阻隔率       |
# | 典型缺陷       | Mura, 亮点/暗点, 热应力翘曲   |

# ### 验证方法
# 1. 参数边界检查：对关键参数进行物理极限校验（例：若声称PPI>1500需验证光学混色距离）
# 2. 时效性验证：技术指标是否被近3年文献更新（自动索引IEEE期刊数据库）
# 3. 成本分解：对降本承诺进行材料/设备/良率因子分解

# ### 输出格式要求（JSON）
# {
#     "quality_rating": {
#         "overall": "high/medium/low",
#         "detailed_scores": {
#             "reasoning_chain": {"score": int, "issues": [str]},
#             "technical_accuracy": {"score": int, "validated": bool},
#             "domain_depth": {"score": int, "benchmark": str}
#         }
#     },
#     "improvement_suggestions": [str]
# }

# ### 待评估样本
# 问题: {question_text}
# 思维链: {reasoning_chain}
# 答案: {answer_text}
# """
# }


# quality_evaluation_prompt_v2 = {
#     "role": "system",
#     "content": """你是一名资深显示技术领域专家。请严格评估以下显示技术相关的问答对是否适合用于监督微调（SFT）的数据集构建。评估需基于以下四个核心维度：

# 1.  **回答相关性 (Relevance)**：回答是否精准聚焦问题核心？是否存在答非所问、偏离主题或遗漏关键点？
# 2.  **逻辑一致性 (Logical Consistency)**：回答的推理过程是否清晰、连贯、无矛盾？是否存在逻辑跳跃、断裂或自相矛盾？
# 3.  **术语使用 (Terminology Usage)**：专业术语的使用是否准确、恰当、完整？是否存在术语误用、滥用、缺失或概念性错误？
# 4.  **事实正确性 (Factual Correctness)**：回答中的技术细节、参数、原理、行业现状等是否符合已知事实和行业共识？是否存在事实性错误或过时信息？

# **总体质量评分标准：**
# *   `low`：**存在严重缺陷**（如明显事实错误、完全偏离主题、逻辑混乱、关键术语错误），**不适合**用于SFT。
# *   `medium`：**存在轻微问题或可优化项**（如部分表述不清、个别术语不严谨、次要逻辑不完美、相关性略有不足），需修改后方可考虑使用。
# *   `high`：**无明显错误**，内容**准确、专业、逻辑清晰、紧扣主题**，**适合**直接用于SFT。

# **你的任务：**
# 1.  对每个维度进行独立评分 (`high`/`medium`/`low`)。
# 2.  给出基于四个维度的**总体质量评分** (`high`/`medium`/`low`)。
# 3.  对于评分非`high`的维度，**必须具体指出**存在的问题及其**类型**（例如：“术语误用：将‘OLED’错误称为‘LED’”；“事实错误：声称当前主流Mini-LED背光分区数普遍超过5000区”）。
# 4.  基于你的专业知识和评估结果，**提供具体、可操作的改进建议**，以提升该问答对的质量。

# **输出格式要求（严格遵循JSON）：**
# {
#     "quality_rating": {
#         "overall": "high/medium/low", // 总体质量评分
#         "detailed_scores": {
#             "Relevance": {"score": "high/medium/low", "issues": ["具体问题描述1", "具体问题描述2", ...]}, // 如无问题，issues为空数组[]
#             "Logical Consistency": {"score": "high/medium/low", "issues": [...]},
#             "Terminology Usage": {"score": "high/medium/low", "issues": [...]},
#             "Factual Correctness": {"score": "high/medium/low", "issues": [...]}
#         }
#     },
#     "improvement_suggestions": ["具体建议1", "具体建议2", ...] // 即使总体是high，也可提供优化建议
# }
# ### 待评估样本
# 问题: {question_text}
# 思维链: {reasoning_chain}
# 答案: {answer_text}
# """
# }


# # 发送单个请求
# def evaluate_qa_quality(question, chain, answer):
#     try:
#         prompt = quality_evaluation_prompt.copy()
#         prompt["content"] = prompt["content"].replace("{question_text}", question) \
#             .replace("{reasoning_chain}", chain) \
#             .replace("{answer_text}", answer)

#         response = client.chat.completions.create(
#             model="qwen3-235b",  # 与 --served-model-name 参数指定的名称一致
#             messages=[prompt],
#             temperature=0.1
#         )
#         # 打印模型的响应
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"请求失败: {str(e)}"


# # ================ 多进程工作函数 ================
# def process_row(args):
#     """
#     处理单个数据行的质量评估任务
#     """
#     row_idx, row_data, server_config = args
#     question, chain, answer = row_data['问题'], row_data['思维链'], row_data['答案']

#     # 1. 初始化客户端（每个进程单独创建）
#     client = OpenAI(
#         base_url=server_config["base_url"],
#         api_key=server_config["api_key"]
#     )

#     try:
#         # 2. 构建评估提示
#         prompt = quality_evaluation_prompt.copy()
#         prompt["content"] = prompt["content"].replace("{question_text}", question) \
#             .replace("{reasoning_chain}", chain) \
#             .replace("{answer_text}", answer)

#         # 3. 发送请求
#         response = client.chat.completions.create(
#             model=server_config["model_name"],
#             messages=[prompt],
#             timeout=90  # 设置超时防止进程卡住
#         )


#         # 4. 解析响应
#         result_text = response.choices[0].message.content
#         if '```json' in result_text:
#             result = json.loads(result_text.split("```json")[-1].strip().replace("```", "").strip())
#         else:
#             result = json.loads(result_text.split("</think>")[-1].strip().replace("```", "").strip())
#         # 5. 更新行数据
#         row_data['quality_rating'] = result['quality_rating']['overall']
#         row_data['detailed_scores'] = str(result['quality_rating']['detailed_scores'])
#         return (row_idx, row_data)

#     except Exception as e:
#         logger.error(f"处理行 {row_idx} 时出错: {str(e)}")
#         # 保留原始行数据并标记错误
#         row_data['quality_rating'] = "ERROR"
#         row_data['detailed_scores'] = str(e)
#         return (row_idx, row_data)





# if __name__ == '__main__':

#     # 0. 服务器配置（避免重复传递）
#     server_config = {
#         "base_url": "http://8.130.143.102:81/v1",
#         "api_key": "EMPTY",
#         "model_name": "qwen3-235b"
#     }

#     # 1. 加载源数据
#     file_name = "/mnt/workspace/LLM/ldd/sft/评估数据输入.xlsx"
#     data = pd.read_excel(file_name)
#     logger.info(f"加载数据完成，共 {data.shape[0]} 行")

#     # 2. 准备多进程处理
#     num_workers = 20
#     logger.info(f"启用 {num_workers} 个工作进程")


#     data = data.iloc[10848:,]


#     # 3. 创建进程池
#     with multiprocessing.Pool(processes=num_workers) as pool:
#         # 准备任务参数 [(行索引, 行数据, 服务器配置), ...]
#         task_args = [(idx, row.copy(), server_config) for idx, row in data.iterrows()]

#         # 4. 并行处理并收集结果
#         results = []
#         for result in tqdm(pool.imap(process_row, task_args), total=len(task_args)):
#             results.append(result)

#     # 5. 按原始顺序更新数据
#     logger.info("开始更新结果数据")
#     for row_idx, updated_row in results:
#         data.loc[row_idx,'quality_rating'] = updated_row['quality_rating']
#         data.loc[row_idx, 'detailed_scores'] = updated_row['detailed_scores']

#     # 6. 保存结果
#     save_dir = './reslut'
#     save_name = '增强前的评估结果.xlsx'
#     save_path = os.path.join(save_dir, save_name)

#     with pd.ExcelWriter(save_path) as writer:
#         data.to_excel(writer, index=False)
#     logger.info(f"结果已保存到: {save_path}")

# -*- coding: utf-8 -*-
# @Time : 2025/5/20 10:40 
# @Author : dumingyu
# @File : cot数据质量评估.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time : 2025/5/20 10:40 
# @Author : dumingyu
# @File : cot数据质量评估.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time : 2025/5/20 10:40 
# @Author : dumingyu
# @File : cot数据质量评估.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time : 2025/5/20 10:40 
# @Author : dumingyu
# @File : cot数据质量评估.py
# @Software: PyCharm


# #带文本
# import os
# import pandas as pd
# from openai import OpenAI
# import datetime
# import json
# import multiprocessing
# import logging
# from tqdm import tqdm
# import time
# import requests
# import re
# import traceback

# # 日志配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# time_label = str(datetime.datetime.today())[:10].replace('-', '')

# # 使用v2版本的评估提示模板
# quality_evaluation_prompt = {
#     "role": "system",
#     "content": """你是一名资深显示技术领域专家。请先仔细思考，然后严格评估以下显示技术相关的问答对是否适合用于监督微调（SFT）的数据集构建。
# [问题]：
# {question_text}
# [文本]：
# {reasoning_text}
# [答案]：
# {answer_text}
# ##核心维度，评估需基于以下六个核心维度：

# 0.  **问题通用性(quedtion)**: 问题是否是针对文本生成的？是否能怎么文本回答？问题是否具有实际意义？问题是否通用？（严格执行）
# 1.  **回答相关性 (Relevance)**：回答是否精准聚焦问题核心？是否存在答非所问、偏离主题或遗漏关键点？是否答案只是仅引导句未提供实质性内容？答案是否基于论文原文回答的，即是否能在上传的本文中找到出处？（严格执行）
# 2.  **逻辑一致性 (Logical Consistency)**：回答的推理过程是否清晰、连贯、无矛盾？是否存在逻辑跳跃、断裂或自相矛盾？是否存在答案中断？答案是否和问题不相关？答案是否只是泛泛而谈？
# 3.  **术语使用 (Terminology Usage)**：专业术语的使用是否准确、恰当、完整？是否存在术语误用、滥用、缺失或概念性错误？
# 4.  **事实正确性 (Factual Correctness)**：回答中的技术细节、参数、原理、行业现状等是否符合已知事实和行业共识？是否存在事实性错误或过时信息？是否具有通用性？
# 5.  **原文本验证(validate against the original text)**:答案中标记了该答案的出处，请仔细阅读原文和答案出处，思考答案出处是否真的来自原文？(严格执行)

# ##总体质量评分标准：
# *   `low`：**存在严重缺陷**（如问题通用性低质量、明显事实错误、完全偏离主题、逻辑混乱、关键术语错误、答案不完整（仅引导句）未提供实质性内容（答案完整性不过关）、答案不是出自原文（不能在原文中找到出处）、答案中标记的出处不是出在原文），**不适合**用于SFT,。
# *   `medium`：**存在轻微问题或可优化项**（如部分表述不清、个别术语不严谨、次要逻辑不完美、相关性略有不足、通用性略有不足、主要逻辑基于论文回答、出处的核心逻辑出自原文），需修改后方可考虑使用。
# *   `high`：**无明显错误**，内容**(问题通用性高质量、准确、专业、逻辑清晰、紧扣主题、准确且完整地回答了问题（严格执行）、基于论文回答（严格执行）、答案中的标记的出处完全出自原文、具有通用性)**，**适合**直接用于SFT。

# ##你的任务：
# 1.  对每个维度进行独立评分 (`high`/`medium`/`low`)。
# 2.  给出基于六个维度和新加要求的**总体质量评分** (`high`/`medium`/`low`)，其中若答案完整性不过关（只是仅引导句未提供实质性内容）、答案完全不是基于原文回答的、原文验证中的出处实际和原文不是很相关、答案为无法作答、答案中答案出处没有、问题通用性不过关，满足其一直接一票否决，判为低质量（严格执行）。
# 3.  对于评分非`high`的维度，**必须具体指出**存在的问题及其**类型**（例如："术语误用：将'OLED'错误称为'LED'"；"事实错误：声称当前主流Mini-LED背光分区数普遍超过5000区"）。
# 4.  基于你的专业知识和评估结果，**提供具体、可操作的改进建议**，以提升该问答对的质量。

# #输出格式要求(严格遵循JSON):
# {
#     "quality_rating": {
#         "overall": "high/medium/low", // 总体质量评分
#         "detailed_scores": {
#             "Relevance": {"score": "high/medium/low", "issues": ["具体问题描述1", "具体问题描述2", ...]}, // 如无问题，issues为空数组[]
#             "Logical Consistency": {"score": "high/medium/low", "issues": [...]},
#             "Terminology Usage": {"score": "high/medium/low", "issues": [...]},
#             "Factual Correctness": {"score": "high/medium/low", "issues": [...]}
#             "validate against the original text":{"score": "high/medium/low", "issues": [...]}
#         }
#     },
#     "improvement_suggestions": ["具体建议1", "具体建议2", ...] // 即使总体是high，也可提供优化建议
# }


# """
# }


# # ================ 多进程工作函数 ================
# def process_row(args):
#     """
#     处理单个数据行的质量评估任务
#     """
#     row_idx, row_data, server_config = args
#     # 使用实际表格中的列名
#     question = row_data['问题'] if '问题' in row_data else row_data.get('问题"', '')
#     chain = row_data['思维链'] if '思维链' in row_data else row_data.get('使用CoT', '')
#     answer = row_data['答案'] if '答案' in row_data else row_data.get('答案"', '')
    
#     # 处理可能的空值
#     question = str(question) if pd.notna(question) else ""
#     chain = str(chain) if pd.notna(chain) else ""
#     answer = str(answer) if pd.notna(answer) else ""

#     # 1. 初始化客户端（每个进程单独创建）
#     client = OpenAI(
#         base_url=server_config["base_url"],
#         api_key=server_config["api_key"]
#     )

#     try:
#         # 2. 构建评估提示
#         prompt = quality_evaluation_prompt.copy()
#         prompt["content"] = prompt["content"].replace("{question_text}", question) \
#             .replace("{reasoning_text}", chain) \
#             .replace("{answer_text}", answer)

#         # 3. 发送请求（带重试机制）
#         max_retries = 5
#         retry_delay = 5  # 初始重试延迟（秒）
#         response = None
        
#         for attempt in range(max_retries):
#             try:
#                 response = client.chat.completions.create(
#                     model=server_config["model_name"],
#                     messages=[prompt],
#                     timeout=300  # 设置较长的超时时间
#                 )
#                 break  # 请求成功，跳出重试循环
#             except Exception as e:
#                 if attempt < max_retries - 1:
#                     logger.warning(f"行 {row_idx} 请求失败，第{attempt+1}次重试: {str(e)}")
#                     time.sleep(retry_delay * (attempt + 1))  # 指数退避策略
#                 else:
#                     raise e  # 最后一次重试仍然失败，抛出异常

#         # 4. 解析响应
#         result_text = response.choices[0].message.content
        
#         # 增强JSON解析 - 处理各种可能的格式
#         result = None
#         try:
#             # 尝试1: 直接解析整个响应
#             result = json.loads(result_text)
#         except json.JSONDecodeError:
#             # 尝试2: 提取可能的JSON部分
#             try:
#                 # 处理代码块格式
#                 if '```json' in result_text:
#                     json_str = re.search(r'```json(.*?)```', result_text, re.DOTALL)
#                     if json_str:
#                         json_str = json_str.group(1).strip()
#                         result = json.loads(json_str)
#                 # 处理思考格式
#                 elif '</think>' in result_text:
#                     json_str = result_text.split('</think>', 1)[1].strip()
#                     result = json.loads(json_str)
#                 # 尝试提取JSON对象
#                 else:
#                     json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
#                     if json_match:
#                         json_str = json_match.group(0).strip()
#                         # 处理可能的多个JSON对象
#                         if json_str.count('{') > 1:
#                             # 取第一个完整的JSON对象
#                             start_idx = json_str.find('{')
#                             end_idx = json_str.rfind('}', 0, json_str.find('}', start_idx) + 1)
#                             json_str = json_str[start_idx:end_idx+1]
#                         result = json.loads(json_str)
#             except:
#                 # 尝试3: 作为最后的手段，提取JSON部分
#                 try:
#                     start_idx = result_text.find('{')
#                     end_idx = result_text.rfind('}')
#                     if start_idx >= 0 and end_idx > start_idx:
#                         json_str = result_text[start_idx:end_idx+1]
#                         result = json.loads(json_str)
#                 except Exception as e:
#                     logger.error(f"JSON解析失败，响应文本: {result_text[:500]}...")
#                     raise ValueError(f"无法解析JSON响应: {str(e)}")
        
#         # 5. 更新行数据
#         if result and 'quality_rating' in result and 'overall' in result['quality_rating']:
#             row_data['quality_rating'] = result['quality_rating']['overall']
#             row_data['detailed_scores'] = str(result['quality_rating'].get('detailed_scores', ''))
#             row_data['improvement_suggestions'] = str(result.get('improvement_suggestions', []))
#         else:
#             raise ValueError("响应中缺少必要的quality_rating字段")
            
#         return (row_idx, row_data)

#     except Exception as e:
#         logger.error(f"处理行 {row_idx} 时出错: {str(e)}")
#         logger.debug(f"错误详情: {traceback.format_exc()}")
#         # 保留原始行数据并标记错误
#         row_data['quality_rating'] = "ERROR"
#         row_data['detailed_scores'] = str(e)
#         return (row_idx, row_data)

# def check_server_health(base_url):
#     """检查服务器是否可用"""
#     health_url = base_url.replace("/v1", "/health")
#     try:
#         response = requests.get(health_url, timeout=10)
#         if response.status_code == 200:
#             logger.info(f"服务器健康检查通过: {health_url}")
#             return True
#         else:
#             logger.warning(f"服务器健康检查失败: 状态码 {response.status_code}")
#             return False
#     except Exception as e:
#         logger.error(f"服务器健康检查错误: {str(e)}")
#         return False

# def extract_json_from_response(text):
#     """从响应文本中提取JSON内容"""
#     # 尝试直接解析
#     try:
#         return json.loads(text)
#     except:
#         pass
    
#     # 尝试提取代码块
#     if '```json' in text:
#         try:
#             json_match = re.search(r'```json(.*?)```', text, re.DOTALL)
#             if json_match:
#                 return json.loads(json_match.group(1).strip())
#         except:
#             pass
    
#     # 尝试提取思考块
#     if '</think>' in text:
#         try:
#             json_part = text.split('</think>', 1)[1].strip()
#             return json.loads(json_part)
#         except:
#             pass
    
#     # 尝试提取JSON对象
#     try:
#         json_match = re.search(r'\{.*\}', text, re.DOTALL)
#         if json_match:
#             return json.loads(json_match.group(0).strip())
#     except:
#         pass
    
#     # 作为最后手段，尝试提取第一个JSON对象
#     try:
#         start_idx = text.find('{')
#         end_idx = text.rfind('}')
#         if start_idx >= 0 and end_idx > start_idx:
#             return json.loads(text[start_idx:end_idx+1])
#     except:
#         pass
    
#     raise ValueError("无法从响应中提取有效的JSON")

# if __name__ == '__main__':
#     # 0. 服务器配置 - 使用vLLM HTTP配置
#     server_config = {
#         "base_url": os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1"),
#         "api_key": "EMPTY",
#         "model_name": os.getenv("VLLM_MODEL_NAME", "qwen-vllm")
#     }
    
#     logger.info(f"使用vLLM服务器: {server_config['base_url']}")
#     logger.info(f"使用模型: {server_config['model_name']}")
    
#     # 检查服务器健康状态
#     if not check_server_health(server_config["base_url"]):
#         logger.error("服务器不可用，请检查vLLM服务是否已启动")
#         exit(1)

#     # 1. 加载源数据
#     file_name = "/mnt/workspace/LLM/ldd/sft/input/Qwen325评估数据v2版输入-无增强.xlsx"
#     print(file_name)
#     data = pd.read_excel(file_name)
#     logger.info(f"加载数据完成，共 {data.shape[0]} 行")
#     logger.info(f"数据列名: {list(data.columns)}")
    
#     # 添加结果列
#     data['quality_rating'] = ""
#     data['detailed_scores'] = ""
#     data['improvement_suggestions'] = ""

#     # 2. 准备多进程处理
#     num_workers = min(8, os.cpu_count())  # 减少并发数以提高稳定性
#     logger.info(f"启用 {num_workers} 个工作进程")

#     # 3. 创建进程池
#     with multiprocessing.Pool(processes=num_workers) as pool:
#         # 准备任务参数 [(行索引, 行数据, 服务器配置), ...]
#         task_args = [(idx, row.copy(), server_config) for idx, row in data.iterrows()]

#         # 4. 并行处理并收集结果（带进度条）
#         results = []
#         progress_bar = tqdm(total=len(task_args), desc="评估进度", unit="行")
        
#         try:
#             for result in pool.imap(process_row, task_args):
#                 results.append(result)
#                 row_idx, row_data = result
#                 progress_bar.update(1)
#                 progress_bar.set_postfix_str(f"状态: {row_data['quality_rating']}")
#         except Exception as e:
#             logger.error(f"处理过程中发生错误: {str(e)}")
#             progress_bar.close()
#             pool.terminate()  # 终止所有进程
#             raise e
        
#         progress_bar.close()

#     # 5. 按原始顺序更新数据
#     logger.info("开始更新结果数据")
#     for row_idx, updated_row in results:
#         data.loc[row_idx, 'quality_rating'] = updated_row['quality_rating']
#         data.loc[row_idx, 'detailed_scores'] = updated_row['detailed_scores']
#         data.loc[row_idx, 'improvement_suggestions'] = updated_row.get('improvement_suggestions', '')

#     # 6. 保存结果
#     save_dir = './result'
#     os.makedirs(save_dir, exist_ok=True)
#     save_name = f'数据质量评估结果_{time_label}—v2无增强版.xlsx'
#     save_path = os.path.join(save_dir, save_name)

#     with pd.ExcelWriter(save_path) as writer:
#         data.to_excel(writer, index=False)
#     logger.info(f"结果已保存到: {save_path}")
    
#     # 打印摘要统计
#     if 'quality_rating' in data.columns:
#         rating_counts = data['quality_rating'].value_counts()
#         logger.info("\n===== 评估结果摘要 =====")
#         logger.info(f"高质量 (high): {rating_counts.get('high', 0)} 行")
#         logger.info(f"中等质量 (medium): {rating_counts.get('medium', 0)} 行")
#         logger.info(f"低质量 (low): {rating_counts.get('low', 0)} 行")
#         logger.info(f"错误行: {rating_counts.get('ERROR', 0)} 行")
#     else:
#         logger.warning("未找到质量评级列，无法生成摘要统计")


# ----------------考虑文本
# -*- coding: utf-8 -*-
# @Time : 2025/5/20 10:40 
# @Author : dumingyu
# @File : cot数据质量评估_火山版.py
# @Software: PyCharm

import os
import pandas as pd
from volcenginesdkarkruntime import Ark
import datetime
import json
import multiprocessing
import logging
from tqdm import tqdm

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

time_label = str(datetime.datetime.today())[:10].replace('-', '')

# 初始化火山引擎客户端
client = Ark(
    api_key="5a032496-1268-4e6f-b6ee-a9affc6b5469",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

quality_evaluation_prompt_v1 = {
    "role": "system",
    "content": """
作为半导体显示领域的专业质量评估专家，请严格按以下标准评估问答对的质量。评估分为核心维度，每个维度包含具体评估点和示例参考。

### 评估维度
1. 思维链逻辑质量（权重35%）
   - 步骤完整性：推理步骤是否覆盖问题所有关键点？是否遗漏必要环节？
   - 因果连贯性：前后步骤是否存在清晰因果关系？有无逻辑断裂？
   - 技术参数合理性：工艺参数是否符合物理规律？（例：LTPS退火温度不能超过玻璃转化点）
   - 错误回溯机制：是否考虑可能故障点？（例：分析Mura缺陷应包含设备精度因素）

2. 技术准确度（权重30%）
   - 材料特性：材料描述是否符合物性？（例：IGZO迁移率范围是否正确）
   - 制程参数：工艺参数是否匹配行业标准？（例：光刻精度是否满足当前产线能力）
   - 标准引用：是否准确引用SEMI/SID等国际标准？
   - 专利技术：技术方案是否规避近期专利？（自动核对2020-2024专利数据库）

3. 领域深度（权重20%）
   - 缺陷机理：是否分析根本原因？（例：亮暗点应关联电致迁移机制）
   - 技术趋势：是否覆盖最新发展？（例：需提及Micro LED巨量转移技术）
   - 工艺瓶颈：是否识别关键限制？（例：指出QD-OLED的喷墨打印精度瓶颈）

4. 应用价值（权重15%）
   - 工程可行性：方案是否具备量产实施条件？
   - 成本优化：是否量化成本效益？（例：应计算采用MMG技术的材料节省）
   - 良率提升路径：是否提供可验证的改善方案？

### 领域关键点（自动核对）
| 要素类型       | 典型内容示例                  |
|----------------|------------------------------|
| 核心材料       | 氧化物TFT, QD材料, LTPO      |
| 工艺痛点       | 蒸镀均匀性, 水氧阻隔率       |
| 典型缺陷       | Mura, 亮点/暗点, 热应力翘曲   |

### 验证方法
1. 参数边界检查：对关键参数进行物理极限校验（例：若声称PPI>1500需验证光学混色距离）
2. 时效性验证：技术指标是否被近3年文献更新（自动索引IEEE期刊数据库）
3. 成本分解：对降本承诺进行材料/设备/良率因子分解

### 输出格式要求（JSON）
{
    "quality_rating": {
        "overall": "high/medium/low",
        "detailed_scores": {
            "reasoning_chain": {"score": int, "issues": [str]},
            "technical_accuracy": {"score": int, "validated": bool},
            "domain_depth": {"score": int, "benchmark": str}
        }
    },
    "improvement_suggestions": [str]
}

### 待评估样本
问题: {question_text}
思维链: {reasoning_chain}
答案: {answer_text}
"""
}
quality_evaluation_prompt = {
    "role": "system",
    "content": """你是一名资深显示技术领域专家。请先仔细思考，然后严格评估以下显示技术相关的问答对是否适合用于监督微调（SFT）的数据集构建。
[问题]：
{question_text}
[文本]：
{reasoning_text}
[答案]：
{answer_text}
##核心维度，评估需基于以下六个核心维度：

0.  **问题通用性(quedtion)**: 问题是否是针对文本生成的？是否能怎么文本回答？问题是否具有实际意义？问题是否通用？（严格执行）
1.  **回答相关性 (Relevance)**：回答是否精准聚焦问题核心？是否存在答非所问、偏离主题或遗漏关键点？是否答案只是仅引导句未提供实质性内容？答案是否基于论文原文回答的，即是否能在上传的本文中找到出处？（严格执行）
2.  **逻辑一致性 (Logical Consistency)**：回答的推理过程是否清晰、连贯、无矛盾？是否存在逻辑跳跃、断裂或自相矛盾？是否存在答案中断？答案是否和问题不相关？答案是否只是泛泛而谈？
3.  **术语使用 (Terminology Usage)**：专业术语的使用是否准确、恰当、完整？是否存在术语误用、滥用、缺失或概念性错误？
4.  **事实正确性 (Factual Correctness)**：回答中的技术细节、参数、原理、行业现状等是否符合已知事实和行业共识？是否存在事实性错误或过时信息？是否具有通用性？
5.  **原文本验证(validate against the original text)**:答案中标记了该答案的出处，请仔细阅读原文和答案出处，思考答案出处是否真的来自原文？(严格执行)

##总体质量评分标准：
*   `low`：**存在严重缺陷**（如问题通用性低质量、明显事实错误、完全偏离主题、逻辑混乱、关键术语错误、答案不完整（仅引导句）未提供实质性内容（答案完整性不过关）、答案不是出自原文（不能在原文中找到出处）、答案中标记的出处不是出在原文），**不适合**用于SFT,。
*   `medium`：**存在轻微问题或可优化项**（如部分表述不清、个别术语不严谨、次要逻辑不完美、相关性略有不足、通用性略有不足、主要逻辑基于论文回答、出处的核心逻辑出自原文），需修改后方可考虑使用。
*   `high`：**无明显错误**，内容**(问题通用性高质量、准确、专业、逻辑清晰、紧扣主题、准确且完整地回答了问题（严格执行）、基于论文回答（严格执行）、答案中的标记的出处完全出自原文、具有通用性)**，**适合**直接用于SFT。

##你的任务：
1.  对每个维度进行独立评分 (`high`/`medium`/`low`)。
2.  给出基于六个维度和新加要求的**总体质量评分** (`high`/`medium`/`low`)，其中若答案完整性不过关（只是仅引导句未提供实质性内容）、答案完全不是基于原文回答的、原文验证中的出处实际和原文不是很相关、答案为无法作答、答案中答案出处没有、问题通用性不过关，满足其一直接一票否决，判为低质量（严格执行）。
3.  对于评分非`high`的维度，**必须具体指出**存在的问题及其**类型**（例如："术语误用：将'OLED'错误称为'LED'"；"事实错误：声称当前主流Mini-LED背光分区数普遍超过5000区"）。
4.  基于你的专业知识和评估结果，**提供具体、可操作的改进建议**，以提升该问答对的质量。

#输出格式要求(严格遵循JSON):
{
    "quality_rating": {
        "overall": "high/medium/low", // 总体质量评分
        "detailed_scores": {
            "Relevance": {"score": "high/medium/low", "issues": ["具体问题描述1", "具体问题描述2", ...]}, // 如无问题，issues为空数组[]
            "Logical Consistency": {"score": "high/medium/low", "issues": [...]},
            "Terminology Usage": {"score": "high/medium/low", "issues": [...]},
            "Factual Correctness": {"score": "high/medium/low", "issues": [...]}
            "validate against the original text":{"score": "high/medium/low", "issues": [...]}
        }
    },
    "improvement_suggestions": ["具体建议1", "具体建议2", ...] // 即使总体是high，也可提供优化建议
}


"""
}

quality_evaluation_prompt_v1 = {
    "role": "system",
    "content": """你是一名资深显示技术领域专家。请严格评估以下显示技术相关的问答对是否适合用于监督微调（SFT）的数据集构建。
[问题]:
{question_text}
[文本]:
{reasoning_text}
[答案]:
{answer_text}
##核心维度，评估需基于以下四个核心维度：

1.  **回答相关性 (Relevance)**：回答是否精准聚焦问题核心？是否存在答非所问、偏离主题或遗漏关键点？是否只回答了部分问题（回答不完整）？答案是否有出处，即是否能在上传的[本文]中找到出处？
2.  **逻辑一致性 (Logical Consistency)**：回答的推理过程是否清晰、连贯、无矛盾？是否存在逻辑跳跃、断裂或自相矛盾？是否存在答案中断？
3.  **术语使用 (Terminology Usage)**：专业术语的使用是否准确、恰当、完整？是否存在术语误用、滥用、缺失或概念性错误？
4.  **事实正确性 (Factual Correctness)**：回答中的技术细节、参数、原理、行业现状等是否符合已知事实和行业共识？是否存在事实性错误或过时信息？


##总体质量评分标准：
*   `low`：**存在严重缺陷**（如明显事实错误、完全偏离主题、逻辑混乱、关键术语错误、答案不完整（仅引导句）未提供实质性内容（答案完整性不过关）），**不适合**用于SFT,。
*   `medium`：**存在轻微问题或可优化项**（如部分表述不清、个别术语不严谨、次要逻辑不完美、相关性略有不足），需修改后方可考虑使用。
*   `high`：**无明显错误**，内容**准确、专业、逻辑清晰、紧扣主题、准确且完整地回答了问题**，**适合**直接用于SFT。

##你的任务：
1.  对每个维度进行独立评分 (`high`/`medium`/`low`)。
2.  给出基于四个维度和新加要求的**总体质量评分** (`high`/`medium`/`low`)，若答案完整性不过关，直接一票否决，判为低质量。
3.  对于评分非`high`的维度，**必须具体指出**存在的问题及其**类型**（例如："术语误用：将'OLED'错误称为'LED'"；"事实错误：声称当前主流Mini-LED背光分区数普遍超过5000区"）。
4.  基于你的专业知识和评估结果，**提供具体、可操作的改进建议**，以提升该问答对的质量。

#输出格式要求(严格遵循JSON):
{
    "quality_rating": {
        "overall": "high/medium/low", // 总体质量评分
        "detailed_scores": {
            "Relevance": {"score": "high/medium/low", "issues": ["具体问题描述1", "具体问题描述2", ...]}, // 如无问题，issues为空数组[]
            "Logical Consistency": {"score": "high/medium/low", "issues": [...]},
            "Terminology Usage": {"score": "high/medium/low", "issues": [...]},
            "Factual Correctness": {"score": "high/medium/low", "issues": [...]}
        }
    },
    "improvement_suggestions": ["具体建议1", "具体建议2", ...] // 即使总体是high，也可提供优化建议
}

"""
}



# 选择使用的prompt版本
quality_evaluation_prompt = quality_evaluation_prompt

# 发送单个请求
def evaluate_qa_quality(question, chain, answer):
    try:
        prompt = quality_evaluation_prompt.copy()
        prompt["content"] = prompt["content"].replace("{question_text}", question) \
            .replace("{reasoning_chain}", chain) \
            .replace("{answer_text}", answer)

        response = client.chat.completions.create(
            model="ep-20250813144949-kchv2",  # 使用您提供的模型端点ID
            messages=[prompt],
            temperature=0.1
        )
        # 打印模型的响应
        return response.choices[0].message.content
    except Exception as e:
        return f"请求失败: {str(e)}"

# ================ 多进程工作函数 ================
def process_row(args):
    """
    处理单个数据行的质量评估任务
    """
    row_idx, row_data, server_config = args
    question, chain, answer = row_data['问题'], row_data['思维链'], row_data['答案']

    # 1. 初始化客户端（每个进程单独创建）
    client = Ark(
        api_key=server_config["api_key"],
        base_url=server_config["base_url"]
    )

    try:
        # 2. 构建评估提示
        prompt = quality_evaluation_prompt.copy()
        prompt["content"] = prompt["content"].replace("{question_text}", question) \
            .replace("{reasoning_chain}", chain) \
            .replace("{answer_text}", answer)

        # 3. 发送请求
        response = client.chat.completions.create(
            model=server_config["model_name"],
            messages=[prompt],
            temperature=0.1
        )

        # 4. 解析响应
        result_text = response.choices[0].message.content
        logger.info(f"行 {row_idx} 收到响应，长度: {len(result_text)}")
        
        try:
            # 更健壮的JSON提取方法
            if '```json' in result_text:
                # 提取```json和```之间的内容
                json_start = result_text.find('```json') + 7
                json_end = result_text.find('```', json_start)
                if json_end == -1:
                    json_end = len(result_text)
                json_text = result_text[json_start:json_end].strip()
            elif '{' in result_text and '}' in result_text:
                # 提取第一个完整的JSON对象
                start = result_text.find('{')
                brace_count = 0
                end = start
                for i, char in enumerate(result_text[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                json_text = result_text[start:end]
            else:
                json_text = result_text.strip()
            
            # 清理可能的多余内容
            json_text = json_text.strip()
            if json_text.endswith('```'):
                json_text = json_text[:-3].strip()
                
            logger.debug(f"行 {row_idx} 提取的JSON: {json_text[:200]}...")
            
            result = json.loads(json_text)
            logger.info(f"行 {row_idx} JSON解析成功")
            
        except json.JSONDecodeError as e:
            logger.error(f"行 {row_idx} JSON解析失败: {e}")
            logger.error(f"提取的JSON文本: {json_text[:500]}...")
            
            # 尝试修复常见的JSON问题
            try:
                # 移除可能的注释
                import re
                cleaned_json = re.sub(r'//.*?\n', '', json_text)
                cleaned_json = re.sub(r'/\*.*?\*/', '', cleaned_json, flags=re.DOTALL)
                result = json.loads(cleaned_json)
                logger.info(f"行 {row_idx} JSON修复后解析成功")
            except:
                logger.error(f"行 {row_idx} JSON修复失败，使用默认值")
                # 使用默认值
                result = {
                    "quality_rating": {
                        "overall": "ERROR",
                        "detailed_scores": {"error": f"JSON解析失败: {str(e)}"}
                    }
                }
        except Exception as e:
            logger.error(f"行 {row_idx} 处理异常: {e}")
            result = {
                "quality_rating": {
                    "overall": "ERROR", 
                    "detailed_scores": {"error": f"处理异常: {str(e)}"}
                }
            }
        
        # 5. 更新行数据
        row_data['quality_rating'] = result['quality_rating']['overall']
        row_data['detailed_scores'] = str(result['quality_rating']['detailed_scores'])
        logger.info(f"行 {row_idx} 处理完成，评分: {row_data['quality_rating']}")
        return (row_idx, row_data)

    except Exception as e:
        logger.error(f"处理行 {row_idx} 时出错: {str(e)}")
        # 保留原始行数据并标记错误
        row_data['quality_rating'] = "ERROR"
        row_data['detailed_scores'] = str(e)
        return (row_idx, row_data)

if __name__ == '__main__':
    # 0. 服务器配置
    server_config = {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": "5a032496-1268-4e6f-b6ee-a9affc6b5469",
        "model_name": "ep-20250813144949-kchv2"
    }

    # 1. 加载源数据
    file_name = "/mnt/workspace/LLM/ldd/sft/input/high_quality_v2.xlsx"
    data = pd.read_excel(file_name)
    logger.info(f"加载数据完成，共 {data.shape[0]} 行")

    # 检查数据切片 - 处理所有数据
    original_data = data.copy()
    data = data.copy()  # 处理全部数据
    logger.info(f"准备处理数据: {data.shape[0]} 行")
    
    # 检查必要列是否存在
    required_columns = ['问题', '思维链', '答案']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"缺少必要列: {missing_columns}")
        logger.info(f"数据列名: {list(data.columns)}")
        exit(1)

    # 2. 准备多进程处理
    num_workers = min(5, data.shape[0])  # 减少进程数，避免过多空闲进程
    logger.info(f"启用 {num_workers} 个工作进程")

    # 先测试单个请求是否正常
    if data.shape[0] > 0:
        test_row = data.iloc[0]
        logger.info("测试单个请求...")
        try:
            test_result = evaluate_qa_quality(
                test_row['问题'], 
                test_row['思维链'], 
                test_row['答案']
            )
            logger.info(f"测试请求成功: {test_result[:100]}...")
        except Exception as e:
            logger.error(f"测试请求失败: {str(e)}")
            exit(1)

    # 3. 创建进程池
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # 准备任务参数 [(行索引, 行数据, 服务器配置), ...]
            task_args = [(idx, row.to_dict(), server_config) for idx, row in data.iterrows()]
            logger.info(f"准备处理 {len(task_args)} 个任务")

            # 4. 并行处理并收集结果
            results = []
            completed_count = 0
            for result in tqdm(pool.imap(process_row, task_args), total=len(task_args), desc="处理进度"):
                results.append(result)
                completed_count += 1
                if completed_count % 1 == 0:  # 每完成1个就打印一次
                    logger.info(f"已完成 {completed_count}/{len(task_args)} 个任务")

        logger.info(f"所有任务完成，共收集到 {len(results)} 个结果")
        
    except Exception as e:
        logger.error(f"多进程处理出错: {str(e)}")
        exit(1)

    # 5. 按原始顺序更新数据
    logger.info("开始更新结果数据")
    updated_count = 0
    for row_idx, updated_row in results:
        if row_idx in data.index:
            data.loc[row_idx,'quality_rating'] = updated_row['quality_rating']
            data.loc[row_idx, 'detailed_scores'] = updated_row['detailed_scores']
            updated_count += 1
        else:
            logger.warning(f"索引 {row_idx} 不在数据中")
    
    logger.info(f"成功更新 {updated_count} 行数据")

    # 6. 保存结果
    save_dir = './result'
    save_name = f'v2版增强前的评估结果_火山版_{time_label}.xlsx'
    save_path = os.path.join(save_dir, save_name)

    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    # 统计质量评分分布
    if 'quality_rating' in data.columns:
        quality_counts = data['quality_rating'].value_counts()
        logger.info("=" * 50)
        logger.info("质量评分统计结果:")
        logger.info("=" * 50)
        
        high_count = quality_counts.get('high', 0)
        medium_count = quality_counts.get('medium', 0)
        low_count = quality_counts.get('low', 0)
        error_count = quality_counts.get('ERROR', 0)
        
        total_evaluated = high_count + medium_count + low_count
        
        logger.info(f"📊 HIGH (高质量):   {high_count:4d} 条 ({high_count/len(data)*100:.1f}%)")
        logger.info(f"📊 MEDIUM (中等质量): {medium_count:4d} 条 ({medium_count/len(data)*100:.1f}%)")
        logger.info(f"📊 LOW (低质量):    {low_count:4d} 条 ({low_count/len(data)*100:.1f}%)")
        if error_count > 0:
            logger.info(f"❌ ERROR (处理失败): {error_count:4d} 条 ({error_count/len(data)*100:.1f}%)")
        logger.info(f"✅ 总计已评估:      {total_evaluated:4d} 条")
        logger.info("=" * 50)
        
        # 保存统计摘要（修复JSON序列化问题）
        summary_stats = {
            'total_samples': int(len(data)),
            'high_quality': int(high_count),
            'medium_quality': int(medium_count),
            'low_quality': int(low_count),
            'error_count': int(error_count),
            'high_percentage': float(round(high_count/len(data)*100, 2)),
            'medium_percentage': float(round(medium_count/len(data)*100, 2)),
            'low_percentage': float(round(low_count/len(data)*100, 2))
        }
        
        # 保存统计结果到JSON
        stats_path = os.path.join(save_dir, f'质量评估统计_火山版_{time_label}.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计结果已保存到: {stats_path}")
    else:
        logger.warning("未找到quality_rating列，无法统计")

    # 检查数据是否为空
    if data.empty:
        logger.error("数据为空，无法保存")
        exit(1)
        
    try:
        with pd.ExcelWriter(save_path) as writer:
            data.to_excel(writer, index=False)
        logger.info(f"结果已保存到: {save_path}")
        logger.info(f"保存的数据形状: {data.shape}")
    except Exception as e:
        logger.error(f"保存文件失败: {str(e)}")
        # 尝试保存为CSV作为备选
        csv_path = save_path.replace('.xlsx', '.csv')
        data.to_csv(csv_path, index=False)
        logger.info(f"已保存为CSV格式: {csv_path}")


#-------------------无文本
# import os
# import pandas as pd
# from volcenginesdkarkruntime import Ark
# import datetime
# import json
# import multiprocessing
# import logging
# from tqdm import tqdm

# # 日志配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# time_label = str(datetime.datetime.today())[:10].replace('-', '')

# # 初始化火山引擎客户端
# client = Ark(
#     api_key="5a032496-1268-4e6f-b6ee-a9affc6b5469",
#     base_url="https://ark.cn-beijing.volces.com/api/v3",
# )

# # 更新后的质量评估提示（移除思维链）
# quality_evaluation_prompt = {
#     "role": "system",
#     "content": """你是一名资深显示技术领域专家。请严格评估以下显示技术相关的问答对是否适合用于监督微调（SFT）的数据集构建。评估需基于以下四个核心维度：

# 1.  **回答相关性 (Relevance)**：回答是否精准聚焦问题核心？是否存在答非所问、偏离主题或遗漏关键点？
# 2.  **逻辑一致性 (Logical Consistency)**：回答的推理过程是否清晰、连贯、无矛盾？是否存在逻辑跳跃、断裂或自相矛盾？
# 3.  **术语使用 (Terminology Usage)**：专业术语的使用是否准确、恰当、完整？是否存在术语误用、滥用、缺失或概念性错误？
# 4.  **事实正确性 (Factual Correctness)**：回答中的技术细节、参数、原理、行业现状等是否符合已知事实和行业共识？是否存在事实性错误或过时信息？

# **总体质量评分标准：**
# *   `low`：**存在严重缺陷**（如明显事实错误、完全偏离主题、逻辑混乱、关键术语错误），**不适合**用于SFT。
# *   `medium`：**存在轻微问题或可优化项**（如部分表述不清、个别术语不严谨、次要逻辑不完美、相关性略有不足），需修改后方可考虑使用。
# *   `high`：**无明显错误**，内容**准确、专业、逻辑清晰、紧扣主题**，**适合**直接用于SFT。

# **你的任务：**
# 1.  对每个维度进行独立评分 (`high`/`medium`/`low`)。
# 2.  给出基于四个维度的**总体质量评分** (`high`/`medium`/`low`)。
# 3.  对于评分非`high`的维度，**必须具体指出**存在的问题及其**类型**（例如："术语误用：将'OLED'错误称为'LED'"；"事实错误：声称当前主流Mini-LED背光分区数普遍超过5000区"）。
# 4.  基于你的专业知识和评估结果，**提供具体、可操作的改进建议**，以提升该问答对的质量。

# **输出格式要求（严格遵循JSON）：**
# {
#     "quality_rating": {
#         "overall": "high/medium/low", // 总体质量评分
#         "detailed_scores": {
#             "Relevance": {"score": "high/medium/low", "issues": ["具体问题描述1", "具体问题描述2", ...]}, // 如无问题，issues为空数组[]
#             "Logical Consistency": {"score": "high/medium/low", "issues": [...]},
#             "Terminology Usage": {"score": "high/medium/low", "issues": [...]},
#             "Factual Correctness": {"score": "high/medium/low", "issues": [...]}
#         }
#     },
#     "improvement_suggestions": ["具体建议1", "具体建议2", ...] // 即使总体是high，也可提供优化建议
# }

# ### 待评估样本
# 问题: {question_text}
# 答案: {answer_text}
# """
# }

# # 发送单个请求（移除思维链参数）
# def evaluate_qa_quality(question, answer):
#     try:
#         prompt = quality_evaluation_prompt.copy()
#         prompt["content"] = prompt["content"].replace("{question_text}", question) \
#             .replace("{answer_text}", answer)

#         response = client.chat.completions.create(
#             model="ep-20250813144949-kchv2",  # 使用您提供的模型端点ID
#             messages=[prompt],
#             temperature=0.1
#         )
#         # 打印模型的响应
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"请求失败: {str(e)}"

# # ================ 多进程工作函数 ================
# def process_row(args):
#     """
#     处理单个数据行的质量评估任务（移除思维链）
#     """
#     row_idx, row_data, server_config = args
#     question, answer = row_data['问题'], row_data['答案']

#     # 1. 初始化客户端（每个进程单独创建）
#     client = Ark(
#         api_key=server_config["api_key"],
#         base_url=server_config["base_url"]
#     )

#     try:
#         # 2. 构建评估提示
#         prompt = quality_evaluation_prompt.copy()
#         prompt["content"] = prompt["content"].replace("{question_text}", question) \
#             .replace("{answer_text}", answer)

#         # 3. 发送请求
#         response = client.chat.completions.create(
#             model=server_config["model_name"],
#             messages=[prompt],
#             temperature=0.1
#         )

#         # 4. 解析响应
#         result_text = response.choices[0].message.content
#         logger.info(f"行 {row_idx} 收到响应，长度: {len(result_text)}")
        
#         try:
#             # 更健壮的JSON提取方法
#             if '```json' in result_text:
#                 # 提取```json和```之间的内容
#                 json_start = result_text.find('```json') + 7
#                 json_end = result_text.find('```', json_start)
#                 if json_end == -1:
#                     json_end = len(result_text)
#                 json_text = result_text[json_start:json_end].strip()
#             elif '{' in result_text and '}' in result_text:
#                 # 提取第一个完整的JSON对象
#                 start = result_text.find('{')
#                 brace_count = 0
#                 end = start
#                 for i, char in enumerate(result_text[start:], start):
#                     if char == '{':
#                         brace_count += 1
#                     elif char == '}':
#                         brace_count -= 1
#                         if brace_count == 0:
#                             end = i + 1
#                             break
#                 json_text = result_text[start:end]
#             else:
#                 json_text = result_text.strip()
            
#             # 清理可能的多余内容
#             json_text = json_text.strip()
#             if json_text.endswith('```'):
#                 json_text = json_text[:-3].strip()
                
#             logger.debug(f"行 {row_idx} 提取的JSON: {json_text[:200]}...")
            
#             result = json.loads(json_text)
#             logger.info(f"行 {row_idx} JSON解析成功")
            
#         except json.JSONDecodeError as e:
#             logger.error(f"行 {row_idx} JSON解析失败: {e}")
#             logger.error(f"提取的JSON文本: {json_text[:500]}...")
            
#             # 尝试修复常见的JSON问题
#             try:
#                 # 移除可能的注释
#                 import re
#                 cleaned_json = re.sub(r'//.*?\n', '', json_text)
#                 cleaned_json = re.sub(r'/\*.*?\*/', '', cleaned_json, flags=re.DOTALL)
#                 result = json.loads(cleaned_json)
#                 logger.info(f"行 {row_idx} JSON修复后解析成功")
#             except:
#                 logger.error(f"行 {row_idx} JSON修复失败，使用默认值")
#                 # 使用默认值
#                 result = {
#                     "quality_rating": {
#                         "overall": "ERROR",
#                         "detailed_scores": {"error": f"JSON解析失败: {str(e)}"}
#                     }
#                 }
#         except Exception as e:
#             logger.error(f"行 {row_idx} 处理异常: {e}")
#             result = {
#                 "quality_rating": {
#                     "overall": "ERROR", 
#                     "detailed_scores": {"error": f"处理异常: {str(e)}"}
#                 }
#             }
        
#         # 5. 更新行数据
#         row_data['quality_rating'] = result['quality_rating']['overall']
#         row_data['detailed_scores'] = str(result['quality_rating']['detailed_scores'])
#         logger.info(f"行 {row_idx} 处理完成，评分: {row_data['quality_rating']}")
#         return (row_idx, row_data)

#     except Exception as e:
#         logger.error(f"处理行 {row_idx} 时出错: {str(e)}")
#         # 保留原始行数据并标记错误
#         row_data['quality_rating'] = "ERROR"
#         row_data['detailed_scores'] = str(e)
#         return (row_idx, row_data)

# if __name__ == '__main__':
#     # 0. 服务器配置
#     server_config = {
#         "base_url": "https://ark.cn-beijing.volces.com/api/v3",
#         "api_key": "5a032496-1268-4e6f-b6ee-a9affc6b5469",
#         "model_name": "ep-20250813144949-kchv2"
#     }

#     # 1. 加载源数据
#     file_name = "/mnt/workspace/LLM/ldd/sft/result/Qwen325结果.xlsx"
#     data = pd.read_excel(file_name)
#     logger.info(f"加载数据完成，共 {data.shape[0]} 行")

#     # 检查数据切片 - 处理所有数据
#     original_data = data.copy()
#     data = data.copy()  # 处理全部数据
#     logger.info(f"准备处理数据: {data.shape[0]} 行")
    
#     # 检查必要列是否存在（移除思维链检查）
#     required_columns = ['问题', '答案']
#     missing_columns = [col for col in required_columns if col not in data.columns]
#     if missing_columns:
#         logger.error(f"缺少必要列: {missing_columns}")
#         logger.info(f"数据列名: {list(data.columns)}")
#         exit(1)

#     # 2. 准备多进程处理
#     num_workers = min(5, data.shape[0])  # 减少进程数，避免过多空闲进程
#     logger.info(f"启用 {num_workers} 个工作进程")

#     # 先测试单个请求是否正常（移除思维链参数）
#     if data.shape[0] > 0:
#         test_row = data.iloc[0]
#         logger.info("测试单个请求...")
#         try:
#             test_result = evaluate_qa_quality(
#                 test_row['问题'], 
#                 test_row['答案']
#             )
#             logger.info(f"测试请求成功: {test_result[:100]}...")
#         except Exception as e:
#             logger.error(f"测试请求失败: {str(e)}")
#             exit(1)

#     # 3. 创建进程池
#     try:
#         with multiprocessing.Pool(processes=num_workers) as pool:
#             # 准备任务参数 [(行索引, 行数据, 服务器配置), ...]
#             task_args = [(idx, row.to_dict(), server_config) for idx, row in data.iterrows()]
#             logger.info(f"准备处理 {len(task_args)} 个任务")

#             # 4. 并行处理并收集结果
#             results = []
#             completed_count = 0
#             for result in tqdm(pool.imap(process_row, task_args), total=len(task_args), desc="处理进度"):
#                 results.append(result)
#                 completed_count += 1
#                 if completed_count % 1 == 0:  # 每完成1个就打印一次
#                     logger.info(f"已完成 {completed_count}/{len(task_args)} 个任务")

#         logger.info(f"所有任务完成，共收集到 {len(results)} 个结果")
        
#     except Exception as e:
#         logger.error(f"多进程处理出错: {str(e)}")
#         exit(1)

#     # 5. 按原始顺序更新数据
#     logger.info("开始更新结果数据")
#     updated_count = 0
#     for row_idx, updated_row in results:
#         if row_idx in data.index:
#             data.loc[row_idx,'quality_rating'] = updated_row['quality_rating']
#             data.loc[row_idx, 'detailed_scores'] = updated_row['detailed_scores']
#             updated_count += 1
#         else:
#             logger.warning(f"索引 {row_idx} 不在数据中")
    
#     logger.info(f"成功更新 {updated_count} 行数据")

#     # 6. 保存结果
#     save_dir = './result'
#     save_name = f'增强前的评估结果_火山版_无思维链_{time_label}.xlsx'
#     save_path = os.path.join(save_dir, save_name)

#     os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

#     # 统计质量评分分布
#     if 'quality_rating' in data.columns:
#         quality_counts = data['quality_rating'].value_counts()
#         logger.info("=" * 50)
#         logger.info("质量评分统计结果:")
#         logger.info("=" * 50)
        
#         high_count = quality_counts.get('high', 0)
#         medium_count = quality_counts.get('medium', 0)
#         low_count = quality_counts.get('low', 0)
#         error_count = quality_counts.get('ERROR', 0)
        
#         total_evaluated = high_count + medium_count + low_count
        
#         logger.info(f"📊 HIGH (高质量):   {high_count:4d} 条 ({high_count/len(data)*100:.1f}%)")
#         logger.info(f"📊 MEDIUM (中等质量): {medium_count:4d} 条 ({medium_count/len(data)*100:.1f}%)")
#         logger.info(f"📊 LOW (低质量):    {low_count:4d} 条 ({low_count/len(data)*100:.1f}%)")
#         if error_count > 0:
#             logger.info(f"❌ ERROR (处理失败): {error_count:4d} 条 ({error_count/len(data)*100:.1f}%)")
#         logger.info(f"✅ 总计已评估:      {total_evaluated:4d} 条")
#         logger.info("=" * 50)
        
#         # 保存统计摘要（修复JSON序列化问题）
#         summary_stats = {
#             'total_samples': int(len(data)),
#             'high_quality': int(high_count),
#             'medium_quality': int(medium_count),
#             'low_quality': int(low_count),
#             'error_count': int(error_count),
#             'high_percentage': float(round(high_count/len(data)*100, 2)),
#             'medium_percentage': float(round(medium_count/len(data)*100, 2)),
#             'low_percentage': float(round(low_count/len(data)*100, 2))
#         }
        
#         # 保存统计结果到JSON
#         stats_path = os.path.join(save_dir, f'质量评估统计_火山版_无思维链_{time_label}.json')
#         with open(stats_path, 'w', encoding='utf-8') as f:
#             json.dump(summary_stats, f, ensure_ascii=False, indent=2)
#         logger.info(f"统计结果已保存到: {stats_path}")
#     else:
#         logger.warning("未找到quality_rating列，无法统计")

#     # 检查数据是否为空
#     if data.empty:
#         logger.error("数据为空，无法保存")
#         exit(1)
        
#     try:
#         with pd.ExcelWriter(save_path) as writer:
#             data.to_excel(writer, index=False)
#         logger.info(f"结果已保存到: {save_path}")
#         logger.info(f"保存的数据形状: {data.shape}")
#     except Exception as e:
#         logger.error(f"保存文件失败: {str(e)}")
#         # 尝试保存为CSV作为备选
#         csv_path = save_path.replace('.xlsx', '.csv')
#         data.to_csv(csv_path, index=False)
#         logger.info(f"已保存为CSV格式: {csv_path}")