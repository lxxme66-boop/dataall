# import json
# import pandas as pd
# import re
# from tqdm import tqdm

# # def extract_think_content(context):
# #     """从context中提取<think>标签内的内容"""
# #     match = re.search(r'<think>(.*?)</think>', context, re.DOTALL)
# #     return match.group(1).strip() if match else context

# def process_json_to_excel(input_path, output_path):
#     # 读取JSON数据
#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     results = []
    
#     for item in tqdm(data, desc="Processing items"):
#         # 提取所需字段
#         result = {
#             "问题": item["question"],
#            # "问题类型": item["question_type"],
#            "思维链": item["source_info"]["text_content"],
#             "答案": item["answer"]
#            # "使用CoT": item["use_cot"],
#            "来源文件": item["source_info"]["text_content"]
#         }
#         results.append(result)
    
#     # 创建DataFrame并保存Excel
#     df = pd.DataFrame(results)
#     df.to_excel(output_path, index=False)
#     print(f"成功生成Excel文件: {output_path}, 共处理 {len(df)} 条记录")

# # 使用示例
# if __name__ == "__main__":
#     input_json = "/mnt/workspace/LLM/ldd/sft/data/output/qa_results/qa_generated.json"  # 替换为实际JSON文件路径
#     output_excel = "./input/Qwen325评估数据v3版带话题输入-无增强.xlsx"      # 输出Excel文件名
#     process_json_to_excel(input_json, output_excel)

import json
import pandas as pd
import re
from tqdm import tqdm

def clean_text_for_excel(text):
    """清理文本中Excel不支持的字符"""
    if not isinstance(text, str):
        return text
    
    # 方法1：使用正则表达式移除控制字符（推荐）
    # 移除0x00-0x08, 0x0B-0x0C, 0x0E-0x1F范围的控制字符
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # 方法2：如果还有问题，可以尝试更严格的清理
    # 只保留常见的可打印字符、中文字符和基本标点
    # text = re.sub(r'[^\x20-\x7E\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\t\n\r]', '', text)
    
    # 限制单元格内容长度（Excel单元格有32767字符限制）
    if len(text) > 32000:
        text = text[:32000] + "..."
    
    return text

def process_json_to_excel(input_path, output_path):
    # 读取JSON数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in tqdm(data, desc="Processing items"):
        # 提取所需字段并清理文本
        result = {
            "问题": clean_text_for_excel(item["question"]),
            #话题
            "思维链": clean_text_for_excel(item["source_info"]["content"]),
            #纯文本
            #"思维链": clean_text_for_excel(item["source_info"]["text_content"]),
            "答案": clean_text_for_excel(item["answer"]),
            "来源文件": item["paper_name"]
        }
        results.append(result)
    
    # 创建DataFrame并保存Excel
    df = pd.DataFrame(results)
    
    # 额外的安全措施：在写入Excel前再次清理DataFrame中的所有字符串列
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: clean_text_for_excel(x) if isinstance(x, str) else x)
    
    try:
        df.to_excel(output_path, index=False)
        print(f"成功生成Excel文件: {output_path}, 共处理 {len(df)} 条记录")
    except Exception as e:
        print(f"写入Excel时出错: {e}")
        # 如果Excel写入失败，尝试保存为CSV
        csv_path = output_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存为CSV文件: {csv_path}")

# 使用示例
if __name__ == "__main__":
    input_json = "/mnt/workspace/LLM/ldd/sft/data/output/qa_results/qa_generated.json"
    output_excel = "./input/Qwen325评估数据v2版输入-无增强.xlsx"
    process_json_to_excel(input_json, output_excel)