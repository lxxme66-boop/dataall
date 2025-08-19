import os
import json
import pickle
import pandas as pd
import concurrent.futures
import argparse
import asyncio
import base64
from typing import List, Dict, Any, Optional, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TextGeneration.prompts_conf import build_prompt, get_prompt


class DataRewriter:
    """数据改写器，用于生成高质量的改写版本"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        初始化数据改写器
        
        Args:
            model_config: 模型配置，包含API信息
        """
        self.model_config = model_config
        self.api_key = model_config.get('api_key')
        self.api_base = model_config.get('api_base', 'https://ark.cn-beijing.volces.com/api/v3')
        self.model_name = model_config.get('model_name', 'ep-20250717164726-xgxnh')
        self.temperature = model_config.get('temperature', 0.7)
        self.max_tokens = model_config.get('max_tokens', 8192)
        
    def encode_image(self, img_path: str) -> str:
        """编码图片为base64格式"""
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            img_type = img_path.split(".")[-1].lower()
            return f"data:image/{img_type};base64,{img_base64}"
        except Exception as e:
            print(f"Error encoding image {img_path}: {e}")
            return None
    
    async def generate_rewritten_qa(self, record: Dict[str, Any], 
                                   prompt_template: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        生成改写的QA对
        
        Args:
            record: 原始记录，包含question, answer等字段
            prompt_template: 自定义prompt模板
            
        Returns:
            改写后的记录
        """
        try:
            # 构建改写prompt
            if prompt_template is None:
                prompt_template = build_prompt(record.get("question", ""), record.get("answer", ""))
            
            # 准备消息内容
            messages = []
            
            # 如果包含图片，添加图片内容
            if "image" in record and record["image"]:
                img_encoded = self.encode_image(record["image"])
                if img_encoded:
                    messages.append({
                        "role": "user", 
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_encoded}},
                            {"type": "text", "text": prompt_template}
                        ]
                    })
                else:
                    # 图片编码失败，只使用文本
                    messages.append({"role": "user", "content": prompt_template})
            else:
                messages.append({"role": "user", "content": prompt_template})
            
            # 调用模型API
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content
            
            # 尝试解析JSON响应
            try:
                parsed_result = json.loads(result)
                # 添加原始信息
                parsed_result['original_question'] = record.get("question", "")
                parsed_result['original_answer'] = record.get("answer", "")
                parsed_result['source_file'] = record.get("source_file", "")
                if "image" in record:
                    parsed_result['image'] = record["image"]
                return parsed_result
            except json.JSONDecodeError:
                # 如果不是JSON格式，返回原始文本
                return {
                    'rewritten_content': result,
                    'original_question': record.get("question", ""),
                    'original_answer': record.get("answer", ""),
                    'source_file': record.get("source_file", ""),
                    'image': record.get("image", "")
                }
                
        except Exception as e:
            print(f"Error generating rewritten QA for record: {e}")
            return None
    
    async def batch_rewrite(self, data: List[Dict[str, Any]], 
                           concurrency: int = 5,
                           prompt_template: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量改写数据
        
        Args:
            data: 待改写的数据列表
            concurrency: 并发数
            prompt_template: 自定义prompt模板
            
        Returns:
            改写后的数据列表
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def rewrite_with_semaphore(record):
            async with semaphore:
                return await self.generate_rewritten_qa(record, prompt_template)
        
        tasks = [rewrite_with_semaphore(record) for record in data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉失败的结果
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            elif isinstance(result, Exception):
                print(f"Task failed with exception: {result}")
        
        return valid_results


def load_data(path: str) -> List[Dict[str, Any]]:
    """加载数据文件"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def save_data(data: List[Dict[str, Any]], path: str):
    """保存数据到文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def generate_rewritten_data(input_path: str, output_path: str, 
                                 model_config: Dict[str, Any],
                                 concurrency: int = 5,
                                 prompt_template: Optional[str] = None):
    """
    生成改写数据的主函数
    
    Args:
        input_path: 输入数据路径
        output_path: 输出数据路径
        model_config: 模型配置
        concurrency: 并发数
        prompt_template: 自定义prompt模板
    """
    # 加载数据
    print(f"Loading data from {input_path}...")
    data = load_data(input_path)
    print(f"Loaded {len(data)} records")
    
    # 初始化改写器
    rewriter = DataRewriter(model_config)
    
    # 执行改写
    print(f"Starting rewriting with concurrency {concurrency}...")
    results = await rewriter.batch_rewrite(data, concurrency, prompt_template)
    
    # 保存结果
    print(f"Saving {len(results)} rewritten records to {output_path}...")
    save_data(results, output_path)
    print("Rewriting completed!")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="使用大模型进行数据改写和增强")
    parser.add_argument("--data-path", required=True,
                        help="输入数据文件路径（支持 .csv/.json/.pkl/.pickle）")
    parser.add_argument("--output-path", required=True,
                        help="结果输出 JSON 文件路径")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="并发请求数，默认 5")
    parser.add_argument("--api-url", default="https://ark.cn-beijing.volces.com/api/v3",
                        help="API 接口地址")
    parser.add_argument("--model-name", default="ep-20250717164726-xgxnh",
                        help="使用模型名称")
    parser.add_argument("--api-key", required=True,
                        help="API Key")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成的随机性温度，默认 0.7")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="最大生成token数")
    parser.add_argument("--prompt-template", type=str, default=None,
                        help="自定义prompt模板文件路径")

    args = parser.parse_args()

    # 准备模型配置
    model_config = {
        'api_key': args.api_key,
        'api_base': args.api_url,
        'model_name': args.model_name,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens
    }

    # 加载自定义prompt模板
    prompt_template = None
    if args.prompt_template and os.path.exists(args.prompt_template):
        with open(args.prompt_template, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

    # 运行改写任务
    asyncio.run(generate_rewritten_data(
        args.data_path,
        args.output_path,
        model_config,
        args.concurrency,
        prompt_template
    ))


if __name__ == "__main__":
    main()