import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import argparse
from ..TextGeneration.prompts_conf import get_prompt


class DataLabeler:
    """数据标注器，用于质量检查和数据标注"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        初始化数据标注器
        
        Args:
            model_config: 模型配置，包含API信息
        """
        self.model_config = model_config
        self.api_key = model_config.get('api_key')
        self.api_base = model_config.get('api_base', 'https://ark.cn-beijing.volces.com/api/v3')
        self.model_name = model_config.get('model_name', 'ep-20250717164726-xgxnh')
        self.temperature = model_config.get('temperature', 0.1)  # 质量检查使用较低的温度
        self.max_tokens = model_config.get('max_tokens', 2048)
        
    async def check_reasoning_validity(self, question: str, reasoning: str, 
                                     answer: str = "", image_path: str = "") -> Dict[str, Any]:
        """
        检查推理过程的有效性
        
        Args:
            question: 问题
            reasoning: 推理过程
            answer: 答案
            image_path: 图片路径（可选）
            
        Returns:
            质量检查结果
        """
        try:
            # 使用prompt 36进行推理有效性检查
            prompt_template = get_prompt(36)
            if not prompt_template:
                prompt_template = """
问题：{question}
推理过程：{reasoning}

请你判断推理过程是否可以从题目中推导而来，
如果推理过程中的所有论据都能在题目或者科学公理中找到证据，则返回 1
如果推理过程中的任一论据无法找到依据，则返回-1。
"""
            
            prompt = prompt_template.format(question=question, reasoning=reasoning)
            
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析结果
            if "1" in result:
                return {"validity": 1, "reason": "推理过程有效"}
            else:
                return {"validity": -1, "reason": "推理过程存在问题"}
                
        except Exception as e:
            print(f"Error checking reasoning validity: {e}")
            return {"validity": 0, "reason": f"检查失败: {str(e)}"}
    
    async def check_question_clarity(self, question: str, image_path: str = "") -> Dict[str, Any]:
        """
        检查问题的清晰度和完整性
        
        Args:
            question: 问题
            image_path: 图片路径（可选）
            
        Returns:
            质量检查结果
        """
        try:
            # 使用prompt 37进行问题清晰度检查
            prompt_template = get_prompt(37)
            if not prompt_template:
                prompt_template = """
问题：{question}

判断问题是否缺少必要的细节和信息以回答问题
如果题目提供了足够的信息则返回 1
否则返回 -1
"""
            
            prompt = prompt_template.format(question=question)
            
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            
            if "1" in result:
                return {"clarity": 1, "reason": "问题清晰完整"}
            else:
                return {"clarity": -1, "reason": "问题缺少必要信息"}
                
        except Exception as e:
            print(f"Error checking question clarity: {e}")
            return {"clarity": 0, "reason": f"检查失败: {str(e)}"}
    
    async def check_answer_correctness(self, question: str, reasoning: str, 
                                     answer: str, image_path: str = "") -> Dict[str, Any]:
        """
        检查答案的正确性
        
        Args:
            question: 问题
            reasoning: 推理过程
            answer: 答案
            image_path: 图片路径（可选）
            
        Returns:
            质量检查结果
        """
        try:
            # 使用prompt 39进行答案正确性检查
            prompt_template = get_prompt(39)
            if not prompt_template:
                prompt_template = """
问题：{question}
推理过程: {reasoning}
答案: {answer}

以上是一个问题,答案和推理过程
判断答案和推理过程是否正确
如果答案和推理过程都正确则返回 1
如果答案和推理过程存在任一一个错误则返回 -1
"""
            
            prompt = prompt_template.format(question=question, reasoning=reasoning, answer=answer)
            
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            
            if "1" in result:
                return {"correctness": 1, "reason": "答案和推理正确"}
            else:
                return {"correctness": -1, "reason": "答案或推理存在错误"}
                
        except Exception as e:
            print(f"Error checking answer correctness: {e}")
            return {"correctness": 0, "reason": f"检查失败: {str(e)}"}
    
    async def check_question_difficulty(self, question: str, reasoning: str, 
                                      answer: str, image_path: str = "") -> Dict[str, Any]:
        """
        检查问题的难度是否合适
        
        Args:
            question: 问题
            reasoning: 推理过程
            answer: 答案
            image_path: 图片路径（可选）
            
        Returns:
            质量检查结果
        """
        try:
            # 使用prompt 40进行难度检查
            prompt_template = get_prompt(40)
            if not prompt_template:
                prompt_template = """
问题：{question}
推理过程: {reasoning}
答案: {answer}

以上是一个问题,答案和推理过程
判断该问题是否过于简单不需要复杂推理过程
如果问题过于简单不需要复杂推理过程则返回 -1
如果问题需要推理过程则返回 1
"""
            
            prompt = prompt_template.format(question=question, reasoning=reasoning, answer=answer)
            
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            
            if "1" in result:
                return {"difficulty": 1, "reason": "问题难度适中"}
            else:
                return {"difficulty": -1, "reason": "问题过于简单"}
                
        except Exception as e:
            print(f"Error checking question difficulty: {e}")
            return {"difficulty": 0, "reason": f"检查失败: {str(e)}"}
    
    async def comprehensive_quality_check(self, qa_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合质量检查
        
        Args:
            qa_record: QA记录，包含question, answer, reasoning等字段
            
        Returns:
            综合质量检查结果
        """
        question = qa_record.get('question', '')
        reasoning = qa_record.get('reasoning', '')
        answer = qa_record.get('answer', '')
        image_path = qa_record.get('image', '')
        
        # 并行执行各项检查
        checks = await asyncio.gather(
            self.check_reasoning_validity(question, reasoning, answer, image_path),
            self.check_question_clarity(question, image_path),
            self.check_answer_correctness(question, reasoning, answer, image_path),
            self.check_question_difficulty(question, reasoning, answer, image_path),
            return_exceptions=True
        )
        
        # 汇总结果
        result = {
            'overall_score': 0,
            'checks': {
                'reasoning_validity': checks[0] if not isinstance(checks[0], Exception) else {"validity": 0, "reason": "检查失败"},
                'question_clarity': checks[1] if not isinstance(checks[1], Exception) else {"clarity": 0, "reason": "检查失败"},
                'answer_correctness': checks[2] if not isinstance(checks[2], Exception) else {"correctness": 0, "reason": "检查失败"},
                'question_difficulty': checks[3] if not isinstance(checks[3], Exception) else {"difficulty": 0, "reason": "检查失败"}
            },
            'passed': False,
            'suggestions': []
        }
        
        # 计算总分
        scores = []
        for check_name, check_result in result['checks'].items():
            if isinstance(check_result, dict):
                # 获取具体的分数字段
                score_field = list(check_result.keys())[0]  # validity, clarity, correctness, difficulty
                if score_field in check_result:
                    scores.append(check_result[score_field])
        
        if scores:
            result['overall_score'] = sum(scores) / len(scores)
            result['passed'] = result['overall_score'] > 0
        
        # 生成建议
        for check_name, check_result in result['checks'].items():
            if isinstance(check_result, dict):
                score_field = list(check_result.keys())[0]
                if check_result.get(score_field, 0) <= 0:
                    result['suggestions'].append(f"{check_name}: {check_result.get('reason', '需要改进')}")
        
        return result


async def quality_check_data(data: List[Dict[str, Any]], model_config: Dict[str, Any],
                           concurrency: int = 5) -> List[Dict[str, Any]]:
    """
    对数据进行批量质量检查
    
    Args:
        data: 待检查的数据列表
        model_config: 模型配置
        concurrency: 并发数
        
    Returns:
        质量检查结果列表
    """
    labeler = DataLabeler(model_config)
    semaphore = asyncio.Semaphore(concurrency)
    
    async def check_with_semaphore(record):
        async with semaphore:
            return await labeler.comprehensive_quality_check(record)
    
    tasks = [check_with_semaphore(record) for record in data]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    quality_results = []
    for i, result in enumerate(results):
        if isinstance(result, dict):
            result['original_record'] = data[i]
            quality_results.append(result)
        else:
            quality_results.append({
                'overall_score': 0,
                'passed': False,
                'error': str(result),
                'original_record': data[i]
            })
    
    return quality_results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="对QA数据进行质量检查和标注")
    parser.add_argument("--data-path", required=True,
                        help="输入数据文件路径（JSON格式）")
    parser.add_argument("--output-path", required=True,
                        help="质量检查结果输出路径")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="并发请求数，默认 5")
    parser.add_argument("--api-url", default="https://ark.cn-beijing.volces.com/api/v3",
                        help="API 接口地址")
    parser.add_argument("--model-name", default="ep-20250717164726-xgxnh",
                        help="使用模型名称")
    parser.add_argument("--api-key", required=True,
                        help="API Key")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="生成的随机性温度，默认 0.1")

    args = parser.parse_args()

    # 准备模型配置
    model_config = {
        'api_key': args.api_key,
        'api_base': args.api_url,
        'model_name': args.model_name,
        'temperature': args.temperature,
        'max_tokens': 2048
    }

    # 加载数据
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")

    # 执行质量检查
    print(f"Starting quality check with concurrency {args.concurrency}...")
    results = asyncio.run(quality_check_data(data, model_config, args.concurrency))

    # 统计结果
    passed_count = sum(1 for r in results if r.get('passed', False))
    total_count = len(results)
    pass_rate = passed_count / total_count if total_count > 0 else 0

    print(f"Quality check completed!")
    print(f"Total records: {total_count}")
    print(f"Passed records: {passed_count}")
    print(f"Pass rate: {pass_rate:.2%}")

    # 保存结果
    output_data = {
        'summary': {
            'total_records': total_count,
            'passed_records': passed_count,
            'pass_rate': pass_rate
        },
        'results': results
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()