import os
from openai import OpenAI, AsyncOpenAI
from TextGeneration.prompts_conf import system_prompt, all_prompts
import asyncio
import json
import pandas as pd
import re

# API configuration
ark_url = "http://0.0.0.0:8080/v1"
api_key = "ae37bba4-73be-4c22-b1a7-c6b1f5ec3a4b"
model = "/mnt/storage/models/Skywork/Skywork-R1V3-38B"


async def get_response(input_prompt, api_key=api_key, qwen_url=ark_url, model=model, stream=False):
    """
    Get response from the model for text-only input.
    """
    # Check if we should use mock mode
    use_mock = os.environ.get('USE_MOCK_API', 'false').lower() == 'true'
    
    if use_mock:
        # Return mock QA data
        mock_response = {
            "qa_pairs": [
                {
                    "question": "What is the main topic discussed in this text?",
                    "answer": "The text discusses semiconductor technology and its applications in modern electronics.",
                    "question_type": "factual",
                    "difficulty": "intermediate",
                    "reasoning": "Based on the content analysis of the provided text."
                }
            ],
            "key_concepts": ["semiconductor", "electronics", "technology"],
            "technical_details": {
                "materials": ["silicon", "germanium"],
                "parameters": ["conductivity", "band gap"],
                "methods": ["doping", "lithography"]
            },
            "main_findings": ["Semiconductors are fundamental to modern electronics"]
        }
        
        return (json.dumps(mock_response, ensure_ascii=False), "Mock reasoning process")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=qwen_url if qwen_url != None else None,
    )
    
    reasoning_content = ""
    answer_content = ""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": input_prompt
        }
    ]
    
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        temperature=0.8,
        max_tokens=4096,
        top_p=0.9,
    )
    
    if stream:
        async for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content:
                        answer_content += delta.content
        
        return answer_content, reasoning_content
    else:
        content = completion.choices[0].message.content
        return content, ""


async def process_response(*args, **kwargs):
    """
    Process a single response to generate QA pairs.
    """
    return await get_response(*args, **kwargs)


async def get_total_responses(index, file_path, pool_size=10, stream=False):
    """
    Generate QA pairs from cleaned text data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_tasks = []
    
    # Question type distribution (adjusted for requirements)
    question_type_prompts = {
        'factual': 'text_qa_basic',      # 15%
        'comparison': 'text_qa_basic',   # 15%
        'reasoning': 'text_qa_advanced',    # 50%
        'open_ended': 'text_qa_advanced'    # 20%
    }
    
    # Target distribution
    target_distribution = {
        'factual': 0.15,
        'comparison': 0.15,
        'reasoning': 0.50,
        'open_ended': 0.20
    }
    
    for response_index, response in enumerate(data):
        # Extract relevant information
        source_file = response.get('source_file', '')
        
        # Get the text content or key concepts
        text_content = response.get('text_content', '')
        key_concepts = response.get('key_concepts', [])
        technical_details = response.get('technical_details', {})
        main_findings = response.get('main_findings', [])
        context = response.get('context', '')
        
        # Prepare context for QA generation
        qa_context = {
            'source_file': source_file,
            'key_concepts': key_concepts,
            'technical_details': technical_details,
            'main_findings': main_findings,
            'context': context,
            'text_content': text_content
        }
        
        # Determine which question types to generate based on distribution
        question_types_to_generate = select_question_types(response_index, len(data), target_distribution)
        
        # Create tasks for each question type
        for q_type in question_types_to_generate:
            prompt_index = question_type_prompts[q_type]
            input_prompt = all_prompts[prompt_index]
            
            # Format the prompt with the context
            final_input = input_prompt.format(
                text_content=text_content if text_content else json.dumps(qa_context, ensure_ascii=False)
            )
            
            task = asyncio.create_task(
                process_response(final_input, stream=stream)
            )
            total_tasks.append((task, source_file, q_type))
        
        print(f"Processed response {response_index + 1}/{len(data)}")
    
    # Execute tasks in batches
    results = []
    for i in range(0, len(total_tasks), pool_size):
        batch_tasks = total_tasks[i:i + pool_size]
        batch_results = await asyncio.gather(*[task for task, _, _ in batch_tasks])
        
        # Process results
        for j, result in enumerate(batch_results):
            task, source_file, q_type = batch_tasks[j]
            content, reasoning = result
            
            # Parse the response
            try:
                if "```json" in content:
                    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(1))
                    else:
                        parsed_result = json.loads(content)
                else:
                    parsed_result = json.loads(content)
                
                # Add source file and ensure question type is set
                parsed_result['source_file'] = source_file
                if 'qa_pairs' in parsed_result:
                    for qa in parsed_result['qa_pairs']:
                        if 'question_type' not in qa:
                            qa['question_type'] = q_type
                
                results.append(parsed_result)
                
            except Exception as e:
                print(f"Error parsing response: {e}")
                continue
    
    # Filter out None results
    results = [result for result in results if result is not None]
    return results


def select_question_types(index, total_count, target_distribution):
    """
    Select which question types to generate for this response based on target distribution.
    """
    # Simple round-robin with weighted selection
    types_to_generate = []
    
    # Calculate how many of each type we should have generated so far
    progress = (index + 1) / total_count
    
    for q_type, target_ratio in target_distribution.items():
        expected_count = progress * total_count * target_ratio
        # Add some randomness to avoid too regular patterns
        if index % (int(1 / target_ratio)) == 0 or (index + 1) == total_count:
            types_to_generate.append(q_type)
    
    # Ensure we generate at least one question type
    if not types_to_generate:
        # Weight by target distribution
        import random
        weights = list(target_distribution.values())
        types = list(target_distribution.keys())
        selected = random.choices(types, weights=weights, k=1)
        types_to_generate = selected
    
    return types_to_generate


async def check_data_quality(url, key, model, output_file, prompt_indexes, pool_size=10, check_times=5, pass_score=1, stream=False):
    """
    Check the quality of generated QA pairs.
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    final_output = []
    api_key = key
    qwen_url = url
    model = model
    
    for sample in data:
        # Remove unnecessary fields
        sample.pop("reasoning_content", None)
        
        tasks = []
        for _ in range(check_times):
            if type(prompt_indexes) == int:
                input_prompt = all_prompts[prompt_indexes]
            else:
                index = _ % len(prompt_indexes)
                input_prompt = all_prompts[prompt_indexes[index]]
            
            final_input = input_prompt.format(**sample)
            task = asyncio.create_task(
                get_response(final_input, api_key=api_key, qwen_url=qwen_url, model=model, stream=stream)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Extract scores
        scores = []
        for result in results:
            content = result[0].strip()
            try:
                score = int(re.search(r'\d+', content).group())
                scores.append(score)
            except:
                scores.append(0)
        
        mean_score = sum(scores) / len(scores)
        if mean_score >= pass_score:
            final_output.append(sample)
        else:
            print(f"Data quality check failed with mean score {mean_score}")
    
    # Save filtered results
    pd.DataFrame(final_output).to_csv(
        output_file.replace(".json", "_filtered.csv"), 
        index=False
    )
    
    print(f"Quality check complete. {len(final_output)}/{len(data)} passed.")
    print(f"Results saved to {output_file.replace('.json', '_filtered.csv')}")
    return True