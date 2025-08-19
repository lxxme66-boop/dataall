import pickle as pkl
import random
import argparse
import os
import re
import json

pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.DOTALL)


def clean_data(input_file, output_dir):
    """
    Main entry point for data cleaning process.
    This function is called by the main pipeline.
    """
    return clean_process(input_file, output_dir)


def clean_process(input_file, output_file):
    with open(input_file, "rb") as f:

        data = pkl.load(f)
    
    output_json = []
    
    for i in range(len(data)):
        try:
            content = data[i]['content']
            matches = pattern.findall(content)
            
            if matches:
                json_dict_str = matches[-1]
                # Try to parse as JSON first
                try:
                    response = json.loads(json_dict_str)
                except:
                    # If JSON parsing fails, try eval (careful with security)
                    response = eval(json_dict_str)
            else:
                # Try to extract JSON without markdown code blocks
                try:
                    response = json.loads(content)
                except:
                    print(f"No valid JSON found in response {i}, skipping")
                    continue
                    
        except Exception as e:
            print(f"Error parsing content for index {i}: {e}")
            continue
            
        # Add source file information
        source_file = data[i].get('source_file', 'unknown')
        response['source_file'] = source_file
        
        # Add chunk information if available
        if 'chunk_index' in data[i]:
            response['chunk_index'] = data[i]['chunk_index']
            response['total_chunks'] = data[i].get('total_chunks', 1)
        
        # Add the original text content if needed
        if 'text_content' in data[i]:
            response['text_content'] = data[i]['text_content']
            
        output_json.append(response)
    
    # Merge responses from the same file if they were chunked
    merged_responses = merge_chunked_responses(output_json)
    
    # Save the cleaned data
    os.makedirs(output_file, exist_ok=True)
    with open(os.path.join(output_file, "total_response.json"), "w", encoding='utf-8') as f:
        json.dump(merged_responses, f, ensure_ascii=False, indent=4)
        print(f"The number of responses is {len(merged_responses)}")


def merge_chunked_responses(responses):
    """
    Merge responses that came from chunks of the same file.
    """
    # Group by source file
    file_groups = {}
    single_responses = []
    
    for resp in responses:
        if 'chunk_index' in resp and resp.get('total_chunks', 1) > 1:
            source = resp['source_file']
            if source not in file_groups:
                file_groups[source] = []
            file_groups[source].append(resp)
        else:
            single_responses.append(resp)
    
    # Merge chunked responses
    merged = []
    for source_file, chunks in file_groups.items():
        # Sort by chunk index
        chunks.sort(key=lambda x: x.get('chunk_index', 0))
        
        # Merge the responses
        merged_response = {
            'source_file': source_file,
            'qa_pairs': [],
            'key_concepts': [],
            'technical_details': {
                'materials': [],
                'parameters': [],
                'methods': []
            },
            'main_findings': []
        }
        
        for chunk in chunks:
            # Merge QA pairs
            if 'qa_pairs' in chunk:
                merged_response['qa_pairs'].extend(chunk['qa_pairs'])
            
            # Merge key concepts
            if 'key_concepts' in chunk:
                merged_response['key_concepts'].extend(chunk['key_concepts'])
            
            # Merge technical details
            if 'technical_details' in chunk:
                for key in ['materials', 'parameters', 'methods']:
                    if key in chunk['technical_details']:
                        merged_response['technical_details'][key].extend(chunk['technical_details'][key])
            
            # Merge main findings
            if 'main_findings' in chunk:
                merged_response['main_findings'].extend(chunk['main_findings'])
        
        # Remove duplicates
        merged_response['key_concepts'] = list(set(merged_response['key_concepts']))
        for key in ['materials', 'parameters', 'methods']:
            merged_response['technical_details'][key] = list(set(merged_response['technical_details'][key]))
        
        merged.append(merged_response)
    
    # Combine with single responses
    merged.extend(single_responses)
    
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean text processing results.")
    parser.add_argument("--input_file", type=str, 
                        default="/workspace/text_qa_generation/data/output/total_response.pkl",
                        help="path to the input pickle file")
    parser.add_argument("--output_file", type=str, 
                        default="/workspace/text_qa_generation/data/output", 
                        help="path to the output folder")
    args = parser.parse_args()
    
    input_file = args.input_file
    output_folder = args.output_file
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    clean_process(input_file, output_folder)
    print(f"Cleaned data saved to {output_folder}/total_response.json")