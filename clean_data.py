import pickle as pkl
import random
import argparse
import os
import re

pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.DOTALL)


def clean_process(input_file,output_file,copy_parsed_pdf=False,):
    with open(input_file, "rb") as f:
        data = pkl.load(f)
    
    output_json = []
    import json
    for i in range(len(data)):
        try:
            # Check if this is text-only data or multimodal data
            if 'image_path' in data[i]:
                # Multimodal data processing
                content = data[i]['content']
                matches = pattern.findall(content)
                if matches:
                    json_dict_str = matches[-1]
                    response = eval(json_dict_str)
                else:
                    continue
                    
                img_path = data[i]['image_path']
                img_path = img_path.split("/")[-3:]
                img_path = "/".join(img_path)
                img_path = "./" + img_path
                response['image_path'] = img_path
            else:
                # Text-only data processing
                response = {
                    'content': data[i].get('content', ''),
                    'source_file': data[i].get('source_file', ''),
                    'text_content': data[i].get('text_content', ''),
                    'qa_pairs': data[i].get('qa_pairs', [])
                }
                
                # If content contains QA pairs in specific format, extract them
                if 'qa_pairs' not in data[i] and 'content' in data[i]:
                    content = data[i]['content']
                    # Try to extract QA pairs from content if they exist
                    matches = pattern.findall(content)
                    if matches:
                        try:
                            json_dict_str = matches[-1]
                            extracted = eval(json_dict_str)
                            response.update(extracted)
                        except:
                            pass

            output_json.append(response)
            
            if copy_parsed_pdf and 'image_path' in data[i]:
                image_path = data[i]['image_path']
                # copy the folder of image_path to the new folder
                # only select the first 200
                folder_path = os.path.join(output_file, image_path.split("/")[-3])
                # original folder path
                original_path = image_path.split("/")
                original_path = "/".join(original_path[:-2])
                # copy the folder
                os.makedirs(folder_path, exist_ok=True)
                # copy original_path to folder_path
                os.system(f"cp -r {original_path} {folder_path}")
                
        except Exception as e:
            print(f"Error parsing content for index {i}: {e}")
            continue
      
    
    with open(os.path.join(output_file,"total_response.json"),  "w",
            encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
        print("The number of responses is {}".format(len(output_json)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--input_file", type=str, \
                        default="/mnt/workspace/MLLM/zc/tclreasoning/data/100TestOutputt/total_response.pkl",\
                         help="path to the input json file")

    parser.add_argument("--output_file", type=str, default="/mnt/workspace/MLLM/zc/tclreasoning/data/100TestOutputt", help="path to the output folder")
    parser.add_argument("--copy_parsed_pdf",type=bool, default=False,help="copy parsed_pdf from original directory to new folder")
    args = parser.parse_args()
    
    new_folder = args.output_file
    input_file = args.input_file
    copy_parsed_pdf = args.copy_parsed_pdf
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    clean_process(input_file, new_folder, copy_parsed_pdf=copy_parsed_pdf)
    print(f"Cleaned data saved to {new_folder}/total_response.json")

async def main(input_file, output_dir, copy_parsed_pdf=False):
    """
    Main async function for data cleaning
    
    Args:
        input_file: Path to input pickle file
        output_dir: Output directory for cleaned data
        copy_parsed_pdf: Whether to copy parsed PDF files
    
    Returns:
        Path to cleaned output file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    clean_process(input_file, output_dir, copy_parsed_pdf=copy_parsed_pdf)
    output_file = os.path.join(output_dir, "total_response.json")
    print(f"Cleaned data saved to {output_file}")
    
    return output_file