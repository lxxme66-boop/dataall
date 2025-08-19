import asyncio
import time
import pickle as pkl
import argparse
import json
import random
import os

# Import only what we need from Datageneration
try:
    from TextGeneration.Datageneration import process_folder_async, process_folder_async_with_history, parse_txt, input_text_process
except ImportError as e:
    print(f"Warning: Could not import from TextGeneration.Datageneration: {e}")
    # Try alternative import
    try:
        from TextGeneration.Datageneration import parse_txt, input_text_process
        print("Successfully imported parse_txt and input_text_process")
        # Define dummy functions for missing imports
        async def process_folder_async(*args, **kwargs):
            print("Error: process_folder_async not available")
            return []
        
        async def process_folder_async_with_history(*args, **kwargs):
            print("Error: process_folder_async_with_history not available")
            return []
    except ImportError as e2:
        print(f"Error: Could not import parse_txt and input_text_process: {e2}")
        # Define all dummy functions if imports fail
        async def process_folder_async(*args, **kwargs):
            print("Error: process_folder_async not available")
            return []
        
        async def process_folder_async_with_history(*args, **kwargs):
            print("Error: process_folder_async_with_history not available")
            return []
        
        async def parse_txt(*args, **kwargs):
            print("Error: parse_txt not available")
            return []
        
        async def input_text_process(*args, **kwargs):
            print("Error: input_text_process not available")
            return None

# Import the enhanced file processor
try:
    from enhanced_file_processor import EnhancedFileProcessor
    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError:
    print("Warning: Enhanced file processor not available, using legacy mode")
    ENHANCED_PROCESSOR_AVAILABLE = False

async def process_folders(folders, txt_path, temporary_folder, index=9, maximum_tasks=20, selected_task_number=500, storage_folder=None, read_hist=False):
    total_tasks = []
    if read_hist and os.path.exists(os.path.join(storage_folder, "total_response.json")):
        print(f"Reading history data from {storage_folder}/total_response.json")
        history_data = json.load(open(os.path.join(storage_folder, "total_response.json"), "r", encoding="utf-8"))
        processed_files = [example["source_file"] for example in history_data if "source_file" in example]
        processed_files = list(set(processed_files))  # remove duplicates
    else:
        processed_files = []
    
    print(f"Already processed files: {len(processed_files) if processed_files else 'None'}")
    await asyncio.sleep(1)
    
    total_responses = []
    
    # 使用增强文件处理器
    if ENHANCED_PROCESSOR_AVAILABLE and ("pdf" in txt_path.lower() or "text" in txt_path.lower()):
        print("使用增强文件处理器...")
        processor = EnhancedFileProcessor()
        
        # 处理目录
        pdf_results, txt_results = await processor.process_directory(txt_path)
        
        # 准备召回数据
        retrieval_data = processor.prepare_for_retrieval(pdf_results, txt_results)
        
        print(f"处理完成: {len(pdf_results)} 个PDF文件, {len(txt_results)} 个文本文件")
        
        # 转换为任务格式
        task_metadata = []  # 保存任务对应的元数据
        for data in retrieval_data:
            if data['source_file'] not in processed_files:
                # 创建任务（不要await，保留为协程）
                task = input_text_process(
                    data['content'], 
                    data['source_file'],
                    chunk_index=0,
                    total_chunks=1,
                    prompt_index=index
                )
                    
                if task:
                    total_tasks.append(task)
                    task_metadata.append(data)  # 保存对应的元数据
                    
                    if len(total_tasks) >= maximum_tasks:
                        print(f"Total tasks {len(total_tasks)} exceeds the maximum_tasks {maximum_tasks}, processing batch...")
                        batch_responses = await asyncio.gather(*total_tasks)
                        # Filter out None responses and add metadata
                        for i, response in enumerate(batch_responses):
                            if response is not None and isinstance(response, dict):
                                response['source_file'] = task_metadata[i]['source_file']
                                response['file_type'] = task_metadata[i]['file_type']
                                total_responses.append(response)
                        
                        total_tasks = []  # reset the tasks list
                        task_metadata = []  # reset metadata list
                        
                        # Save batch
                        os.makedirs(os.path.join(storage_folder, temporary_folder), exist_ok=True)
                        batch_num = len(total_responses) // maximum_tasks + 1
                        with open(os.path.join(storage_folder, temporary_folder, f"batch_{batch_num}.json"), "w",
                            encoding="utf-8") as f:
                            json.dump(batch_responses, f, ensure_ascii=False, indent=4)
                            print(f"Batch {batch_num} responses saved")
        
        # 处理剩余的任务
        if total_tasks:
            print(f"Processing remaining {len(total_tasks)} tasks...")
            batch_responses = await asyncio.gather(*total_tasks)
            for i, response in enumerate(batch_responses):
                if response is not None and isinstance(response, dict):
                    response['source_file'] = task_metadata[i]['source_file']
                    response['file_type'] = task_metadata[i]['file_type']
                    total_responses.append(response)
        
    else:
        # 原有的处理逻辑（仅处理txt文件）
        for folder in folders:
            file_path = os.path.join(txt_path, folder)
            
            # Skip if not a txt file
            if not folder.endswith('.txt'):
                continue
                
            # Skip if already processed
            if folder in processed_files:
                print(f"File {folder} already processed, skipping.")
                continue
                
            if not os.path.isfile(file_path):
                continue
                
            try:
                # Process txt file
                tasks = await parse_txt(file_path, index=index)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
            if len(tasks) == 0:
                print(f"No tasks found in {file_path}, skipping this file.")
                continue
                
            total_tasks.extend(tasks)
            
            if len(total_tasks) >= maximum_tasks:
                print(f"Total tasks {len(total_tasks)} exceeds the maximum_tasks {maximum_tasks}, waiting for tasks to complete.")
                batch_responses = await asyncio.gather(*total_tasks)
                # Filter out None responses
                batch_responses = [response for response in batch_responses if response is not None]
                total_responses.extend(batch_responses)
                total_tasks = []  # reset the tasks list
                
                os.makedirs(os.path.join(storage_folder, temporary_folder), exist_ok=True)
                with open(os.path.join(storage_folder, temporary_folder, f"batch_{len(total_responses)//maximum_tasks + 1}.json"), "w",
                    encoding="utf-8") as f:
                    json.dump(batch_responses, f, ensure_ascii=False, indent=4)
                    print(f"Batch {len(total_responses)//maximum_tasks + 1} responses saved to {storage_folder}/{temporary_folder}/batch_{len(total_responses)//maximum_tasks + 1}.json")
        
        # 处理原有逻辑的剩余任务
        if total_tasks:
            print(f"Processing remaining {len(total_tasks)} tasks...")
            batch_responses = await asyncio.gather(*total_tasks)
            batch_responses = [response for response in batch_responses if response is not None]
            total_responses.extend(batch_responses)
                
    # 注意：剩余任务的处理已经在上面的增强处理器和原有处理逻辑中完成了
    return total_responses


async def main(index=43, parallel_batch_size=100, pdf_path=None, storage_folder=None, 
              selected_task_number=1000, read_hist=False):
    """
    Main function for text retrieval processing
    
    Args:
        index: Task index for processing
        parallel_batch_size: Number of parallel tasks
        pdf_path: Path to input files (can be PDF or text directory)
        storage_folder: Output storage folder
        selected_task_number: Number of tasks to select
        read_hist: Whether to read history file
    """
    if pdf_path is None:
        pdf_path = "/workspace/text_qa_generation/data/input_texts"
    if storage_folder is None:
        storage_folder = "/workspace/text_qa_generation/data/output"
    
    temporary_folder = "TEMP"
    
    # Create output directory
    os.makedirs(storage_folder, exist_ok=True)
    
    # Get all txt files
    if os.path.isdir(pdf_path):
        files = [f for f in os.listdir(pdf_path) if f.endswith('.txt')]
    else:
        # Single file processing
        files = [os.path.basename(pdf_path)]
        pdf_path = os.path.dirname(pdf_path)
    
    # Set global args for backward compatibility
    global args
    args = type('Args', (), {
        'index': index,
        'parallel_batch_size': parallel_batch_size,
        'txt_path': pdf_path,
        'storage_folder': storage_folder,
        'temporary_folder': temporary_folder,
        'selected_task_number': selected_task_number,
        'read_hist': read_hist
    })()
    
    # Process files
    final_results = await process_folders(files, pdf_path, temporary_folder, 
                                        index=index, 
                                        maximum_tasks=parallel_batch_size,
                                        selected_task_number=selected_task_number,
                                        storage_folder=storage_folder,
                                        read_hist=read_hist)
    
    # Save results
    output_file = os.path.join(storage_folder, "total_response.pkl")
    with open(output_file, "wb") as f:
        if read_hist and os.path.exists(os.path.join(storage_folder, "total_response.json")):
            history_data = json.load(open(os.path.join(storage_folder, "total_response.json"), "r", encoding="utf-8"))
            final_results.extend(history_data)
        pkl.dump(final_results, f)
    
    print(f"Total {len(final_results)} responses saved to {output_file}")
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files for QA generation.")
    parser.add_argument("--index", type=int, default=43, help="index for the task")
    parser.add_argument("--parallel_batch_size", type=int, default=100, help="number of parallel tasks")
    parser.add_argument("--txt_path", type=str, default="/workspace/text_qa_generation/data/input_texts", help="path to the txt folder")
    parser.add_argument("--storage_folder", type=str, default="/workspace/text_qa_generation/data/output", help="path to the storage folder")
    parser.add_argument("--temporary_folder", type=str, default="TEMP", help="path to the temporary folder")
    parser.add_argument("--selected_task_number", type=int, default=1000, help="number of tasks to select")
    parser.add_argument("--read_hist", type=bool, default=False, help="whether to read the history file")
    args = parser.parse_args()
    
    selected_task_number = args.selected_task_number
    index = args.index
    parallel_batch_size = args.parallel_batch_size
    txt_path = args.txt_path
    storage_folder = args.storage_folder
    os.makedirs(storage_folder, exist_ok=True)
    temporary_folder = args.temporary_folder
    
    # Get all txt files
    files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]
    
    total_responses = []
    os.makedirs(storage_folder, exist_ok=True)
    
    final_results = asyncio.run(process_folders(files, txt_path, temporary_folder, index=index, 
                                               maximum_tasks=parallel_batch_size,
                                               selected_task_number=selected_task_number,
                                               storage_folder=storage_folder,
                                               read_hist=args.read_hist))
    
    # Save results
    f = open(os.path.join(storage_folder, "total_response.pkl"), "wb")
    if args.read_hist:
        history_data = json.load(open(os.path.join(storage_folder, "total_response.json"), "r", encoding="utf-8"))
        final_results.extend(history_data)
        pkl.dump(final_results, f)
    else:
        pkl.dump(final_results, f)
        
    print(f"Total {len(final_results)} responses saved to {storage_folder}/total_response.pkl")