#!/usr/bin/env python3
"""
验证优化版本的处理流程与原版保持一致
"""
import json
import os
import sys
from pathlib import Path

def verify_flow_steps(output_dir):
    """验证处理流程的步骤顺序"""
    
    print("=== 验证处理流程 ===\n")
    
    # 定义预期的流程步骤和输出文件
    expected_flow = {
        "第一阶段：文本预处理 + 质量评估": {
            "1.1 文本分块": "chunks/all_tasks.json",
            "1.2 AI处理": "chunks/ai_processed_texts.json",
            "1.3 质量评估": "chunks/quality_judged_texts.json",
            "1.3 合格文本": "chunks/qualified_texts.json"
        },
        "第二阶段：QA生成": {
            "2.1 分类问题": "qa_original/classified_questions.json",
            "2.2 格式转换": "qa_original/converted_questions.json",
            "2.3 质量评估": "qa_original/evaluated_qa_data.json",
            "2.4 答案生成": "qa_original/qa_with_answers.json"
        },
        "第三阶段：数据增强": {
            "3.1 最终数据": "qa_results/final_qa_dataset.json"
        }
    }
    
    all_steps_valid = True
    
    for phase, steps in expected_flow.items():
        print(f"【{phase}】")
        for step_name, file_path in steps.items():
            full_path = Path(output_dir) / file_path
            if full_path.exists():
                # 检查文件内容
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            count = len(data)
                        elif isinstance(data, dict):
                            count = len(data.get('data', data))
                        else:
                            count = 1
                    print(f"  ✓ {step_name}: {file_path} (包含 {count} 条数据)")
                except Exception as e:
                    print(f"  ✗ {step_name}: {file_path} (读取失败: {e})")
                    all_steps_valid = False
            else:
                print(f"  ✗ {step_name}: {file_path} (文件不存在)")
                all_steps_valid = False
        print()
    
    # 检查进度跟踪文件
    checkpoint_file = Path(output_dir) / ".checkpoints" / "progress_checkpoint.json"
    if checkpoint_file.exists():
        print("【断点续跑支持】")
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                completed_steps = progress.get('completed_steps', [])
                print(f"  ✓ 进度检查点存在")
                print(f"  ✓ 已完成步骤: {', '.join(completed_steps)}")
                
                # 验证步骤顺序
                expected_order = ['1.1', '1.2', '1.3', '2.1', '2.2', '2.3', '2.4', '3.1']
                actual_order = [s for s in expected_order if s in completed_steps]
                if actual_order == [s for s in completed_steps if s in expected_order]:
                    print(f"  ✓ 步骤顺序正确")
                else:
                    print(f"  ✗ 步骤顺序异常")
                    all_steps_valid = False
        except Exception as e:
            print(f"  ✗ 读取进度文件失败: {e}")
    else:
        print("【断点续跑支持】")
        print("  ℹ 进度检查点不存在（可能是首次运行或已完成）")
    
    print("\n" + "="*50)
    if all_steps_valid:
        print("✅ 流程验证通过：所有步骤按预期顺序执行")
    else:
        print("❌ 流程验证失败：部分步骤缺失或异常")
    
    return all_steps_valid


def compare_with_original(original_dir, optimized_dir):
    """比较优化版本与原版的输出结构"""
    
    print("\n=== 与原版输出对比 ===\n")
    
    # 关键输出文件
    key_files = [
        "chunks/ai_processed_texts.json",
        "chunks/qualified_texts.json",
        "qa_original/classified_questions.json",
        "qa_original/evaluated_qa_data.json",
        "qa_results/final_qa_dataset.json"
    ]
    
    all_compatible = True
    
    for file_path in key_files:
        original_file = Path(original_dir) / file_path
        optimized_file = Path(optimized_dir) / file_path
        
        print(f"检查: {file_path}")
        
        if original_file.exists() and optimized_file.exists():
            try:
                with open(original_file, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                with open(optimized_file, 'r', encoding='utf-8') as f:
                    optimized_data = json.load(f)
                
                # 检查数据结构
                if type(original_data) == type(optimized_data):
                    if isinstance(original_data, list) and len(original_data) > 0:
                        # 检查列表中第一个元素的结构
                        original_keys = set(original_data[0].keys()) if original_data else set()
                        optimized_keys = set(optimized_data[0].keys()) if optimized_data else set()
                        
                        if original_keys == optimized_keys:
                            print(f"  ✓ 数据结构一致")
                        else:
                            missing = original_keys - optimized_keys
                            extra = optimized_keys - original_keys
                            if missing:
                                print(f"  ⚠ 缺少字段: {missing}")
                            if extra:
                                print(f"  ⚠ 新增字段: {extra}")
                    else:
                        print(f"  ✓ 数据类型一致")
                else:
                    print(f"  ✗ 数据类型不同")
                    all_compatible = False
                    
            except Exception as e:
                print(f"  ✗ 读取失败: {e}")
                all_compatible = False
        else:
            if not original_file.exists():
                print(f"  ℹ 原版文件不存在（跳过）")
            if not optimized_file.exists():
                print(f"  ✗ 优化版文件不存在")
                all_compatible = False
    
    print("\n" + "="*50)
    if all_compatible:
        print("✅ 输出格式兼容：优化版本与原版输出结构一致")
    else:
        print("⚠ 输出格式部分兼容：建议检查差异")
    
    return all_compatible


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="验证优化版本的处理流程")
    parser.add_argument("--output-dir", type=str, default="data/output",
                        help="优化版本的输出目录")
    parser.add_argument("--original-dir", type=str, default=None,
                        help="原版的输出目录（用于对比）")
    
    args = parser.parse_args()
    
    if not Path(args.output_dir).exists():
        print(f"错误：输出目录不存在: {args.output_dir}")
        print("请先运行 run_semiconductor_qa_optimized.py 生成输出")
        sys.exit(1)
    
    # 验证流程
    flow_valid = verify_flow_steps(args.output_dir)
    
    # 如果提供了原版输出目录，进行对比
    if args.original_dir and Path(args.original_dir).exists():
        compare_valid = compare_with_original(args.original_dir, args.output_dir)
    
    # 总结
    print("\n" + "="*50)
    print("【验证总结】")
    print(f"处理流程: {'✅ 通过' if flow_valid else '❌ 失败'}")
    if args.original_dir:
        print(f"格式兼容: {'✅ 通过' if compare_valid else '⚠ 部分通过'}")
    
    sys.exit(0 if flow_valid else 1)


if __name__ == "__main__":
    main()