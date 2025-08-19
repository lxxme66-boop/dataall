import pandas as pd
import argparse

def filter_qa_pairs(input_file, output_file, quality_level):
    """根据质量级别筛选问答对并保存到Excel"""
    df = pd.read_excel(input_file)

    # 列名映射
    col_map = {col.lower(): col for col in df.columns}

    # 检查必须列
    required_cols = {'quality_rating', '答案', '问题', '来源文件', '思维链'}
    missing = [col for col in required_cols if col.lower() not in col_map]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    rating_col = col_map['quality_rating']
    answer_col = col_map['答案']
    question_col = col_map['问题']
    type_col = col_map['来源文件']
    chain_col = col_map['思维链']

    # 筛选
    if quality_level == "high":
        filtered_df = df[df[rating_col].str.lower() == 'high']
    elif quality_level == "medium_high":
        filtered_df = df[df[rating_col].str.lower().isin(['high', 'medium'])]
    else:
        raise ValueError("无效的质量级别，请选择 'high' 或 'medium_high'")

    # 保留必要列（包含思维链）
    filtered_df = filtered_df[[question_col, type_col, answer_col, chain_col]]

    # 保存
    filtered_df.to_excel(output_file, index=False)
    print(f"已保存 {len(filtered_df)}/{len(df)} 条问答对到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="问答对质量筛选工具")
    parser.add_argument("--input", type=str, required=True, help="输入Excel文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出Excel文件路径")
    parser.add_argument("--quality", type=str, required=True,
                        choices=["high", "medium_high"],
                        help="质量级别: high(仅高质量) 或 medium_high(中+高质量)")

    args = parser.parse_args()

    try:
        filter_qa_pairs(args.input, args.output, args.quality)
    except Exception as e:
        print(f"处理失败: {str(e)}")