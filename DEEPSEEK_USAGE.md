# DeepSeek QA生成系统使用指南

## 概述

本系统使用DeepSeek-R1模型替代火山API，保持了原有的7步骤处理流程，支持大规模并行处理。

## 安装依赖

```bash
pip install aiohttp asyncio tqdm
```

## 设置API密钥

### 方法1: 环境变量（推荐）
```bash
export DEEPSEEK_API_KEY='your_deepseek_api_key'
```

### 方法2: 命令行参数
```bash
python run_deepseek_qa.py --api-key 'your_deepseek_api_key'
```

## 快速开始

### 1. 测试API连接
```bash
python test_deepseek.py
```

### 2. 运行QA生成
```bash
python run_deepseek_qa.py \
    --input-dir data/texts \
    --output-dir data/output \
    --max-files 5000 \
    --target-qa 20000 \
    --max-workers 16 \
    --batch-size 10
```

## 处理流程

系统按照以下7个步骤处理：

1. **步骤1.1: 文本预处理**
   - 读取文本文件
   - 分块处理（每2000字符一块）
   - 并行文件读取

2. **步骤1.2: 文本召回与批量推理**
   - 使用DeepSeek-R1生成问题
   - 批量并行处理
   - 自动重试机制

3. **步骤1.3: 数据清洗**
   - 过滤低质量问题
   - 去除冗余内容
   - 标准化格式

4. **步骤1.4: 核心QA生成**
   - 为问题生成高质量答案
   - 并行答案生成
   - 上下文感知

5. **步骤1.5: 质量检查**
   - 评估QA对质量
   - 计算质量分数
   - 筛选高质量数据

6. **步骤1.6: 数据增强与重写**
   - 生成问题变体
   - 增加数据多样性
   - 保持语义一致性

7. **步骤1.7: 最终输出整理**
   - 保存JSON格式结果
   - 生成统计报告
   - 性能分析

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | data/texts | 输入文本目录 |
| `--output-dir` | data/output | 输出目录 |
| `--max-files` | 5000 | 最大处理文件数 |
| `--target-qa` | 20000 | 目标QA对数量 |
| `--api-key` | None | DeepSeek API密钥 |
| `--model` | deepseek-r1 | 使用的模型 |
| `--max-workers` | 16 | 最大并发数 |
| `--batch-size` | 10 | 批处理大小 |

## 输出文件

系统会在输出目录生成以下文件：

1. `deepseek_qa_results_[timestamp].json` - QA数据结果
2. `deepseek_qa_stats_[timestamp].json` - 统计信息

## 性能优化建议

1. **并发设置**
   - 根据API限制调整 `--max-workers`
   - 批处理大小影响内存使用

2. **文本处理**
   - 大文件自动分块处理
   - 支持多线程文件读取

3. **错误处理**
   - 自动重试失败请求
   - 跳过处理失败的文件

## 故障排除

### API密钥错误
```
错误: 请设置DeepSeek API密钥
```
解决：确保正确设置了API密钥

### 速率限制
```
达到速率限制，等待X秒后重试...
```
解决：减少 `--max-workers` 参数值

### 内存不足
解决：减少 `--batch-size` 参数值

## 与原系统的差异

1. **API替换**: 使用DeepSeek-R1替代火山API
2. **并行优化**: 增强的并行处理能力
3. **错误处理**: 更完善的重试机制
4. **质量控制**: 保留原有的7步骤流程

## 示例输出

```json
{
  "question": "什么是半导体的能带结构？",
  "answer": "半导体的能带结构是指...",
  "source_file": "semiconductor_basics.txt",
  "chunk_id": 0,
  "metadata": {
    "model": "deepseek-r1",
    "timestamp": 1234567890
  }
}
```

## 注意事项

1. 确保输入目录包含 `.txt` 文件
2. API调用会产生费用，请注意使用量
3. 大规模处理建议分批进行
4. 保持网络连接稳定

## 技术支持

如遇到问题，请检查：
1. API密钥是否正确
2. 网络连接是否正常
3. 输入文件格式是否正确
4. 日志文件中的错误信息