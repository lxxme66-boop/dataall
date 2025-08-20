# 半导体QA生成系统 - 快速启动指南

## 🚀 快速开始

### 首次运行（使用火山API + 多线程）

```bash
# 设置火山API密钥（替换为您的实际密钥）
export VOLCANO_API_KEY="your-api-key-here"

# 运行完整流程
./run_enhanced.sh \
    --use-volcano \
    --volcano-api-key $VOLCANO_API_KEY \
    --max-workers 8
```

### 基础运行（不使用火山API）

```bash
# 使用原有模型，4线程并行
./run_enhanced.sh --max-workers 4
```

### 从断点恢复

```bash
# 自动从最新断点恢复
./run_enhanced.sh --resume

# 从特定步骤恢复（例如从2.1恢复）
./run_enhanced.sh --resume --resume-from 2.1
```

## 📋 完整命令示例

### 1. 完整配置运行

```bash
./run_enhanced.sh \
    --input-dir data/semiconductor_texts \
    --output-dir results/qa_dataset \
    --config config.json \
    --model deepseek-r1 \
    --quality-threshold 0.8 \
    --batch-size 32 \
    --max-workers 16 \
    --use-volcano \
    --volcano-api-key $VOLCANO_API_KEY
```

### 2. 断点恢复运行

```bash
# 查看可用断点
ls -la checkpoints/

# 从最新断点恢复
./run_enhanced.sh --resume

# 从步骤2.3恢复
./run_enhanced.sh --resume --resume-from 2.3
```

### 3. 仅运行特定阶段

```bash
# 从步骤2.1开始（跳过文本处理）
./run_enhanced.sh --resume-from 2.1
```

## 🔄 流程步骤说明

| 步骤 | 名称 | 说明 |
|------|------|------|
| 1.1 | 文本分块和预处理 | 将输入文本分块处理 |
| 1.2 | AI文本处理 | 使用AI模型处理文本 |
| 1.3 | 文本质量评估 | 评估文本质量，筛选合格文本 |
| 2.1 | 分类问题生成 | 生成分类问题 |
| 2.2 | 问题格式转换 | 转换问题格式 |
| 2.3 | 问题质量评估 | 评估问题质量 |
| 2.4 | 答案生成 | 为高质量问题生成答案 |
| 3.1 | 数据增强 | 增强和重写QA数据 |

## 🔧 参数说明

### 基本参数
- `--input-dir`: 输入文本目录（默认: data/input）
- `--output-dir`: 输出结果目录（默认: data/output）
- `--config`: 配置文件路径（默认: config.json）
- `--model`: 使用的模型（默认: deepseek-r1）
- `--quality-threshold`: 质量阈值（默认: 0.7）
- `--batch-size`: 批处理大小（默认: 16）

### 性能参数
- `--max-workers`: 最大线程数（默认: 4，建议: 8-16）
- `--use-volcano`: 启用火山API
- `--volcano-api-key`: 火山API密钥

### 断点参数
- `--resume`: 从最新断点恢复
- `--resume-from`: 从指定步骤恢复（1.1-3.1）
- `--checkpoint-dir`: 断点保存目录（默认: checkpoints）

## 📊 监控进度

### 实时查看日志
```bash
# 查看最新日志
tail -f logs/enhanced_run_*.log

# 查看特定步骤日志
grep "步骤2.1" logs/enhanced_run_*.log
```

### 查看断点状态
```bash
# 列出所有断点
ls -la checkpoints/

# 查看最新断点
ls -lt checkpoints/ | head -5
```

### 查看统计信息
```bash
# 查看流程统计
cat data/output/pipeline_stats.json | python -m json.tool
```

## ⚡ 性能优化建议

### 1. 线程数设置
- CPU核心数 ≤ 4: 使用 4 线程
- CPU核心数 8-16: 使用 8 线程
- CPU核心数 > 16: 使用 16 线程

### 2. 批处理大小
- 内存 < 16GB: batch_size = 8
- 内存 16-32GB: batch_size = 16
- 内存 > 32GB: batch_size = 32

### 3. 火山API优化
- 使用火山API可以显著提升处理速度
- 建议配合多线程使用（8-16线程）
- 确保网络连接稳定

## 🐛 故障排除

### 1. 断点恢复失败
```bash
# 清理损坏的断点
rm checkpoints/*

# 重新开始
./run_enhanced.sh
```

### 2. 内存不足
```bash
# 减少线程数和批处理大小
./run_enhanced.sh --max-workers 2 --batch-size 8
```

### 3. 火山API连接失败
```bash
# 测试API连接
curl -X POST https://ark.cn-beijing.volces.com/api/v3/chat/completions \
  -H "Authorization: Bearer $VOLCANO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1","messages":[{"role":"user","content":"test"}]}'
```

## 📁 输出文件结构

```
data/output/
├── chunks/                      # 文本处理结果
│   ├── ai_processed_texts.json  # AI处理后的文本
│   ├── quality_judged_texts.json # 质量评估结果
│   └── qualified_texts.json     # 合格文本
├── qa_original/                 # QA生成中间结果
│   ├── classified_questions.json # 分类问题
│   ├── converted_questions.json  # 格式转换后的问题
│   ├── evaluated_qa_data.json   # 评估后的QA
│   ├── qa_with_context.json     # 带上下文的QA
│   └── qa_with_answers.json     # 带答案的QA
├── qa_results/                  # 最终结果
│   ├── qa_generated.json        # 生成的QA数据
│   └── final_qa_dataset.json    # 最终增强后的数据集
└── pipeline_stats.json          # 流程统计信息
```

## 💡 使用技巧

1. **首次运行建议**：先用小数据集测试，确保流程正常
2. **断点使用**：长时间运行时，断点会自动保存，无需担心中断
3. **并行优化**：合理设置线程数，避免过载
4. **质量控制**：调整quality_threshold控制输出质量（0.7-0.9）
5. **资源监控**：运行时监控CPU和内存使用情况

## 📞 获取帮助

```bash
# 查看完整帮助
./run_enhanced.sh --help

# 查看Python脚本帮助
python3 run_semiconductor_qa_enhanced_full.py --help
```