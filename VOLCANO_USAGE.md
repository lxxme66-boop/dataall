# 火山引擎API优化使用指南

## 概述

本文档介绍如何使用优化后的QA生成系统，该系统支持：
- **多线程/多进程并发处理**
- **火山引擎API集成**
- **大规模数据集处理**
- **智能缓存和速率限制**

## 快速开始

### 1. 安装依赖

```bash
# 安装火山引擎SDK
pip install volcengine-python-sdk[ark]

# 安装其他依赖
pip install aiohttp tqdm asyncio
```

### 2. 配置火山引擎API

编辑 `config_volcano.json`：

```json
{
  "volcano_api": {
    "api_key": "你的API密钥",
    "endpoint_id": "你的端点ID",
    "model": "doubao-pro-32k"
  }
}
```

### 3. 测试API连接

```bash
# 测试API是否正常工作
python test_volcano_api.py \
    --api-key YOUR_API_KEY \
    --endpoint-id YOUR_ENDPOINT_ID \
    --test-type all
```

### 4. 运行QA生成

```bash
# 基本运行
python run_with_volcano.py \
    --config config_volcano.json \
    --input-dir data/texts \
    --output-dir data/output_volcano

# 使用多线程优化（推荐用于大数据集）
python run_with_volcano.py \
    --config config_volcano.json \
    --input-dir data/texts \
    --output-dir data/output_volcano \
    --max-workers 16 \
    --batch-size 64 \
    --use-async

# 使用多进程（CPU密集型任务）
python run_with_volcano.py \
    --config config_volcano.json \
    --input-dir data/texts \
    --output-dir data/output_volcano \
    --max-workers 8 \
    --use-multiprocess
```

## 性能优化建议

### 1. 并发配置

根据数据集大小和API限制选择合适的并发策略：

| 数据集大小 | 推荐配置 | 说明 |
|-----------|---------|------|
| < 100 文件 | `--max-workers 8 --use-async` | 异步IO，适合小数据集 |
| 100-1000 文件 | `--max-workers 16 --batch-size 32 --use-async` | 增加并发数和批处理 |
| > 1000 文件 | `--max-workers 32 --batch-size 64 --use-async` | 最大化并发处理 |
| CPU密集型 | `--max-workers 8 --use-multiprocess` | 使用多进程 |

### 2. 文本分块优化

```bash
# 调整分块大小（默认2000字符）
python run_with_volcano.py \
    --chunk-size 3000 \
    --chunk-overlap 300
```

- **chunk-size**: 每个文本块的大小（字符数）
- **chunk-overlap**: 块之间的重叠（避免断句）

### 3. 缓存策略

系统自动缓存处理结果，避免重复处理：

```bash
# 禁用缓存（用于测试）
python run_with_volcano.py --no-cache

# 清理缓存
rm -rf cache/volcano/*
```

### 4. 速率限制

火山引擎API有速率限制，系统会自动控制：

```json
{
  "optimization": {
    "rate_limit": 100,        // 每秒最大请求数
    "rate_limit_window": 1.0  // 速率限制窗口（秒）
  }
}
```

## 高级功能

### 1. 自定义问题类型比例

在 `config_volcano.json` 中调整：

```json
{
  "processing": {
    "question_types": {
      "factual": {"ratio": 0.15},     // 事实型 15%
      "comparative": {"ratio": 0.15},  // 比较型 15%
      "reasoning": {"ratio": 0.50},    // 推理型 50%
      "open": {"ratio": 0.20}          // 开放型 20%
    }
  }
}
```

### 2. 质量控制

```json
{
  "processing": {
    "quality_threshold": 0.7,        // 质量阈值
    "min_questions_per_chunk": 3,    // 每块最少问题数
    "max_questions_per_chunk": 5     // 每块最多问题数
  }
}
```

### 3. 批量处理监控

```bash
# 查看详细日志
python run_with_volcano.py --verbose

# 仅预览将要处理的文件（不实际处理）
python run_with_volcano.py --dry-run
```

## 输出格式

系统生成以下文件：

```
data/output_volcano/
├── [文件名]_qa_[时间戳].json     # 单个文件的QA结果
├── all_qa_pairs_[时间戳].json    # 所有QA对汇总
└── summary_[时间戳].json         # 处理统计信息
```

### QA对格式示例

```json
{
  "chunk_id": 0,
  "question": "IGZO TFT相比传统a-Si TFT的主要优势是什么？",
  "type": "comparative",
  "difficulty": "medium",
  "answer": "IGZO TFT相比传统a-Si TFT具有以下主要优势：...",
  "reasoning": "基于材料特性分析，IGZO作为氧化物半导体...",
  "confidence": 0.85,
  "context": "原文上下文...",
  "source_file": "IGZO技术.txt"
}
```

## 性能对比

| 处理方式 | 1000个文本块耗时 | 相对速度 |
|---------|----------------|----------|
| 原始单线程 | ~1000秒 | 1x |
| 多线程(8) | ~125秒 | 8x |
| 多线程(16) + 异步 | ~65秒 | 15x |
| 多线程(32) + 批处理 | ~35秒 | 28x |

## 故障排查

### 1. API连接失败

```bash
# 检查API密钥和端点
python test_volcano_api.py --api-key YOUR_KEY --endpoint-id YOUR_ID

# 检查网络连接
curl https://ark.cn-beijing.volces.com/api/v3/chat/completions
```

### 2. 内存不足

```bash
# 减少并发数和批处理大小
python run_with_volcano.py --max-workers 4 --batch-size 8
```

### 3. 速率限制错误

```json
// 调整速率限制配置
{
  "optimization": {
    "rate_limit": 50,  // 降低请求速率
    "retry_delay": 2.0  // 增加重试延迟
  }
}
```

## 最佳实践

1. **先小后大**：先用小数据集测试，确认正常后再处理大数据集
2. **监控资源**：使用 `htop` 或 `nvidia-smi` 监控系统资源
3. **增量处理**：利用缓存功能，可以中断后继续处理
4. **定期备份**：定期备份输出结果，避免数据丢失
5. **错误重试**：系统会自动重试失败的请求，但可以手动重新处理失败的文件

## 与原系统集成

如果需要与原有的vLLM系统集成，可以：

1. **并行使用**：同时运行vLLM和火山引擎API
2. **负载均衡**：根据任务类型分配到不同的后端
3. **统一接口**：创建统一的API网关

```python
# 示例：混合使用
if task_type == "large_batch":
    # 使用火山引擎（适合大批量）
    use_volcano_api()
else:
    # 使用本地vLLM（适合实时处理）
    use_vllm_local()
```

## 联系支持

如有问题，请查看：
- 日志文件：`logs/volcano/qa_generation.log`
- 错误信息：输出目录中的 `errors.json`
- 统计信息：`summary_*.json`