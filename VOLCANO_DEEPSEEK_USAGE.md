# 火山引擎 DeepSeek-R1 QA生成系统使用指南

## 概述

本系统使用火山引擎API（DeepSeek-R1模型）进行大规模半导体技术QA数据集生成。系统支持多线程并行处理，能够高效处理4000个文档并生成20000条高质量QA数据。

## 主要特性

- ✅ **火山引擎API集成**：直接调用火山API，无需本地部署模型
- ✅ **DeepSeek-R1模型支持**：使用先进的DeepSeek-R1模型生成高质量内容
- ✅ **多线程并行处理**：支持32个并发线程，大幅提升处理速度
- ✅ **大规模数据处理**：可处理4000个文档，生成20000条QA数据
- ✅ **智能文本分块**：自动将长文本分割成适合处理的块
- ✅ **质量控制**：内置质量评估和筛选机制
- ✅ **进度跟踪**：实时显示处理进度和统计信息
- ✅ **断点续传**：支持中断后继续处理
- ✅ **数据增强**：可选的数据增强功能

## 快速开始

### 1. 环境准备

```bash
# 设置环境变量
export VOLCANO_API_KEY="your-volcano-api-key"
export VOLCANO_ENDPOINT_ID="your-endpoint-id"

# 运行设置脚本
chmod +x setup_volcano_deepseek.sh
./setup_volcano_deepseek.sh
```

### 2. 基础使用

```bash
# 处理示例数据
python run_volcano_qa.py \
  --input-dir data/texts \
  --output-dir data/volcano_output
```

### 3. 大规模处理

```bash
# 处理4000个文档，生成20000条QA
python run_volcano_qa.py \
  --input-dir data/texts \
  --output-dir data/volcano_output \
  --max-files 4000 \
  --target-qa 20000 \
  --workers 32
```

## 详细配置

### 配置文件说明

系统使用 `config_volcano_deepseek.json` 作为主配置文件：

```json
{
  "api": {
    "use_volcano": true,
    "volcano_api_key": "${VOLCANO_API_KEY}",
    "volcano_endpoint_id": "${VOLCANO_ENDPOINT_ID}",
    "volcano_region": "cn-beijing"
  },
  "models": {
    "volcano": {
      "model_name": "deepseek-r1",
      "temperature": 0.7,
      "max_tokens": 4096
    }
  },
  "parallel_processing": {
    "max_workers": 32,
    "batch_size": 200,
    "max_concurrent_api_calls": 100
  },
  "large_scale_processing": {
    "target_qa_count": 20000,
    "max_text_files": 4000,
    "questions_per_chunk": 5
  }
}
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-dir` | 输入文本目录 | `data/texts` |
| `--output-dir` | 输出目录 | `data/volcano_output` |
| `--config` | 配置文件路径 | `config_volcano_deepseek.json` |
| `--max-files` | 最大处理文件数 | `4000` |
| `--target-qa` | 目标QA数量 | `20000` |
| `--workers` | 并行工作线程数 | `32` |
| `--enable-enhancement` | 启用数据增强 | `False` |

## 处理流程

### 1. 文本预处理
- 扫描输入目录中的所有文本文件
- 将长文本分割成2000字的块
- 使用AI优化文本内容

### 2. 问题生成
- 为每个文本块生成5个高质量问题
- 问题类型包括：
  - 事实型（15%）
  - 比较型（15%）
  - 推理型（50%）
  - 开放型（20%）

### 3. 答案生成
- 使用Chain of Thought方式生成详细答案
- 包含推理步骤和关键信息总结

### 4. 质量控制
- 自动评估QA质量
- 过滤低质量内容
- 去重处理

### 5. 数据输出
- JSON格式：完整的QA数据
- 统计信息：处理指标和性能数据
- Pickle格式：用于后续处理

## 输出格式

### QA数据格式

```json
{
  "question": "OLED技术相比LCD技术有哪些优势？",
  "answer": "OLED技术相比LCD技术具有以下主要优势：\n1. 自发光特性...",
  "context": "原始文本的部分内容...",
  "timestamp": "2024-01-01T12:00:00",
  "quality_score": 0.85,
  "source_file": "semiconductor_tech.txt",
  "chunk_id": "chunk_001"
}
```

### 统计信息

```json
{
  "processed_files": 4000,
  "processed_chunks": 12000,
  "generated_qa_pairs": 20000,
  "failed_chunks": 50,
  "total_tokens": 15000000,
  "elapsed_time": 3600,
  "average_speed": 5.56
}
```

## 性能优化

### 1. 并行处理优化

```python
# 调整并行工作线程数
--workers 64  # 增加到64个线程

# 调整批处理大小
--batch-size 500  # 增加批大小
```

### 2. API调用优化

- 使用连接池复用HTTP连接
- 实现自适应速率限制
- 自动重试失败的请求

### 3. 内存优化

- 流式处理大文件
- 及时释放处理完的数据
- 使用生成器避免内存溢出

## 监控和调试

### 实时监控

```bash
# 查看实时日志
tail -f volcano_qa_generation.log

# 查看处理进度
watch -n 5 'grep "进度:" volcano_qa_generation.log | tail -1'

# 查看API调用统计
python -c "import json; print(json.dumps(json.load(open('metrics/volcano_processing_stats.json')), indent=2))"
```

### 错误处理

系统会自动处理以下错误：
- API调用失败：自动重试3次
- 文本处理错误：跳过并记录
- 内存不足：自动降低批大小

## 高级功能

### 1. 断点续传

```bash
# 从上次中断的地方继续
python run_volcano_qa.py \
  --resume-from checkpoints/volcano/latest.pkl
```

### 2. 数据增强

```bash
# 启用数据增强功能
python run_volcano_qa.py \
  --enable-enhancement \
  --enhancement-ratio 1.5
```

### 3. 自定义问题模板

```python
# 在配置文件中定义自定义模板
"question_templates": {
  "custom_type": {
    "template": "基于{context}，请解释{topic}的{aspect}",
    "ratio": 0.1
  }
}
```

## 常见问题

### Q1: API调用失败怎么办？

检查以下几点：
1. 环境变量是否正确设置
2. API密钥是否有效
3. 网络连接是否正常
4. 查看错误日志：`logs/volcano_errors.log`

### Q2: 处理速度太慢？

优化建议：
1. 增加工作线程数：`--workers 64`
2. 增大批处理大小：`--batch-size 500`
3. 启用缓存：配置文件中设置 `"enable_caching": true`

### Q3: 内存不足？

解决方案：
1. 减少工作线程数
2. 减小批处理大小
3. 启用内存优化：配置文件中设置 `"enable_memory_optimization": true`

### Q4: 如何提高QA质量？

方法：
1. 调整温度参数：降低到0.3-0.5
2. 增加质量阈值：`"min_quality_score": 0.8`
3. 启用数据增强：`--enable-enhancement`

## 最佳实践

### 1. 数据准备
- 确保输入文本质量高
- 文本应包含专业术语和技术细节
- 避免过短或重复的文本

### 2. 参数调优
- 根据硬件资源调整并行线程数
- 根据API限制调整并发调用数
- 根据质量要求调整温度参数

### 3. 结果验证
- 定期抽查生成的QA质量
- 使用质量评估工具验证
- 收集反馈并持续优化

## 系统要求

- Python 3.8+
- 内存：建议16GB以上
- 网络：稳定的互联网连接
- API：有效的火山引擎API密钥

## 支持和反馈

如有问题或建议，请：
1. 查看日志文件：`volcano_qa_generation.log`
2. 检查错误日志：`logs/volcano_errors.log`
3. 查看统计信息：`metrics/volcano_processing_stats.json`

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持火山引擎API集成
- 实现多线程并行处理
- 支持大规模数据处理（4000文档，20000 QA）
- 添加DeepSeek-R1模型支持

## 许可证

本项目采用MIT许可证。