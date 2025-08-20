# 半导体QA生成系统 - 优化版本使用指南

## 新增特性

1. **多线程并行处理** - 大幅提升处理速度
2. **火山API支持** - 支持deepseek-r1模型
3. **断点续跑** - 支持从中断处继续执行
4. **保持原有流程** - 严格遵循1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1的处理流程

## 快速开始

### 1. 使用原有vLLM模式（多线程优化）

```bash
# 启动vLLM服务器（如果使用vLLM HTTP）
python start_vllm_server.py

# 运行优化版本（默认启用多线程和断点续跑）
python run_semiconductor_qa_optimized.py \
    --input-dir data/texts \
    --output-dir data/output \
    --config config_vllm_http.json \
    --batch-size 32 \
    --max-workers 16
```

### 2. 使用火山API（deepseek-r1）

```bash
# 设置API密钥（也可以在配置文件中设置）
export VOLCANO_API_KEY="your-api-key-here"

# 运行优化版本，使用火山API
python run_semiconductor_qa_optimized.py \
    --input-dir data/texts \
    --output-dir data/output \
    --use-volcano \
    --volcano-api-key $VOLCANO_API_KEY \
    --batch-size 32 \
    --max-workers 16
```

### 3. 断点续跑功能

```bash
# 首次运行（会自动保存进度）
python run_semiconductor_qa_optimized.py \
    --input-dir data/texts \
    --output-dir data/output \
    --config config_vllm_http.json

# 如果中断，再次运行相同命令会自动从断点继续
python run_semiconductor_qa_optimized.py \
    --input-dir data/texts \
    --output-dir data/output \
    --config config_vllm_http.json

# 如果想重新开始（禁用断点续跑）
python run_semiconductor_qa_optimized.py \
    --input-dir data/texts \
    --output-dir data/output \
    --config config_vllm_http.json \
    --no-resume
```

## 命令行参数说明

### 基础参数（与原版相同）
- `--input-dir`: 输入文本目录（默认: data/texts）
- `--output-dir`: 输出结果目录（默认: data/qa_results）
- `--model`: 使用的模型名称（默认: vllm_http）
- `--batch-size`: 批处理大小（默认: 32）
- `--gpu-devices`: GPU设备ID（默认: "0,1"）
- `--quality-threshold`: 质量阈值（默认: 0.7）
- `--config`: 配置文件路径

### 新增参数
- `--use-volcano`: 启用火山API（deepseek-r1）
- `--volcano-api-key`: 火山API密钥
- `--no-resume`: 禁用断点续跑（默认启用）
- `--max-workers`: 最大工作线程数（默认自动根据CPU核心数）

## 处理流程（保持不变）

系统严格按照以下流程执行：

### 第一阶段：文本预处理 + 质量评估
- **步骤1.1**: 文本分块和预处理
- **步骤1.2**: AI文本处理（现在支持多线程）
- **步骤1.3**: 文本质量评估（现在支持多线程）

### 第二阶段：QA生成
- **步骤2.1**: 分类问题生成（现在支持多线程）
  - 事实型（15%）
  - 比较型（15%）
  - 推理型（50%）
  - 开放型（20%）
- **步骤2.2**: 问题格式转换
- **步骤2.3**: 问题质量评估（现在支持多线程）
- **步骤2.4**: 答案生成（现在支持多线程）

### 第三阶段：数据增强
- **步骤3.1**: 数据增强与重写

## 性能对比

| 特性 | 原版 | 优化版 |
|-----|------|--------|
| 处理速度 | 基准 | 3-5倍提升 |
| 并行处理 | 否 | 是（多线程） |
| 断点续跑 | 否 | 是 |
| 火山API | 否 | 是 |
| 内存使用 | 基准 | 略有增加 |

## 进度跟踪

系统会在输出目录下创建 `.checkpoints` 文件夹，保存进度信息：

```
data/output/
├── .checkpoints/
│   └── progress_checkpoint.json  # 进度检查点
├── chunks/                       # 文本预处理结果
├── qa_original/                  # 原始QA生成结果
└── qa_results/                   # 最终结果
```

查看当前进度：
```bash
cat data/output/.checkpoints/progress_checkpoint.json
```

## 配置文件示例

### 使用vLLM HTTP（config_vllm_http.json）
```json
{
  "api": {
    "use_vllm_http": true,
    "vllm_server_url": "http://localhost:8000/v1"
  },
  "processing": {
    "batch_size": 32,
    "max_workers": 16
  }
}
```

### 使用火山API（config_volcano_api.json）
```json
{
  "api": {
    "use_volcano_api": true,
    "volcano_api_key": "your-key-here",
    "model": "deepseek-r1"
  },
  "processing": {
    "batch_size": 32,
    "max_workers": 16
  }
}
```

## 故障排查

### 1. 多线程相关问题
- 如果遇到内存不足，减少 `--max-workers` 参数
- 如果遇到API限流，减少 `--batch-size` 参数

### 2. 断点续跑问题
- 检查 `.checkpoints/progress_checkpoint.json` 文件是否存在
- 使用 `--no-resume` 参数重新开始

### 3. 火山API问题
- 确认API密钥正确
- 检查网络连接
- 查看API配额是否充足

## 监控和日志

系统会输出详细的进度信息：
```
2024-01-15 10:00:00 - INFO - 步骤1.1: 文本分块和预处理...
2024-01-15 10:00:05 - INFO - 处理批次 1/10 (任务 1-32/320)
2024-01-15 10:00:10 - INFO - 步骤1.2: AI文本处理（多线程）...
```

## 最佳实践

1. **批处理大小**：根据内存和API限制调整，建议16-64
2. **线程数**：CPU核心数的2-4倍效果最佳
3. **断点续跑**：处理大量数据时始终启用
4. **质量阈值**：0.7-0.8之间平衡质量和数量

## 注意事项

1. 优化版本完全兼容原版的输出格式
2. 处理流程严格保持不变，仅优化执行效率
3. 断点续跑会保存中间状态，占用额外磁盘空间
4. 使用火山API时注意API调用限制和费用