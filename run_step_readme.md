# 模块化半导体QA生成系统使用指南

## 概述

此模块化版本的半导体QA生成系统允许您独立运行任意步骤，方便调试、中断恢复和分布式处理。每个步骤都有明确的输入输出，支持从任意步骤开始执行。

## 系统架构

```
阶段1: 文本预处理
├── 1.1 文本预处理和分块
├── 1.2 AI文本处理  
└── 1.3 文本质量评估

阶段2: QA生成
├── 2.1 分类问题生成
├── 2.2 问题格式转换
├── 2.3 问题质量评估
└── 2.4 答案生成

阶段3: 数据增强
└── 3.0 数据增强与重写
```

## 命令格式

```bash
python modular_qa_pipeline.py --step [步骤号] [其他参数]
```

## 详细步骤说明

### 步骤1.1: 文本预处理和分块

**功能**: 读取输入文本文件，进行分块处理，为后续AI处理做准备。

**命令**:
```bash
python run_step.py --step 1.1 \
  --input-dir data/texts \
  --output-dir data_step/qa_results
```

**输入**: 
- `--input-dir`: 包含.txt文件的目录

**输出**: 
- `chunks/preprocessing_tasks.json`: 分块任务列表

**说明**: 这是整个流水线的起点，必须首先执行。

---

### 步骤1.2: AI文本处理

**功能**: 使用AI模型对分块的文本进行预处理和优化。

**命令**:
```bash
python run_step.py --step 1.2 \
  --output-dir data_step/qa_results \
  --batch-size 4 \
  --config config.json
```

**依赖**: 需要步骤1.1完成

**输入**: 
- 从步骤1.1的输出自动加载
- `--batch-size`: 并行处理的批次大小

**输出**: 
- `chunks/ai_processed_texts.json`: AI处理后的文本内容

---

### 步骤1.3: 文本质量评估

**功能**: 评估AI处理后文本的质量，筛选适合生成QA的文本。

**命令**:
```bash
export USE_VLLM_HTTP=true
export VLLM_SERVER_URL=http://localhost:8000/v1
python run_step.py --step 1.3 \
  --output-dir data_step/qa_results \
  --model qwq_32 \
  --batch-size 2 \
  --gpu-devices "6,7"
```

**依赖**: 需要步骤1.2完成

**输入**: 
- 从步骤1.2的输出自动加载
- `--model`: 评估使用的模型
- `--gpu-devices`: GPU设备配置

**输出**: 
- `chunks/quality_judged_texts.json`: 质量评估结果
- `chunks/qualified_texts.json`: 通过评估的合格文本

---

### 步骤2.1: 分类问题生成

**功能**: 为合格文本生成不同类型的问题（事实型、比较型、推理型、开放型）。

**命令**:
```bash
export USE_VLLM_HTTP=true
export VLLM_SERVER_URL=http://localhost:8000/v1
python run_step.py --step 2.1 \
  --output-dir data_step/qa_results \
  --model qwq_32 \
  --batch-size 2 \
  --gpu-devices "6,7"
```

**依赖**: 需要步骤1.3完成

**输出**: 
- `qa_original/classified_questions.json`: 按类型分类的问题集合

**问题类型**:
- **factual**: 事实型问题（获取参数、数值等）
- **comparative**: 比较型问题（比较不同方案）
- **reasoning**: 推理型问题（机制原理解释）
- **open**: 开放型问题（优化建议）

---

### 步骤2.2: 问题格式转换

**功能**: 将分类的问题转换为统一的独立问题格式。

**命令**:
```bash
export USE_VLLM_HTTP=true
export VLLM_SERVER_URL=http://localhost:8000/v1
python run_step.py --step 2.2 \
  --output-dir data_step/qa_results \
  --model qwq_32
```

**依赖**: 需要步骤2.1完成

**输出**: 
- `qa_original/converted_questions.json`: 格式统一的独立问题列表

---

### 步骤2.3: 问题质量评估

**功能**: 评估问题质量，根据质量阈值筛选高质量问题。

**命令**:
```bash
export USE_VLLM_HTTP=true
export VLLM_SERVER_URL=http://localhost:8000/v1
python run_step.py --step 2.3 \
  --output-dir data_step/qa_results \
  --quality-threshold 0.7 \
  --model qwq_32
  
  
  
  --gpu-devices "6,7"
```

**依赖**: 需要步骤2.2完成

**参数**:
- `--quality-threshold`: 质量阈值（0-1之间，默认0.7）

**输出**: 
- `qa_original/evaluated_qa_data.json`: 带质量评分的问题数据
- `qa_original/high_quality_questions.json`: 高质量问题（≥阈值）

---

### 步骤2.4: 答案生成

**功能**: 为高质量问题生成详细答案。

**命令**:
```bash
export USE_VLLM_HTTP=true
export VLLM_SERVER_URL=http://localhost:8000/v1
python run_step.py --step 2.4 \
  --output-dir data_step/qa_results \
  --model qwq_32 \
  --batch-size 2 \
  --gpu-devices "6,7"
```

**依赖**: 需要步骤2.3完成

**输出**: 
- `qa_original/qa_with_context.json`: 带上下文的问题
- `qa_original/qa_with_answers.json`: 带答案的QA对
- `qa_results/qa_generated.json`: 最终生成的QA数据集

---

### 步骤3.0: 数据增强

**功能**: 对生成的QA数据进行增强和重写，提高数据质量和多样性。

**命令**:
```bash
export USE_VLLM_HTTP=true
export VLLM_SERVER_URL=http://localhost:8000/v1
python run_step.py --step 3.0 \
  --output-dir data_step/qa_results
```

**依赖**: 需要步骤2.4完成

**输出**: 
- `qa_results/final_qa_dataset.json`: 最终的增强QA数据集

## 特殊命令

### 查看状态

查看所有步骤的完成状态：

```bash
python run_step.py --step status --output-dir data/qa_results
```

### 生成报告

生成详细的流水线执行报告：

```bash
python modular_qa_pipeline.py --step report --output-dir data/qa_results
```

### 执行完整流水线

一次性执行所有步骤：

```bash
python modular_qa_pipeline.py --step all \
  --input-dir data/texts \
  --output-dir data/qa_results \
  --model qwq_32 \
  --batch-size 2 \
  --gpu-devices "0,1" \
  --quality-threshold 0.7 \
  --config config.json
```

## 输出目录结构

```
data/qa_results/
├── chunks/                          # 文本处理阶段输出
│   ├── preprocessing_tasks.json     # 1.1: 预处理任务
│   ├── ai_processed_texts.json      # 1.2: AI处理结果
│   ├── quality_judged_texts.json    # 1.3: 质量评估结果
│   └── qualified_texts.json         # 1.3: 合格文本
├── qa_original/                     # QA生成阶段中间输出
│   ├── classified_questions.json    # 2.1: 分类问题
│   ├── converted_questions.json     # 2.2: 转换后问题
│   ├── evaluated_qa_data.json       # 2.3: 质量评估结果
│   ├── high_quality_questions.json  # 2.3: 高质量问题
│   ├── qa_with_context.json         # 2.4: 带上下文QA
│   └── qa_with_answers.json         # 2.4: 带答案QA
├── qa_results/                      # 最终输出
│   ├── qa_generated.json            # 2.4: 生成的QA数据
│   └── final_qa_dataset.json        # 3.0: 最终数据集
├── step_status.json                 # 步骤状态跟踪
└── pipeline_report.json             # 流水线执行报告
```

## 使用场景

### 1. 完整流水线
```bash
# 从头开始执行完整流程
python modular_qa_pipeline.py --step all --input-dir data/texts --output-dir results
```

### 2. 中断恢复
```bash
# 查看当前状态
python modular_qa_pipeline.py --step status --output-dir results

# 从步骤2.1开始继续
python modular_qa_pipeline.py --step 2.1 --output-dir results --model qwq_32
```

### 3. 单步调试
```bash
# 只执行问题质量评估，调整阈值
python modular_qa_pipeline.py --step 2.3 --output-dir results --quality-threshold 0.8
```

### 4. 参数调优
```bash
# 重新执行答案生成，使用不同模型
python modular_qa_pipeline.py --step 2.4 --output-dir results --model different_model
```

## 配置文件示例

创建 `config.json` 文件：

```json
{
  "api": {
    "use_vllm_http": true,
    "vllm_server_url": "http://localhost:8000/v1",
    "use_local_models": true
  },
  "processing": {
    "batch_size": 4,
    "chunk_size": 1000
  },
  "quality_control": {
    "text_quality_threshold": 0.6,
    "question_quality_threshold": 0.7
  },
  "paths": {
    "text_dir": "data/texts",
    "output_dir": "data/qa_results"
  }
}
```

## 错误处理和恢复

1. **依赖检查**: 系统会自动检查前置步骤是否完成
2. **状态保存**: 每个步骤完成后都会保存状态到 `step_status.json`
3. **中断恢复**: 可以从任意已完成步骤的下一步开始
4. **错误日志**: 详细的错误信息和堆栈跟踪
5. **输出验证**: 每个步骤都会验证输出文件的存在和格式

## 最佳实践

1. **从小数据开始**: 先用少量数据测试完整流程
2. **分阶段执行**: 在生产环境中建议分阶段执行，便于监控和调试
3. **资源监控**: 注意GPU内存和磁盘空间使用情况
4. **定期检查状态**: 使用 `--step status` 监控进度
5. **保存中间结果**: 重要的中间结果都会保存，可供分析和调试
6. **配置管理**: 使用配置文件管理不同环境的参数

## 注意事项

- 每个步骤都会验证前置依赖，确保数据的完整性
- GPU资源会在需要时自动分配，注意避免资源冲突
- 大量数据处理时建议适当调整批次大小
- 质量阈值的设置直接影响最终数据集的规模和质量