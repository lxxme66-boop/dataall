# 半导体显示技术QA数据集生成系统

## 项目概述

本项目是一个专门用于生成半导体显示技术领域高质量问答（QA）数据集的自动化系统。系统采用分离式架构，将大语言模型服务（vLLM）与业务处理逻辑分离，支持多种问题类型生成、答案生成和数据增强等功能。

## 系统特点

- **分离式架构**：模型服务和业务逻辑分离，便于独立扩展和部署
- **多种问题类型**：支持事实型、比较型、推理型和开放型问题的生成
- **智能答案生成**：使用Chain of Thought（COT）方式生成高质量答案
- **质量控制**：多级质量评估确保生成数据的准确性和专业性
- **批量处理**：支持大规模文本的并行处理
- **数据增强**：对生成的QA进行进一步优化和扩充

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    系统架构总览                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐        ┌─────────────────────┐   │
│  │   vLLM Server   │ HTTP   │   QA Generator      │   │
│  │                 │◄──────►│                     │   │
│  │  - Model Load   │ :8000  │  - Text Processing  │   │
│  │  - GPU Inference│        │  - Question Gen     │   │
│  │  - OpenAI API   │        │  - Answer Gen       │   │
│  └─────────────────┘        └─────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 文件结构

```
/workspace/
├── README.md                    # 本文档
├── VLLM_HTTP_USAGE.md          # vLLM HTTP服务使用指南
├── requirements.txt            # 项目依赖
├── config.json                 # 主配置文件
├── config_vllm_http.json       # vLLM HTTP配置文件
│
├── start_vllm_server.py        # vLLM服务启动脚本
├── run_semiconductor_qa.py     # 主程序入口
├── semiconductor_qa_generator.py # QA生成核心模块
│
├── text_processor.py           # 文本预处理模块
├── enhanced_file_processor.py  # 增强文件处理模块
├── enhanced_document_processor.py # 文档处理模块
├── argument_data.py            # 数据增强模块
├── clean_text_data.py          # 数据清洗模块
│
├── LocalModels/                # 本地模型客户端
│   ├── vllm_http_client.py     # vLLM HTTP客户端
│   ├── local_model_manager.py  # 模型管理器
│   └── ...
│
├── TextGeneration/             # 文本生成相关模块
├── TextQA/                     # QA处理相关模块
│
└── data/                       # 数据目录
    ├── texts/                  # 输入文本目录
    └── output/                 # 输出目录
        ├── chunks/             # 文本预处理结果
        ├── qa_original/        # 原始QA生成结果
        └── qa_results/         # 最终QA数据集
```

## 处理流程

系统采用三阶段处理流程：

### 第一阶段：文本预处理 + 质量评估
1. **文本分块**：将长文本分割成适合处理的块
2. **AI文本优化**：使用AI模型理解和优化文本内容
3. **质量评估**：评估文本是否适合生成QA

### 第二阶段：QA生成（4个步骤）
1. **分类问题生成**：按比例生成不同类型的问题
   - 事实型（15%）：获取指标、数值、性能参数等
   - 比较型（15%）：比较不同材料、结构或方案
   - 推理型（50%）：机制原理解释，探究原因
   - 开放型（20%）：优化建议，改进方法
2. **格式转换**：将问题列表转换为标准QA格式
3. **质量评估**：对生成的问题进行质量打分
4. **答案生成**：使用COT方式为高质量问题生成答案

### 第三阶段：数据增强
- 对生成的QA进行进一步优化
- 扩充问题和答案的多样性
- 提升整体数据集质量

## 快速开始

### 环境准备

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备模型：
确保已下载所需的大语言模型（如Qwen-32B）到指定路径。

### 一键运行

最简单的运行方式：

```bash
# 1. 启动vLLM服务器（终端1）
CUDA_VISIBLE_DEVICES=6,7 
python start_vllm_server.py

# 2. 运行QA生成（终端2）
python run_semiconductor_qa.py \
    --input-dir data/texts \
    --output-dir data/output \
    --config config_vllm_http.json
```

### 分步运行

如需更精细的控制，可以分步执行：

#### 步骤1：启动vLLM服务器

```bash
python start_vllm_server.py \
    --model-path /path/to/your/model \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2
```

#### 步骤2：运行QA生成

```bash
python run_semiconductor_qa.py \
    --input-dir data/texts \
    --output-dir data/output \
    --config config_vllm_http.json \
    --model-name qwq_32 \
    --batch-size 2 \
    --quality-threshold 0.7
```


#### 步骤2.5：答案出处验证




#### 步骤3：评估
3.0 增强问答(选做)
python argument_data.py \
    --input-file /mnt/workspace/LLM/ldd/sft/data/output/qa_results/qa_generated.json \
    --output-file overwrite/v2_rewrite.json \
    --quality-report

3.1 将问答对转为xlsx表格
python qa_data_format.py

文件输入：
/mnt/workspace/LLM/ldd/sft/reslut/v2_rewrite.json
/mnt/workspace/LLM/ldd/sft/data/output/qa_results/qa_generated.json

输出：
/mnt/workspace/LLM/ldd/sft/input/Qwen325评估数据输入-无增强.xlsx

3.2 评估质量
python cot数据质量评估.py

输入
input
输出
result


3.3 选取问答对
# 提取高质量问答对
python filter_qa.py --input qa_quality_high_medium_simple.xlsx --output high_quality.xlsx --quality high

# 提取中+高质量问答对
python filter_qa.py --input qa_quality_high_medium_simple.xlsx --output medium_high_quality.xlsx --quality medium_high
## 配置说明

### 主要配置参数（config_vllm_http.json）

```json
{
  "api": {
    "use_vllm_http": true,
    "vllm_server_url": "http://localhost:8000/v1",
    "default_backend": "vllm_http"
  },
  "models": {
    "local_models": {
      "vllm_http": {
        "base_url": "http://localhost:8000/v1",
        "model_name": "qwen-vllm",
        "temperature": 0.7,
        "max_tokens": 2048
      }
    }
  },
  "processing": {
    "batch_size": 32,
    "quality_threshold": 0.7
  }
}
```

### 命令行参数

- `--input-dir`: 输入文本目录
- `--output-dir`: 输出结果目录
- `--config`: 配置文件路径
- `--model-name`: 使用的模型名称
- `--batch-size`: 批处理大小
- `--quality-threshold`: 质量阈值（0-1）
- `--gpu-devices`: GPU设备ID（如"0,1"）

## 输出说明

系统会在输出目录生成以下结构：

```
output/
├── chunks/                      # 文本预处理结果
│   ├── ai_processed_texts.json  # AI处理后的文本
│   ├── quality_judged_texts.json # 质量评估结果
│   └── qualified_texts.json     # 合格文本列表
│
├── qa_original/                 # 原始QA生成结果
│   ├── classified_questions.json # 分类问题
│   ├── converted_questions.json  # 格式化问题
│   ├── evaluated_qa_data.json   # 评估后的QA
│   ├── qa_with_context.json     # 带上下文的QA
│   └── qa_with_answers.json     # 带答案的QA
│
├── qa_results/                  # 最终结果
│   ├── qa_generated.json        # 生成的完整QA数据集
│   └── final_qa_dataset.json    # 增强后的最终数据集
│
└── pipeline_stats.json          # 处理统计信息
```

## 故障排查

### 1. vLLM服务连接失败
- 检查vLLM服务是否已启动
- 确认端口号是否正确（默认8000）
- 检查防火墙设置

### 2. GPU内存不足
- 减小batch_size参数
- 降低gpu_memory_utilization
- 使用更多GPU进行张量并行

### 3. 生成质量问题
- 调整quality_threshold参数
- 检查输入文本质量
- 优化prompt模板

## 高级功能

### 自定义问题类型

可以在`run_semiconductor_qa.py`的`QUESTION_TYPES`中修改问题类型定义：

```python
QUESTION_TYPES = {
    "factual": {
        "ratio": 0.15,
        "description": "事实型问题",
        "examples": [...]
    },
    # 添加更多类型...
}
```

### 自定义答案生成模板

在`semiconductor_qa_generator.py`的`generate_answers`方法中修改答案模板。

## 性能优化建议

1. **批处理优化**：根据GPU内存调整batch_size
2. **并行处理**：使用多个vLLM实例进行负载均衡
3. **缓存策略**：对常见问题启用缓存
4. **模型量化**：考虑使用量化模型减少内存占用

## 贡献指南

欢迎提交Issue和Pull Request来改进本项目。

## 许可证

本项目采用MIT许可证。

基于我对代码的分析，以下是完整的文档加载流程：

1. 支持的文件格式 ✅
系统支持多种文件格式的处理：

TXT文件: 通过text_processor.py和TextGeneration/Datageneration.py处理
PDF文件: 通过pdf_processor_main.py和MultiModal/pdf_processor.py处理
Markdown文件: 通过enhanced_document_processor.py处理
JSON文件: 通过enhanced_document_processor.py处理
2. 文档加载流程（以run_semiconductor_qa.py为例）
第一阶段：文本预处理 + 质量评估
文件扫描（第453-463行）：

扫描input_dir目录下的所有.txt文件
对每个文件调用parse_txt函数进行分块
文本分块（parse_txt函数）：

读取文件内容
按照配置的chunk_size（默认1000字符）进行分块
为每个块创建处理任务
AI文本处理（第466-487行）：

对每个文本块调用input_text_process进行AI增强处理
批量处理以提高效率
结果保存到chunks/ai_processed_texts.json
文本质量评估（第499-543行）：

使用judge_paper_quality方法评估文本质量
根据评分筛选合格的文本
结果保存到chunks/quality_judged_texts.json和chunks/qualified_texts.json
第二阶段：QA生成
问题生成（第556-565行）：

加载合格的文本
使用generate_classified_questions生成分类问题
结果保存到qa_original/classified_questions.json
答案生成和质量评估：

生成答案
评估QA质量
结果保存到qa_original/目录下的多个文件
第三阶段：数据增强
数据增强（第655-678行）：
加载原始QA数据
使用ArgumentDataProcessor进行增强
最终结果保存到qa_results/final_qa_dataset.json
3. 数据流向总结
输入文件（TXT/PDF/MD等）
    ↓
文本分块（parse_txt）
    ↓
AI文本处理（input_text_process）
    ↓
保存到 output/chunks/ai_processed_texts.json
    ↓
文本质量评估（judge_paper_quality）
    ↓
保存到 output/chunks/qualified_texts.json
    ↓
QA生成（generate_classified_questions）
    ↓



    graph TD
    A[原始文本] --> B[文本预处理]
    B --> C[分类问题生成<br/>classified_questions.json]
    C --> D[问题格式转换<br/>converted_questions.json]
    D --> E[问题质量评估<br/>evaluated_qa_data.json]
    E --> F{质量筛选<br/>score >= threshold?}
    F -->|是| G[高质量问题<br/>qa_with_context.json]
    F -->|否| H[丢弃低质量问题]
    G --> I[答案生成<br/>qa_with_answers.json]
    I --> J[复制到结果目录<br/>qa_generated.json]
    E --> K[数据增强输入<br/>再次筛选高质量数据]
    K --> L[最终数据集<br/>final_qa_dataset.json]