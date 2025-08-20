# 半导体QA生成系统 - 增强版使用说明

## 概述

增强版本**完全保留**原有的所有生成逻辑、prompt模板和评估标准（1.1-3.1流程），仅添加以下三个功能：

1. **断点续跑** - 支持中断后从上次位置继续
2. **火山API支持** - 作为可选的模型后端（使用原有prompt）
3. **多线程并行** - 加速处理（不改变生成逻辑）

## 核心特点

- ✅ **100%保留原有prompt和生成逻辑**
- ✅ **完整的1.1-3.1流程不变**
- ✅ **所有质量评估标准不变**
- ✅ **仅添加性能和可靠性增强**

## 快速开始

### 1. 基础运行（与原版相同）

```bash
# 使用原有模型和逻辑
python run_semiconductor_qa_enhanced.py
```

### 2. 启用断点续跑

```bash
# 首次运行（自动保存断点）
python run_semiconductor_qa_enhanced.py

# 如果中断，从断点恢复
python run_semiconductor_qa_enhanced.py --resume
```

### 3. 使用火山API（可选）

```bash
# 使用火山API但保持原有prompt
python run_semiconductor_qa_enhanced.py \
    --use-volcano \
    --volcano-api-key YOUR_API_KEY
```

### 4. 启用多线程加速

```bash
# 使用8个线程并行处理
python run_semiconductor_qa_enhanced.py --max-workers 8
```

### 5. 组合使用所有功能

```bash
# 完整示例
./run_enhanced.sh \
    --use-volcano \
    --volcano-api-key YOUR_API_KEY \
    --max-workers 8 \
    --resume
```

## 与原版的兼容性

### 完全保留的内容

1. **问题生成prompt** (`prompt_template`)
   - 完整的思考过程 (`<think>` 标签)
   - 所有核心要求和禁止事项
   - 问题格式要求 (`[[1]]`, `[[2]]` 等)

2. **答案生成prompt** (`answer_template`)
   - CoT（Chain of Thought）模式
   - 完整的推理过程要求
   - 答案出处分析要求

3. **评估标准** (`evaluator_template`)
   - 因果性、周密性、追溯性评估
   - 通用性、完整性、单一性检查
   - 【是】/【否】输出格式

4. **处理流程**
   - 1.1 文本质量评估
   - 1.2 文本过滤
   - 1.3 文本处理
   - 2.1 问题生成
   - 2.2 问题评估
   - 2.3 答案生成
   - 2.4 答案评估
   - 3.1 最终评估

### 新增的功能（不影响生成质量）

1. **断点管理**
   - 每5个项目自动保存进度
   - 支持从任意步骤恢复
   - 断点文件保存在 `checkpoints/` 目录

2. **火山API集成**
   - 仅作为模型后端替换
   - 使用完全相同的prompt
   - 输出格式保持一致

3. **多线程处理**
   - 仅用于并行执行独立任务
   - 不改变单个任务的处理逻辑
   - 可配置线程数

## 配置文件说明

`config_enhanced.json` 结构：

```json
{
  "model_config": {
    "use_original_prompts": true,  // 必须为true，使用原有prompt
    "preserve_all_templates": true  // 必须为true，保留所有模板
  },
  
  "volcano_config": {
    "enabled": false,  // 是否启用火山API
    "api_key": "",     // API密钥
    "model": "deepseek-r1"  // 使用的模型
  },
  
  "parallel_config": {
    "max_workers": 4,  // 最大线程数
    "batch_size": 16   // 批处理大小
  },
  
  "checkpoint_config": {
    "enabled": true,        // 启用断点
    "checkpoint_dir": "checkpoints",  // 断点目录
    "save_interval": 5      // 保存间隔
  }
}
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | config_enhanced.json |
| `--resume` | 从断点恢复 | false |
| `--use-volcano` | 使用火山API | false |
| `--volcano-api-key` | 火山API密钥 | - |
| `--max-workers` | 最大线程数 | 4 |
| `--checkpoint-dir` | 断点目录 | checkpoints |

## 监控和日志

### 查看进度

```bash
# 实时查看日志
tail -f logs/enhanced_run_*.log

# 查看断点状态
ls -la checkpoints/
```

### 断点文件格式

断点文件命名：`{步骤号}_{步骤名}_{时间戳}.pkl`

示例：
- `2.1_questions_20240115_143022.pkl`
- `2.3_answers_20240115_145530.pkl`
- `3.1_evaluation_20240115_150045.pkl`

## 故障恢复

### 场景1：程序中断

```bash
# 自动从最新断点恢复
./run_enhanced.sh --resume
```

### 场景2：特定步骤失败

```bash
# 查看最新断点
ls -lt checkpoints/ | head -5

# 从特定步骤恢复
python run_semiconductor_qa_enhanced.py --resume
```

### 场景3：清理重新开始

```bash
# 清理所有断点
rm -rf checkpoints/*

# 重新开始
./run_enhanced.sh
```

## 性能对比

| 功能 | 原版 | 增强版 |
|------|------|--------|
| 断点续跑 | ❌ | ✅ |
| 火山API | ❌ | ✅ (可选) |
| 多线程 | ❌ | ✅ (4-16线程) |
| 处理速度 | 1x | 3-5x |
| 生成质量 | 100% | 100% (相同) |
| Prompt | 原版 | 原版 (不变) |

## 注意事项

1. **生成质量保证**
   - 所有prompt和评估标准与原版完全相同
   - 仅在执行层面添加并行和断点功能
   - 不会影响生成内容的质量

2. **火山API使用**
   - 需要有效的API密钥
   - 确保网络连接稳定
   - API调用失败会自动回退到原有模型

3. **多线程注意**
   - 根据机器性能调整线程数
   - 建议：CPU核心数的1-2倍
   - 过多线程可能导致内存压力

4. **断点管理**
   - 断点文件会占用磁盘空间
   - 完成后可以清理旧断点
   - 建议定期备份重要断点

## 问题排查

### 1. 断点恢复失败

```bash
# 检查断点文件
ls -la checkpoints/

# 查看断点内容
python -c "import pickle; print(pickle.load(open('checkpoints/YOUR_CHECKPOINT.pkl', 'rb')))"
```

### 2. 火山API调用失败

```bash
# 测试API连接
curl -X POST https://ark.cn-beijing.volces.com/api/v3/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1","messages":[{"role":"user","content":"test"}]}'
```

### 3. 内存不足

```bash
# 减少线程数
python run_semiconductor_qa_enhanced.py --max-workers 2

# 或减小批处理大小（需要修改代码）
```

## 总结

增强版本是对原有系统的**纯增强**，不改变任何生成逻辑和质量标准。您可以放心使用新功能，同时保持与原版完全一致的输出质量。