# 增强版 run_step.py 使用说明

## 新增功能

### 1. 火山API支持
- 支持使用火山引擎API进行文本处理
- 可以替代本地模型，提高处理速度

### 2. 多线程处理
- 支持多线程并发处理，充分利用系统资源
- 可配置最大工作线程数

### 3. 断点恢复
- 支持从中断的地方继续执行
- 自动保存进度，防止数据丢失
- 可以跳过已完成的步骤

## 使用示例

### 基础用法（与之前相同）
```bash
# 执行单个步骤
python run_step.py --step 1.1 --input-dir data/texts --output-dir data/qa_results

# 执行完整流程
python run_step.py --step all --input-dir data/texts --output-dir data/qa_results
```

### 使用火山API
```bash
# 使用火山API执行步骤1.2
python run_step.py --step 1.2 --use-volcano --volcano-api-key YOUR_API_KEY

# 或者设置环境变量
export VOLCANO_API_KEY=YOUR_API_KEY
python run_step.py --step 1.2 --use-volcano

# 执行完整流程并使用火山API
python run_step.py --step all --use-volcano --volcano-api-key YOUR_API_KEY
```

### 多线程处理
```bash
# 指定最大工作线程数
python run_step.py --step 1.2 --max-workers 16

# 禁用多线程（单线程处理）
python run_step.py --step 1.2 --no-multithread

# 使用火山API + 多线程
python run_step.py --step all --use-volcano --max-workers 32
```

### 断点恢复
```bash
# 从断点恢复执行
python run_step.py --step all --resume

# 强制重新开始（忽略断点）
python run_step.py --step all --no-resume

# 查看当前进度状态
python run_step.py --step status --output-dir data/qa_results
```

### 组合使用
```bash
# 完整示例：火山API + 多线程 + 断点恢复
python run_step.py \
    --step all \
    --input-dir data/texts \
    --output-dir data/qa_results \
    --use-volcano \
    --volcano-api-key YOUR_API_KEY \
    --max-workers 16 \
    --resume \
    --batch-size 4 \
    --quality-threshold 0.8
```

## 配置文件

### 使用增强配置文件
```bash
# 使用配置文件（包含所有增强功能的配置）
python run_step.py --step all --config config_enhanced.json
```

### 配置文件示例 (config_enhanced.json)
```json
{
  "api": {
    "use_volcano_api": true,
    "volcano_api_key": "YOUR_API_KEY"
  },
  "processing": {
    "max_workers": 16,
    "use_multithread": true,
    "batch_size": 4
  },
  "resume": {
    "enabled": true,
    "auto_resume": true
  }
}
```

## 步骤说明

| 步骤 | 描述 | 支持功能 |
|------|------|----------|
| 1.1 | 文本预处理和分块 | 断点恢复 |
| 1.2 | AI文本处理 | 火山API、多线程、断点恢复 |
| 1.3 | 文本质量评估 | 断点恢复 |
| 2.1 | 分类问题生成 | 断点恢复 |
| 2.2 | 问题格式转换 | 断点恢复 |
| 2.3 | 问题质量评估 | 断点恢复 |
| 2.4 | 答案生成 | 断点恢复 |
| 3.0 | 数据增强 | 断点恢复 |

## 监控和调试

### 查看进度
```bash
# 查看当前执行状态
python run_step.py --step status --output-dir data/qa_results

# 生成执行报告
python run_step.py --step report --output-dir data/qa_results
```

### 日志文件
- 所有执行日志都会输出到控制台
- 断点信息保存在 `output_dir/step_status.json`
- 每个步骤的输出保存在对应的目录中

## 性能优化建议

1. **火山API**：适合大规模文本处理，减少本地GPU负载
2. **多线程**：CPU核心数的2倍通常是最佳线程数
3. **批处理大小**：根据内存和API限制调整
4. **断点恢复**：长时间运行的任务建议启用

## 故障排除

### 火山API连接失败
- 检查API密钥是否正确
- 确认网络连接正常
- 查看API配额是否充足

### 多线程死锁
- 减少线程数
- 使用 `--no-multithread` 临时禁用

### 断点恢复失败
- 检查 `step_status.json` 文件是否损坏
- 使用 `--no-resume` 强制重新开始

## 注意事项

1. 火山API需要有效的API密钥
2. 多线程处理会增加内存使用
3. 断点文件会占用少量磁盘空间
4. 建议定期备份输出目录