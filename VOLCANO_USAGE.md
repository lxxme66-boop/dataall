# 火山API使用说明

## 优化版本特性
- 多线程并行处理
- 火山API支持 (deepseek-r1)
- 断点续跑功能

## 使用方法

### 1. 使用火山API运行
```bash
python run_semiconductor_qa_optimized.py \
    --config config_volcano_api.json \
    --use-volcano-api \
    --volcano-api-key YOUR_KEY \
    --batch-size 64 \
    --max-workers 16
```

### 2. 断点续跑
```bash
python run_semiconductor_qa_optimized.py \
    --config config_volcano_api.json \
    --use-volcano-api \
    --resume
```
