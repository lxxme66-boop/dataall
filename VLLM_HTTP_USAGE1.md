# vLLM HTTP服务使用指南

本文档说明如何使用分离式的vLLM服务架构，将模型加载和文件处理分开运行。

## 架构优势

1. **资源隔离**：模型服务和文件处理分开，便于资源管理
2. **服务复用**：一个vLLM服务可以供多个处理进程使用
3. **灵活部署**：可以在不同机器上部署模型服务和处理服务
4. **稳定性提升**：模型服务崩溃不会影响文件处理进程

## 使用步骤

### 步骤1：启动vLLM服务器（终端1）

```bash
# 基本启动
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --trust-remote-code \
  --served-model-name qwen-vllm

# 自定义参数启动
python start_vllm_server.py \
    --model-path /mnt/workspace/models/Qwen/QwQ-32B/ \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --max-model-len 32768
```

启动后会看到类似输出：
```
============================================================
启动vLLM服务器
模型路径: /mnt/workspace/models/Qwen/QwQ-32B/
服务地址: http://0.0.0.0:8000
GPU内存利用率: 0.95
张量并行大小: 1
最大模型长度: 32768
============================================================
```

### 步骤2：运行文件处理（终端2）

在另一个终端中运行文件处理：

```bash
# 使用vLLM HTTP配置文件
python run_semiconductor_qa.py \
    --input-dir data/texts \
    --output-dir data/qa_results \
    --config config_vllm_http.json
```

或者通过环境变量配置：

```bash
# 设置vLLM服务器地址
export VLLM_SERVER_URL=http://localhost:8000/v1

# 运行处理
python run_semiconductor_qa.py \
    --input-dir data/texts \
    --output-dir data/qa_results
```

## 配置说明

### vLLM HTTP配置（config_vllm_http.json）

```json
{
  "api": {
    "use_local_models": true,
    "use_vllm_http": true,
    "vllm_server_url": "http://localhost:8000/v1",
    "default_backend": "vllm_http"
  },
  "models": {
    "local_models": {
      "vllm_http": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "dummy-key",
        "model_name": "qwen-vllm",
        "temperature": 0.7,
        "max_tokens": 2048
      }
    }
  }
}
```

### 环境变量配置

```bash
# vLLM服务器地址
export VLLM_SERVER_URL=http://localhost:8000/v1

# API密钥（如果需要）
export VLLM_API_KEY=your-api-key
```

## 测试连接

测试vLLM HTTP客户端连接：

```bash
python -m LocalModels.vllm_http_client
```

成功连接会显示：
```
成功连接到vLLM服务器
Question: 什么是半导体？
Answer: [模型回答内容]
```

## 故障排除

### 1. 连接失败

如果看到"无法连接到vLLM服务器"错误：

- 检查vLLM服务是否已启动
- 确认端口号是否正确（默认8000）
- 检查防火墙设置

### 2. 模型加载失败

如果vLLM服务启动失败：

- 检查模型路径是否正确
- 确认GPU内存是否充足
- 查看GPU是否被其他进程占用

### 3. 生成超时

如果生成请求超时：

- 增加timeout配置值
- 减少max_tokens参数
- 检查网络连接

## 高级配置

### 1. 多GPU部署

```bash
# 使用4个GPU进行张量并行
python start_vllm_server.py --tensor-parallel-size 4
```

### 2. 远程部署

如果vLLM服务部署在远程服务器：

```json
{
  "models": {
    "local_models": {
      "vllm_http": {
        "base_url": "http://remote-server:8000/v1"
      }
    }
  }
}
```

### 3. 负载均衡

可以启动多个vLLM服务实例，通过nginx等进行负载均衡。

## 性能优化建议

1. **批处理大小**：根据GPU内存调整batch_size
2. **并发请求**：适当控制并发请求数，避免服务过载
3. **缓存策略**：对常见问题启用缓存，减少重复计算
4. **模型量化**：考虑使用量化模型减少内存占用

## 监控和日志

vLLM服务会输出详细的性能指标：

- 吞吐量（tokens/s）
- 延迟（ms）
- GPU利用率
- 内存使用情况

建议使用监控工具（如Prometheus）收集这些指标。