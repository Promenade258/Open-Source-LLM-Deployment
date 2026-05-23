# Qwen-OpenAI 兼容API服务

### 代码描述
基于[huihui-ai/Huihui-Qwen3.5-9B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated)实现的 OpenAI 标准 Chat Completions API 兼容服务，支持流式/非流式响应、API Key 鉴权、完整的请求参数兼容，可直接对接各类支持 OpenAI API 的客户端（如 ChatGPT 客户端、LangChain、OpenAI SDK 等）。

## 核心特性
1. **OpenAI 标准兼容**：完全对齐 `/v1/chat/completions` 接口规范，支持 `messages`/`stream`/`max_tokens`/`temperature` 等核心参数
2. **流式/非流式双响应**：支持 `stream=true` 流式输出（SSE 格式）和常规 JSON 响应，适配不同客户端场景
3. **严格的鉴权机制**：支持 API Key 验证（兼容大小写请求头、Bearer 前缀校验）
4. **模型优化配置**：基于 BitsAndBytes 4bit 量化加载模型，降低显存占用，支持自动设备映射
5. **完善的异常处理**：覆盖参数校验、鉴权失败、模型生成异常等场景，返回标准 HTTP 状态码
6. **日志与监控**：完整的请求日志、错误日志输出，便于问题排查

## 快速使用
### 环境依赖
```bash
pip install transformers torch fastapi uvicorn bitsandbytes
```

## 画饼
此版本仅解决了将大模型接入openclaw中时模型不回复，但是控制台中访问是成功的问题。未来我将着重研究发生这个问题的底层原因，并重新写代码。
