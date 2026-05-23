import os
import logging
import json

os.environ["OMP_NUM_THREADS"] = "8"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio
import uuid
from typing import Optional
from starlette.responses import StreamingResponse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 创建FastAPI实例
app = FastAPI(title="OpenAI Compatible API")

# 允许跨域（增强：覆盖更多客户端场景）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 模型路径
model_dir = "Huihui-Qwen3.5-9B-abliterated"

print("开始加载模型和分词器...")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
).eval()
print("模型加载完成！")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "custom-qwen",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
                "root": "custom-qwen",
                "parent": None
            }
        ]
    }


# 工具函数：生成流式响应数据
async def generate_stream_response(output_text: str, request_id: str, input_tokens: int, output_tokens: int):
    """生成OpenAI标准的流式响应格式"""
    total_tokens = input_tokens + output_tokens
    # 流式响应的单条数据模板
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "custom-qwen",
        "system_fingerprint": "fp_123456",  # 补充客户端依赖的字段
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": output_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens
        }
    }
    # 流式输出格式：data: {JSON}\n\n
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    # 流式结束标记
    yield "data: [DONE]\n\n"


# =============== 兼容流式/非流式响应 ===============
@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    try:
        # ========== 1. 增强API Key验证（兼容大小写、边缘场景） ==========
        auth_header = request.headers.get("Authorization", request.headers.get("authorization"))
        valid_api_key = "你的API Key"
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.error("请求缺少有效Authorization头")
            raise HTTPException(status_code=401, detail="无效的API Key：缺少Bearer前缀")
        api_key = auth_header.split(" ")[1].strip()
        if api_key != valid_api_key:
            logger.error(f"API Key验证失败：收到 {api_key}")
            raise HTTPException(status_code=401, detail="无效的API Key")

        # ========== 2. 解析请求参数（重点处理stream参数） ==========
        request_data = await request.json()
        logger.info(f"收到请求参数：{json.dumps(request_data, ensure_ascii=False)}")

        # 必选参数校验
        messages = request_data.get("messages")
        if not messages or not isinstance(messages, list):
            logger.error("messages参数错误：非空列表")
            raise HTTPException(status_code=400, detail="参数错误：messages必须是非空列表")

        # 提取核心参数（兼容OpenAI标准）
        stream = request_data.get("stream", False)  # 关键：处理流式请求
        max_new_tokens = request_data.get("max_tokens", 2048)  # 降低默认值，避免超时
        temperature = request_data.get("temperature", 0.7)  # 调整默认值更符合OpenAI习惯
        top_p = request_data.get("top_p", 0.9)
        top_k = request_data.get("top_k", 50)
        repetition_penalty = request_data.get("repetition_penalty", 1.05)

        # ========== 3. 构建模型输入 ==========
        history = messages
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_tokens = len(inputs["input_ids"][0])
        request_id = f"chatcmpl-{str(uuid.uuid4())[:10]}"

        # ========== 4. 模型生成（优化超时/显存） ==========
        def generate_response():
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=0.0,
                        repetition_penalty=repetition_penalty,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        timeout=300,  # 增加超时限制（5分钟）
                    )
                torch.cuda.empty_cache()
                return outputs
            except Exception as e:
                logger.error(f"模型生成失败：{str(e)}")
                raise

        # 异步执行生成
        outputs = await asyncio.to_thread(generate_response)
        response_text = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        output_tokens = len(outputs[0]) - input_tokens
        total_tokens = input_tokens + output_tokens

        # ========== 5. 兼容流式/非流式响应 ==========
        if stream:
            # 流式响应：返回StreamingResponse
            logger.info(f"返回流式响应：{request_id} | 生成文本：{response_text[:50]}...")
            return StreamingResponse(
                generate_stream_response(response_text, request_id, input_tokens, output_tokens),
                media_type="text/event-stream"
            )
        else:
            # 非流式响应（curl调用）：返回标准JSON
            logger.info(f"返回非流式响应：{request_id} | 生成文本：{response_text[:50]}...")
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "custom-qwen",
                "system_fingerprint": "fp_123456",  # 补充字段
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop" if output_tokens < max_new_tokens else "length"
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"服务内部错误：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务内部错误：{str(e)}")


# 启动服务
if __name__ == "__main__":
    import uvicorn

    # 增加超时配置、日志输出
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=你的端口,
        workers=1,
        log_level="info"
    )