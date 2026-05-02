"""
Week 1 - Day 1 - Step 3: 流式输出(Streaming)
=============================================

🎯 学习目标:
  1. 理解 LLM streaming 的本质(底层是 SSE: Server-Sent Events)
  2. 掌握"首字节时间(TTFT, Time To First Token)"这个体验关键指标
  3. 这是 ChatGPT 那种逐字蹦字效果的实现原理

▶️ 运行方式:
   uv run week-01/day-01-llm-basics/03_stream.py

📝 课后思考:
  - 为什么所有 ChatGPT 类产品都用 streaming?
  - TTFT(首字节)和总耗时,哪个更影响用户体验?
  - Agent 在调用 Tool 时,流式输出还有意义吗?(想想 Anthropic UI 的实现)
"""

import time

from anthropic import Anthropic
from openai import OpenAI

from config import settings


def stream_anthropic(prompt: str):
    client = Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )

    print(f"🟪 Anthropic [{settings.anthropic_model}] 流式输出:\n")
    start = time.perf_counter()
    first_token_time = None

    with client.messages.stream(
        model=settings.anthropic_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            print(text, end="", flush=True)

    total_time = time.perf_counter() - start
    print(
        f"\n\n  📊 TTFT: {first_token_time:.2f}s | 总耗时: {total_time:.2f}s\n"
    )


def stream_openai(prompt: str):
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    print(f"🟩 OpenAI [{settings.openai_model}] 流式输出:\n")
    start = time.perf_counter()
    first_token_time = None

    stream = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            print(delta, end="", flush=True)

    total_time = time.perf_counter() - start
    print(
        f"\n\n  📊 TTFT: {first_token_time:.2f}s | 总耗时: {total_time:.2f}s\n"
    )


if __name__ == "__main__":
    prompt = "用 200 字详细解释什么是 Multi-Agent 协作系统,以及它解决了什么问题"

    print("=" * 60)
    stream_anthropic(prompt)

    print("=" * 60)
    stream_openai(prompt)
