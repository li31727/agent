"""
Week 1 - Day 1 - Step 2: 异步调用 LLM
=====================================

🎯 学习目标:
  1. 理解 sync vs async 的区别(对你 5 年后端来说应该秒懂)
  2. 用 asyncio.gather 并发调用多个 LLM,对比响应速度
  3. 这是生产中"模型路由"的基础(同时打 Claude/GPT/DeepSeek 取最快)

▶️ 运行方式(在仓库根目录):
   uv run week-01/day-01-llm-basics/02_async.py

📝 课后思考:
  - 串行 vs 并发,实际节省了多少时间?
  - 5 个模型并发,延迟会接近 max(5个) 还是 sum(5个)?
  - 为什么生产中常见 "asyncio.gather + asyncio.wait_for(timeout=Xs)"?
"""

import asyncio
import time

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from config import settings
from utils import build_thinking_kwargs, extract_text


async def call_anthropic_async(prompt: str) -> str:
    client = AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )
    thinking_kwargs = build_thinking_kwargs(settings.anthropic_thinking)

    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        **thinking_kwargs,
    )
    # 兼容 thinking 模型(见 utils.py 的解释)
    return extract_text(response)


async def call_openai_async(prompt: str) -> str:
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    response = await client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def main():
    prompt = "用一句话解释 RAG"

    # ======== 串行调用(浪费时间)========
    print("📌 串行调用(一个一个等)")
    start = time.perf_counter()
    r1 = await call_anthropic_async(prompt)
    r2 = await call_openai_async(prompt)
    serial_time = time.perf_counter() - start
    print(f"  ⏱  耗时: {serial_time:.2f}s\n")

    # ======== 并发调用(同时进行,典型场景)========
    print("⚡ 并发调用(asyncio.gather)")
    start = time.perf_counter()
    r1, r2 = await asyncio.gather(
        call_anthropic_async(prompt),
        call_openai_async(prompt),
    )
    parallel_time = time.perf_counter() - start
    print(f"  ⏱  耗时: {parallel_time:.2f}s")
    print(f"  💡 加速比: {serial_time / parallel_time:.2f}x\n")

    print(f"🟪 Anthropic [{settings.anthropic_model}, thinking={settings.anthropic_thinking}]:")
    print(f"  {r1}")
    print(f"🟩 OpenAI    [{settings.openai_model}]:")
    print(f"  {r2}")


if __name__ == "__main__":
    asyncio.run(main())
