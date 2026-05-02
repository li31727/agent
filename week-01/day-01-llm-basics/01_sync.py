"""
Week 1 - Day 1 - Step 1: 同步调用 LLM
=====================================

🎯 学习目标:
  1. 跑通 Anthropic + OpenAI 两家 SDK 的最基本调用
  2. 理解请求/响应的核心字段(model / messages / max_tokens / usage)
  3. 用 config.py 统一管理配置 — 永远不要 hardcode!
  4. ⭐ 掌握思考型模型的 thinking 开关(enabled / disabled / auto)

▶️ 运行方式(在仓库根目录):
   uv run week-01/day-01-llm-basics/01_sync.py

🔬 实验建议(对比效果):
   1. 在 .env 设 ANTHROPIC_THINKING=enabled,跑一次,记录耗时 + tokens
   2. 改成 ANTHROPIC_THINKING=disabled,再跑一次
   3. 对比:thinking 关掉后,速度快多少?token 省多少?答案质量差多少?

📝 课后思考(写到 Obsidian Day 1 笔记):
  - 思考型模型 vs 普通模型的成本差?token 多了多少?
  - 什么场景值得开 thinking?(数学/代码/规划)
  - 什么场景该关 thinking?(闲聊/翻译/格式转换)
"""

from anthropic import Anthropic
from openai import OpenAI

from config import settings
from utils import build_thinking_kwargs, extract_text, extract_thinking


def call_anthropic(prompt: str) -> dict:
    """调用 Anthropic 协议 LLM,支持 thinking 开关"""
    client = Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )

    # 根据 .env 的 ANTHROPIC_THINKING 配置自动构造参数
    thinking_kwargs = build_thinking_kwargs(settings.anthropic_thinking)

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt},
        ],
        **thinking_kwargs,  # 透传 thinking 配置(auto 时为空,不影响)
    )

    return {
        "text": extract_text(response),
        "thinking": extract_thinking(response),
        "tokens": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    }


def call_openai(prompt: str) -> dict:
    """调用 OpenAI 协议 LLM"""
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return {
        "text": response.choices[0].message.content,
        "thinking": getattr(response.choices[0].message, "reasoning_content", "") or "",  # ⭐ 加这一行
        "tokens": {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    }


if __name__ == "__main__":
    prompt = "用 50 个汉字解释什么是 ReAct(推理+行动)模式"

    print("=" * 60)
    print(f"🟪 Anthropic [{settings.anthropic_model}] thinking={settings.anthropic_thinking}")
    print("=" * 60)
    result = call_anthropic(prompt)

    if result["thinking"]:
        print("💭 [模型 thinking 过程]")
        print(result["thinking"])
        print("─" * 40)

    print("📤 [最终答案]")
    print(result["text"])
    print(
        f"\n[tokens] input={result['tokens']['input']}, "
        f"output={result['tokens']['output']}"
    )

    print()
    print("=" * 60)
    print(f"🟩 OpenAI [{settings.openai_model}]")
    print("=" * 60)
    result = call_openai(prompt)

    if result["thinking"]:
        print("💭 [模型 thinking 过程]")
        print(result["thinking"])
        print("─" * 40)

    print("📤 [最终答案]")
    print(result["text"])
    print(
        f"\n[tokens] input={result['tokens']['input']}, "
        f"output={result['tokens']['output']}"
    )
