"""
通用工具函数
=============

🎯 设计目的:
  把跨多个 demo 复用的小逻辑抽出来,保持 demo 代码聚焦在"教学点"上。

📚 知识点 1:为什么需要 extract_text?
-----------------------------------
现代"思考型"LLM(Claude 3.7+ / Claude 4 / DeepSeek-R1 / o1 等)
返回的 response.content 不是单一文本,而是多种 block 的列表:

  response.content = [
      ThinkingBlock(thinking="让我想想..."),   ← 推理过程,无 .text 属性
      TextBlock(text="ReAct 是..."),           ← 用户看的最终答案
      ToolUseBlock(...),                        ← 工具调用(后续 Day 学)
  ]

直接 response.content[0].text 在 thinking 模型上会崩(ThinkingBlock 没 .text)。
生产代码必须**遍历 + 按 type 过滤**,这是面试常考点。

📚 知识点 2:为什么需要 build_thinking_kwargs?
---------------------------------------------
DeepSeek / OpenAI o1 / Claude 4 等"思考型"模型有 thinking 开关:
  - enabled  → 思考(慢,贵,准确)
  - disabled → 不思考(快,便宜,适合简单对话)

通过 extra_body 透传给底层 HTTP 请求,SDK 不会拦截。
参考: https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
"""

from typing import Any


# ============================================================
# 文本提取
# ============================================================


def extract_text(response: Any) -> str:
    """
    从 Anthropic 响应中提取纯文本答案,自动忽略 thinking / tool_use 等其他 block。

    Args:
        response: Anthropic Message 对象(client.messages.create 的返回值)

    Returns:
        所有 TextBlock 的文本拼接(用换行连接)
    """
    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )


def extract_thinking(response: Any) -> str:
    """
    提取模型的 thinking(推理过程)— 调试 / 学习用。

    生产中通常不展示给用户,但开发时打印出来很有助于理解模型在想什么。
    """
    return "\n".join(
        block.thinking
        for block in response.content
        if block.type == "thinking"
    )


# ============================================================
# Thinking 模式开关
# ============================================================


def build_thinking_kwargs(thinking_mode: str) -> dict:
    """
    根据 thinking_mode 构造传给 client.messages.create() 的额外参数。

    用法:
        kwargs = build_thinking_kwargs(settings.anthropic_thinking)
        response = client.messages.create(
            model=...,
            messages=[...],
            **kwargs,  # 自动 unpack 进去
        )

    Args:
        thinking_mode: "auto"(不传参) / "enabled"(开) / "disabled"(关)

    Returns:
        - "auto" 时返回 {} (相当于不传)
        - 其他时返回 {"extra_body": {"thinking": {"type": ...}}}
    """
    if thinking_mode == "auto":
        return {}

    return {
        "extra_body": {
            "thinking": {"type": thinking_mode},
        }
    }
