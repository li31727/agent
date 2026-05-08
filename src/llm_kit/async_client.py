"""
异步版 LLMClient — 支持 await 调用
=====================================

📚 与同步版的关系:
  - 接口 100% 一致(chat / make_assistant_message)
  - 底层用 AsyncOpenAI(openai 官方 SDK 提供)
  - 所有 thinking / reasoning_content 适配逻辑相同

📚 何时用异步?
  - 多 LLM 并发调用(模型路由 / 多 Agent)
  - LLM 调用嵌入异步框架(FastAPI / aiohttp)
  - 真实 API 工具(asyncio.gather 并行)
"""

from typing import Any

from openai import AsyncOpenAI

from .client import ThinkingMode


class AsyncLLMClient:
    """OpenAI 兼容协议的异步 LLM 客户端

    用法:
        client = AsyncLLMClient(api_key="...", model="...")
        response = await client.chat([{"role": "user", "content": "..."}])
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o",
        thinking: ThinkingMode = "auto",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.thinking: ThinkingMode = thinking

    def _build_thinking_kwargs(self) -> dict:
        """与同步版完全一致的 thinking 配置逻辑"""
        if self.thinking == "auto":
            return {}
        return {"extra_body": {"thinking": {"type": self.thinking}}}

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        tools: list | None = None,
        response_format: dict | None = None,
        **extra_kwargs,
    ) -> dict:
        """异步 chat — 接口与 LLMClient.chat 一致,只是用 await"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self._build_thinking_kwargs(),
            **extra_kwargs,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        return {
            "text": msg.content or "",
            "thinking": getattr(msg, "reasoning_content", "") or "",
            "tool_calls": list(msg.tool_calls or []),
            "finish_reason": response.choices[0].finish_reason,
            "tokens": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            },
            "raw": response,
        }

    def make_assistant_message(self, response: dict) -> dict:
        """续写支持 — 与同步版完全一致(纯函数,无 IO)

        ⭐ 思考型模型续写必须把 reasoning_content 原样带回(Day 4 踩坑修复)
        """
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": response["text"],
        }

        if response["tool_calls"]:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response["tool_calls"]
            ]

        if response["thinking"]:
            msg["reasoning_content"] = response["thinking"]

        return msg
