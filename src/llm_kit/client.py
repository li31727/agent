"""
统一 LLM 客户端
================

整合 Day 1-5 的所有踩坑经验:
  - 思考模式开关(Day 3,extra_body.thinking)
  - reasoning_content 续写适配(Day 4 自己 debug 的坑)
  - 双协议(暂时只做 OpenAI 兼容,Anthropic 后续加)
  - 统一返回结构

📚 设计哲学:
  让业务代码只关心"问什么"和"答什么",不关心:
  - thinking 怎么传(extra_body 还是顶层?)
  - 错误怎么续写(reasoning_content 必须带回?)
  - tool_calls 怎么解析(JSON 字符串还是 dict?)
"""

from typing import Any, Literal

from openai import OpenAI

ThinkingMode = Literal["auto", "enabled", "disabled"]


class LLMClient:
    """OpenAI 兼容协议的 LLM 客户端

    用法:
        client = LLMClient(
            api_key="...",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            thinking="disabled",
        )
        response = client.chat([{"role": "user", "content": "你好"}])
        print(response["text"])
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
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.thinking: ThinkingMode = thinking

    # ============================================================
    # 内部:构造 thinking 相关参数
    # ============================================================

    def _build_thinking_kwargs(self) -> dict:
        """根据 thinking 模式构造请求额外参数

        - auto:不传参,用模型/网关默认
        - enabled / disabled:通过 extra_body 透传
        """
        if self.thinking == "auto":
            return {}
        return {"extra_body": {"thinking": {"type": self.thinking}}}

    # ============================================================
    # 主接口:chat
    # ============================================================

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        tools: list | None = None,
        response_format: dict | None = None,
        **extra_kwargs,
    ) -> dict:
        """统一 chat 接口,返回标准化响应

        返回 dict 结构:
        {
            "text": str,              # message.content
            "thinking": str,           # message.reasoning_content
            "tool_calls": list,        # message.tool_calls(原始对象,保留 .id 等)
            "finish_reason": str,
            "tokens": {"input": int, "output": int},
            "raw": ChatCompletion,     # 原始响应,需要时可访问
        }
        """
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

        response = self._client.chat.completions.create(**kwargs)
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

    # ============================================================
    # 续写支持:把上一次响应转成 assistant message
    # ============================================================

    def make_assistant_message(self, response: dict) -> dict:
        """从 chat() 响应构造 assistant message,用于多轮对话续写

        ⭐ 关键(Day 4 踩坑):思考模式下必须把 reasoning_content 原样带回,
        否则 API 会返回 400(尤其在 tool_calls 续写场景)
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

        # 思考型模型续写必备 — Day 4 自己 debug 出来的坑
        if response["thinking"]:
            msg["reasoning_content"] = response["thinking"]

        return msg
