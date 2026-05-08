"""
异步版 ReActAgent — 工具并行执行
==================================

📚 与同步版的核心差异:
  ┌────────────────────────────────────────────────┐
  │ 同步 ReActAgent:                                │
  │   for tc in tool_calls:                        │
  │       result = tool_executor(name, args)       │
  │       # ↑ 串行执行,N 个工具 = N × 单工具延迟    │
  │                                                │
  │ 异步 AsyncReActAgent:                           │
  │   results = await asyncio.gather(*[             │
  │       tool_executor_async(name, args)          │
  │       for ...                                  │
  │   ])                                           │
  │   # ↑ 并发执行,N 个工具 = max(N 个延迟)         │
  └────────────────────────────────────────────────┘

📚 加速比预期:
  Day 4 任务(11 工具调用):同步 21.9s → 异步预期 ~5s(假设单工具 0.5-1s)

📚 何时用?
  - 工具是真实 API(网络 IO 慢)→ 必须用
  - 工具是本地纯计算 → 同步版就够
"""

import asyncio
import json
from typing import Awaitable, Callable

from .async_client import AsyncLLMClient
from .exceptions import AgentMaxIterationsError


class AsyncReActAgent:
    """异步 ReAct Agent — 工具并发执行

    用法:
        async def my_tool(name: str, args: dict) -> str:
            await asyncio.sleep(0.5)  # 模拟 API 延迟
            return json.dumps({"result": "..."})

        agent = AsyncReActAgent(
            client=async_client,
            tools=TOOLS,
            tool_executor=my_tool,  # ⭐ async 函数
        )
        result = await agent.run("用户问题")
    """

    def __init__(
        self,
        client: AsyncLLMClient,
        tools: list,
        tool_executor: Callable[[str, dict], Awaitable[str]],  # ⭐ async!
        system_prompt: str = "你是一个有用的助手。",
        max_iterations: int = 10,
    ):
        self.client = client
        self.tools = tools
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    async def run(
        self,
        user_message: str,
        verbose: bool = False,
        extra_messages: list[dict] | None = None,
    ) -> dict:
        """执行异步 ReAct loop

        关键差异:每轮的多个 tool_calls 用 asyncio.gather 并发执行

        Returns:
            与 ReActAgent.run() 相同的字典结构
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_message})

        total_tokens = 0
        tool_call_history: list[dict] = []

        for iteration in range(1, self.max_iterations + 1):
            response = await self.client.chat(
                messages=messages,
                tools=self.tools,
                max_tokens=1024,
                temperature=0,
            )
            total_tokens += (
                response["tokens"]["input"] + response["tokens"]["output"]
            )

            if verbose:
                print(
                    f"🔄 第 {iteration} 轮: finish={response['finish_reason']}, "
                    f"tokens={response['tokens']}"
                )

            # ───── 终止条件 ─────
            if not response["tool_calls"]:
                return {
                    "answer": response["text"],
                    "iterations": iteration,
                    "total_tokens": total_tokens,
                    "tool_calls": tool_call_history,
                    "messages": messages,
                }

            # 加入 assistant message
            messages.append(self.client.make_assistant_message(response))

            # ⭐ 关键:asyncio.gather 并发执行所有工具
            tool_calls = response["tool_calls"]
            args_list = [
                json.loads(tc.function.arguments) for tc in tool_calls
            ]

            if verbose:
                for tc, args in zip(tool_calls, args_list):
                    args_str = json.dumps(args, ensure_ascii=False)
                    print(f"   🔧 [并发] {tc.function.name}({args_str})")

            # 并发执行所有工具(关键的一行!)
            results = await asyncio.gather(
                *[
                    self.tool_executor(tc.function.name, args)
                    for tc, args in zip(tool_calls, args_list)
                ]
            )

            # 收集结果
            for tc, args, result in zip(tool_calls, args_list, results):
                tool_call_history.append(
                    {
                        "iteration": iteration,
                        "name": tc.function.name,
                        "args": args,
                        "result_preview": result[:120],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        # 超过最大轮数
        raise AgentMaxIterationsError(
            iterations=self.max_iterations,
            last_response=response,
        )
