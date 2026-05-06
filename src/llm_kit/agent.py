"""
ReAct Agent — 自动多步工具调用
================================

把 Day 4 那 30 行 ReAct loop 抽象成可复用类。

📚 ReAct loop 核心:
   while not done:
       response = llm.chat(messages, tools)
       if response.tool_calls:
           execute and append tool results
       else:
           return response.text

📚 这就是 LangGraph / Eino 的核心 — 它们只是在此基础上加:
   - 状态机
   - Checkpoint
   - Human-in-the-loop
   - 错误重试
"""

import json
from typing import Callable

from .client import LLMClient
from .exceptions import AgentMaxIterationsError


class ReActAgent:
    """通用 ReAct Agent

    用法:
        agent = ReActAgent(
            client=client,
            tools=OPENAI_TOOLS,
            tool_executor=lambda name, args: execute_tool(name, args),
            system_prompt="你是导购助手。",
        )
        result = agent.run("帮我找 500 元以内的礼物")
        print(result["answer"])
    """

    def __init__(
        self,
        client: LLMClient,
        tools: list,
        tool_executor: Callable[[str, dict], str],
        system_prompt: str = "你是一个有用的助手。",
        max_iterations: int = 10,
    ):
        self.client = client
        self.tools = tools
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    def run(
        self,
        user_message: str,
        verbose: bool = False,
        extra_messages: list[dict] | None = None,
    ) -> dict:
        """执行 ReAct loop

        Args:
            user_message:用户输入
            verbose:是否打印每轮进度
            extra_messages:额外的初始 messages(如多轮对话历史)

        Returns:
            {
                "answer": str,                  # 最终答案
                "iterations": int,              # 实际轮数
                "total_tokens": int,            # 累计 token
                "tool_calls": list[dict],       # 调用历史
                "messages": list[dict],         # 完整 messages 历史(可继续对话)
            }

        Raises:
            AgentMaxIterationsError:超过最大轮数仍未收敛
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
            response = self.client.chat(
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

            # ───── 终止条件:没工具调用 = 最终答案 ─────
            if not response["tool_calls"]:
                return {
                    "answer": response["text"],
                    "iterations": iteration,
                    "total_tokens": total_tokens,
                    "tool_calls": tool_call_history,
                    "messages": messages,
                }

            # ───── 加 assistant message(自动处理 reasoning_content)─────
            messages.append(self.client.make_assistant_message(response))

            # ───── 执行所有工具(串行,生产可改异步并行)─────
            for tc in response["tool_calls"]:
                args = json.loads(tc.function.arguments)
                result = self.tool_executor(tc.function.name, args)

                tool_call_history.append(
                    {
                        "iteration": iteration,
                        "name": tc.function.name,
                        "args": args,
                        "result_preview": result[:120],
                    }
                )

                if verbose:
                    print(
                        f"   🔧 {tc.function.name}({json.dumps(args, ensure_ascii=False)})"
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
