"""
llm_kit — Week 1 Day 6 整合 SDK
=================================

把 Day 1-5 学到的所有"工程化经验 + 踩坑修复"封装成一个干净的包:

📦 主要组件:
  - LLMClient:统一 LLM 调用 + 思考模式开关 + reasoning_content 适配
  - ReActAgent:多步工具调用 Agent(Day 4 30 行 loop 的工程化版本)
  - StructuredExtractor:Pydantic 结构化输出(Day 5,Mode.JSON 兼容国内网关)

📚 用法:
    from llm_kit import LLMClient, ReActAgent, StructuredExtractor

    client = LLMClient(
        api_key="...",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash",
        thinking="disabled",
    )

    # 1. 普通聊天
    response = client.chat([{"role": "user", "content": "你好"}])

    # 2. ReAct Agent
    agent = ReActAgent(client=client, tools=TOOLS, tool_executor=execute_tool)
    result = agent.run("帮我推荐礼物")

    # 3. 结构化输出
    extractor = StructuredExtractor(api_key="...", base_url="...", model="...")
    review = extractor.extract(response_model=Review, messages=[...])
"""

from .agent import ReActAgent
from .async_agent import AsyncReActAgent
from .async_client import AsyncLLMClient
from .client import LLMClient, ThinkingMode
from .exceptions import (
    AgentMaxIterationsError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    StructuredExtractionError,
)
from .structured import StructuredExtractor

__version__ = "0.2.0"  # Day 8 加入异步支持

__all__ = [
    # 同步主类
    "LLMClient",
    "ReActAgent",
    "StructuredExtractor",
    # 异步主类(Day 8 新增)
    "AsyncLLMClient",
    "AsyncReActAgent",
    # 类型
    "ThinkingMode",
    # 异常
    "LLMError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMServerError",
    "AgentMaxIterationsError",
    "StructuredExtractionError",
]
