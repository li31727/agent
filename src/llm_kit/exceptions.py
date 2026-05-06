"""
llm_kit 异常体系
=================

设计原则:
- 所有 SDK 抛出的异常都继承 LLMError(便于统一捕获)
- 区分"网络/API 问题"和"业务逻辑问题"
- 异常携带可调试信息(retry 次数 / 上一次响应等)
"""


class LLMError(Exception):
    """SDK 所有异常的基类"""


class LLMTimeoutError(LLMError):
    """LLM API 调用超时"""


class LLMRateLimitError(LLMError):
    """LLM API 速率限制(429)"""


class LLMServerError(LLMError):
    """LLM API 5xx 错误"""


class AgentMaxIterationsError(LLMError):
    """Agent 超过最大循环次数(可能死循环)"""

    def __init__(self, iterations: int, last_response: dict | None = None):
        self.iterations = iterations
        self.last_response = last_response
        super().__init__(f"Agent 超过 {iterations} 轮仍未收敛")


class StructuredExtractionError(LLMError):
    """Pydantic 校验失败 + 重试耗尽"""

    def __init__(self, retries: int, last_error: Exception | None = None):
        self.retries = retries
        self.last_error = last_error
        super().__init__(f"结构化提取失败({retries} 次重试)")
