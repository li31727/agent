"""
结构化输出提取器
==================

基于 Day 5 的 Pydantic + Instructor 经验封装。

⭐ 关键(Day 5 自己 debug):
  Mode.JSON 是国内网关唯一可用配置
  默认 Mode.TOOLS 会撞 "tool_choice not supported"
"""

from typing import TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel

from .exceptions import StructuredExtractionError

T = TypeVar("T", bound=BaseModel)


class StructuredExtractor:
    """Pydantic 结构化输出提取器

    用法:
        from pydantic import BaseModel
        from typing import Literal

        class Review(BaseModel):
            sentiment: Literal["正面", "中性", "负面"]
            rating: int

        extractor = StructuredExtractor(
            api_key="...",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )

        result: Review = extractor.extract(
            response_model=Review,
            messages=[{"role": "user", "content": "..."}],
        )
        print(result.sentiment)  # IDE 自动补全
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o",
        timeout: float = 60.0,
        thinking: str = "auto",
    ):
        # ⭐ Mode.JSON — Day 5 自己 debug 出的国内网关唯一可用配置
        # 默认 Mode.TOOLS 会用 tool_choice="required",DeepSeek 系不支持
        base_client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout
        )
        self._client = instructor.from_openai(
            base_client, mode=instructor.Mode.JSON
        )
        self.model = model
        self.thinking: str = thinking

    def _build_thinking_kwargs(self) -> dict:
        """根据 thinking 模式构造请求额外参数

        - auto:不传参,用模型/网关默认
        - enabled / disabled:通过 extra_body 透传
        """
        if self.thinking == "auto":
            return {}
        return {"extra_body": {"thinking": {"type": self.thinking}}}

    def extract(
        self,
        response_model: type[T],
        messages: list[dict],
        max_retries: int = 3,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> T:
        """提取结构化对象

        Args:
            response_model:Pydantic 类(BaseModel 子类)
            messages:对话历史
            max_retries:Pydantic 校验失败时的重试次数(Instructor 自动把错误注入对话让模型修)
            max_tokens / temperature:常规 LLM 参数

        Returns:
            response_model 的实例(类型化对象)

        Raises:
            StructuredExtractionError:重试耗尽仍校验失败
        """
        try:
            return self._client.chat.completions.create(
                model=self.model,
                response_model=response_model,
                max_retries=max_retries,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **self._build_thinking_kwargs(),
            )
        except Exception as e:
            raise StructuredExtractionError(
                retries=max_retries, last_error=e
            ) from e
