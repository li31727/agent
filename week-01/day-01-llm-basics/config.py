"""
统一配置层 — 集中管理 API Key / Model / Base URL / Thinking 开关
================================================================

🎯 设计思想:
  - 所有配置都从 .env 读取,代码不写死任何敏感信息或业务参数
  - 用 pydantic-settings 自动校验类型 + 提供默认值
  - Literal 类型可以让非法值在启动时就报错(比 str 更安全)
  - 这是生产级最佳实践,面试常考

📝 .env 配置示例(在仓库根目录的 .env 添加):

  # ====== 必填(API Keys)======
  ANTHROPIC_API_KEY=sk-ant-xxx
  OPENAI_API_KEY=sk-xxx

  # ====== 可选(模型选择,有默认值)======
  ANTHROPIC_MODEL=claude-sonnet-4-5
  OPENAI_MODEL=gpt-4o

  # ====== 可选(Base URL,留空走官方;走代理或国内中转时填)======
  # ANTHROPIC_BASE_URL=https://api.anthropic.com
  # OPENAI_BASE_URL=https://api.openai.com/v1

  # ====== 可选(思考模式)======
  # auto     = 不传参,用模型/网关默认(DeepSeek-R1 类默认开)
  # enabled  = 强制开思考(慢但准)
  # disabled = 强制关思考(快,普通对话用)
  # 参考 DeepSeek 官方:https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
  ANTHROPIC_THINKING=auto

🔍 验证配置:
   uv run week-01/day-01-llm-basics/config.py
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 仓库根目录(本文件 → day-01-llm-basics → week-01 → 仓库根)
REPO_ROOT = Path(__file__).resolve().parents[2]

# 思考模式的合法值(用 Literal 让 pydantic 启动时校验)
ThinkingMode = Literal["auto", "enabled", "disabled"]


class Settings(BaseSettings):
    """全局配置对象 — 自动从 .env 加载"""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== Anthropic 协议 =====
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")

    # 思考模式开关 — 可在 .env 里随时切换,代码不用改
    anthropic_thinking: ThinkingMode = Field(default="auto", alias="ANTHROPIC_THINKING")
    openai_thinking: ThinkingMode = Field(default="auto", alias="OPENAI_THINKING")

    # ===== OpenAI 协议 =====
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")


# 全局单例 — 业务代码只需 `from config import settings`
settings = Settings()


if __name__ == "__main__":
    print("✅ 配置加载成功:\n")
    print(f"  🟪 Anthropic Model     : {settings.anthropic_model}")
    print(f"  🟪 Anthropic Base URL  : {settings.anthropic_base_url or '(默认官方)'}")
    print(f"  🟪 Anthropic API Key   : {'*' * 10}{settings.anthropic_api_key[-4:]}")
    print(f"  🟪 Anthropic Thinking  : {settings.anthropic_thinking}")
    print()
    print(f"  🟩 OpenAI Model        : {settings.openai_model}")
    print(f"  🟩 OpenAI Base URL     : {settings.openai_base_url or '(默认官方)'}")
    print(f"  🟩 OpenAI API Key      : {'*' * 10}{settings.openai_api_key[-4:]}")
    print(f"  🟩 OpenAI Thinking     : {settings.openai_thinking}")