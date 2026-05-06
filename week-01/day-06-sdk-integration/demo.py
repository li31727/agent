"""
Week 1 - Day 6: LLM SDK 整合 Demo
===================================

🎯 学习目标:
  1. 看清"零散 demo → 可复用 SDK"的封装过程
  2. 体会 Day 1-5 的所有工程经验集中输出
  3. 用 SDK 重写 Day 1 / Day 4 / Day 5 的核心场景
     看代码量减少了多少(预期:从 ~150 行/demo → ~20 行/demo)

📚 SDK 内置的"已踩过的坑"(以后永远不用再撞):
  ✓ 思考模式开关(Day 3 学的 extra_body.thinking)
  ✓ reasoning_content 续写适配(Day 4 自己 debug 出来的)
  ✓ Mode.JSON 兼容国内网关(Day 5 自己 debug 出来的)
  ✓ MAX_ITERATIONS 防死循环(Day 4)
  ✓ Pydantic 失败自动重试(Day 5)
  ✓ 统一返回结构(text / thinking / tool_calls / tokens)

▶️ 运行:
   uv run week-01/day-06-sdk-integration/demo.py
"""

import json
import sys
from pathlib import Path
from typing import Literal

# 把 src/ 和 day-01-llm-basics/ 都加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "day-01-llm-basics"))

from pydantic import BaseModel, Field  # noqa: E402

from config import settings  # noqa: E402
from llm_kit import LLMClient, ReActAgent, StructuredExtractor  # noqa: E402


# ============================================================
# Demo 1:基础聊天(替代 Day 1)
# ============================================================


def demo_basic_chat():
    print("\n" + "=" * 70)
    print("🟦 Demo 1:基础聊天(替代 Day 1 的 01_sync.py)")
    print("=" * 70)

    client = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        thinking=settings.openai_thinking,
    )

    response = client.chat(
        messages=[{"role": "user", "content": "用 30 字解释什么是 ReAct 模式"}],
        max_tokens=200,
    )

    if response["thinking"]:
        print(f"💭 thinking: {response['thinking'][:80]}...")
    print(f"📤 答案: {response['text']}")
    print(f"📊 tokens: input={response['tokens']['input']}, output={response['tokens']['output']}")
    print(f"🏷️ finish_reason: {response['finish_reason']}")


# ============================================================
# Demo 2:ReAct Agent(替代 Day 4 的 02_react_loop.py)
# ============================================================

PRODUCTS = {
    "小米手环 8": {"id": "p001", "price": 199, "stock": 23, "rating": 4.5},
    "Apple Watch SE": {"id": "p003", "price": 1499, "stock": 5, "rating": 4.8},
    "祖玛珑香水套装": {"id": "p007", "price": 480, "stock": 30, "rating": 4.7},
    "野兽派永生花礼盒": {"id": "p005", "price": 388, "stock": 50, "rating": 4.9},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "搜索商品(按最高价格)",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {"type": "number", "description": "最高价"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "查商品详情",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """简化版工具调度器"""
    if name == "search_products":
        max_price = args.get("max_price", 99999)
        results = [
            {"name": k, **v} for k, v in PRODUCTS.items() if v["price"] <= max_price
        ]
        return json.dumps({"count": len(results), "products": results}, ensure_ascii=False)
    elif name == "get_product_info":
        product_name = args.get("name", "")
        if product_name in PRODUCTS:
            return json.dumps(
                {"found": True, "name": product_name, **PRODUCTS[product_name]},
                ensure_ascii=False,
            )
        return json.dumps({"found": False}, ensure_ascii=False)
    return json.dumps({"error": f"unknown tool: {name}"})


def demo_react_agent():
    print("\n" + "=" * 70)
    print("🟪 Demo 2:ReAct Agent(替代 Day 4 的 02_react_loop.py)")
    print("=" * 70)

    client = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        thinking=settings.openai_thinking,

    )

    agent = ReActAgent(
        client=client,
        tools=TOOLS,
        tool_executor=execute_tool,
        system_prompt=(
            "你是导购助手。当用户问商品时:\n"
            "1. 先用 search_products 搜索\n"
            "2. 再用 get_product_info 查感兴趣的详情\n"
            "3. 最后给推荐"
        ),
        max_iterations=5,
    )

    result = agent.run(
        user_message="500 元以内有什么礼物推荐?",
        verbose=True,
    )

    print(f"\n📤 最终答案:\n{result['answer'][:300]}...")
    print(
        f"\n📊 统计:{result['iterations']} 轮 / "
        f"{result['total_tokens']} tokens / "
        f"{len(result['tool_calls'])} 次工具调用"
    )


# ============================================================
# Demo 3:结构化输出(替代 Day 5 的 02_pydantic_instructor.py)
# ============================================================


class ReviewAnalysis(BaseModel):
    """商品评论分析"""

    product: str = Field(description="商品全名")
    sentiment: Literal["正面", "中性", "负面"] = Field(description="情感")
    rating: int = Field(ge=1, le=5, description="评分 1-5")
    one_liner: str = Field(
        description="一句话总结", min_length=10, max_length=50
    )


def demo_structured_output():
    print("\n" + "=" * 70)
    print("🟩 Demo 3:结构化输出(替代 Day 5 的 02_pydantic_instructor.py)")
    print("=" * 70)

    extractor = StructuredExtractor(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        thinking=settings.openai_thinking,
    )

    review = "买了小米手环 8,屏幕清晰但续航一般,综合 3 星推荐。"

    result: ReviewAnalysis = extractor.extract(
        response_model=ReviewAnalysis,
        messages=[
            {"role": "user", "content": f"分析这条评论:{review}"},
        ],
    )

    print(f"📦 类型: {type(result).__name__}")
    print(f"   product:   {result.product}")
    print(f"   sentiment: {result.sentiment}")
    print(f"   rating:    {result.rating} ⭐")
    print(f"   one_liner: {result.one_liner}")
    print(f"\n💎 转 JSON:\n{result.model_dump_json(indent=2)}")


# ============================================================
# 主流程
# ============================================================


if __name__ == "__main__":
    print("🎯 Week 1 Day 6 — LLM SDK 整合 Demo")
    print(f"模型: {settings.openai_model}")
    print(f"thinking: {settings.openai_thinking}")

    demo_basic_chat()
    demo_react_agent()
    demo_structured_output()

    print("\n" + "=" * 70)
    print("✅ 全部跑通!Day 1-5 的精华都封装在 src/llm_kit/ 里了")
    print("=" * 70)
    print("""
💡 对比代码量:
   Day 1 01_sync.py:        ~110 行 → SDK 用法 ~10 行
   Day 4 02_react_loop.py:  ~170 行 → SDK 用法 ~25 行
   Day 5 02_pydantic_xxx.py: ~160 行 → SDK 用法 ~15 行
   ─────────────────────────────────────────
   总: ~440 行 demo 代码  →  ~50 行 SDK 调用

   这就是"工程化"的 ROI:**一次封装,N 次复用**。

📌 后续扩展(留作 Day 7 / W2 任务):
   - 加 Anthropic 协议支持
   - 加重试 / 限流 / 熔断
   - 加 streaming 接口
   - 加 prompt caching
   - 加 Langfuse observability(W10)
   - Go 版本(配合你的 Eino 路线)
""")
