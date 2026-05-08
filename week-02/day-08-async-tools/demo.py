"""
Week 2 - Day 1 (Day 8): 多工具并行调用实测
============================================

🎯 学习目标:
  1. 实测 asyncio.gather 在 Tool Calling 场景的加速效果
  2. 看 Day 4 的 21.9s ReAct 任务能压到几秒
  3. 体会"工具是真实 API"vs"本地函数"的加速差异

📚 测试设计:
  跑同一个任务(推荐 500 元礼物),两次:
    A. SyncReActAgent  + 同步工具(time.sleep 模拟 API 延迟)
    B. AsyncReActAgent + 异步工具(asyncio.sleep 模拟 API 延迟)

  关键变量:工具延迟(0.5s 模拟真实 API,可改 0/1.0/2.0 看不同效果)

📚 预期(单工具延迟 0.5s):
  同步:LLM 时间 + 11 个工具 × 0.5s = LLM 时间 + 5.5s
  异步:LLM 时间 + 4 轮 × max(每轮工具) × 0.5s ≈ LLM 时间 + 2s
  加速:工具部分 5.5s → 2s,接近 3x

▶️ 运行:
   uv run week-02/day-08-async-tools/demo.py

📝 课后思考(知识回顾 Q7.1-Q7.3):
  - Q7.1 工具是本地函数(无延迟)时,异步还有意义吗?
  - Q7.2 LLM 调用本身能并行吗?(在 ReAct loop 内?)
  - Q7.3 异步 + 错误恢复时,一个工具失败 gather 会怎样?(下个 Demo 学)
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# SDK 路径 + Day 1 config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "week-01" / "day-01-llm-basics"))

from config import settings  # noqa: E402

from llm_kit import (  # noqa: E402
    AsyncLLMClient,
    AsyncReActAgent,
    LLMClient,
    ReActAgent,
)


# ============================================================
# 模拟"真实 API"延迟(改成 0 看纯 LLM 的对比)
# ============================================================

TOOL_LATENCY_SECONDS = 0.5  # 模拟一次 API 调用的延迟

PRODUCTS = {
    "小米手环 8": {"id": "p001", "price": 199, "stock": 23, "rating": 4.5},
    "Apple Watch SE": {"id": "p003", "price": 1499, "stock": 5, "rating": 4.8},
    "祖玛珑香水套装": {"id": "p007", "price": 480, "stock": 30, "rating": 4.7},
    "野兽派永生花礼盒": {"id": "p005", "price": 388, "stock": 50, "rating": 4.9},
    "观夏香薰蜡烛": {"id": "p008", "price": 280, "stock": 40, "rating": 4.8},
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
                    "max_price": {"type": "number", "description": "最高价格"},
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
                    "name": {"type": "string", "description": "商品全名"},
                },
                "required": ["name"],
            },
        },
    },
]


# ============================================================
# 工具实现:同步版(time.sleep 阻塞)
# ============================================================


def execute_tool_sync(name: str, args: dict) -> str:
    """同步版工具 — 用 time.sleep 模拟 API 延迟"""
    time.sleep(TOOL_LATENCY_SECONDS)  # ⭐ 阻塞等待

    if name == "search_products":
        max_price = args.get("max_price", 99999)
        results = [
            {"name": k, **v}
            for k, v in PRODUCTS.items()
            if v["price"] <= max_price
        ]
        return json.dumps(
            {"count": len(results), "products": results}, ensure_ascii=False
        )
    elif name == "get_product_info":
        product_name = args.get("name", "")
        if product_name in PRODUCTS:
            return json.dumps(
                {"found": True, "name": product_name, **PRODUCTS[product_name]},
                ensure_ascii=False,
            )
        return json.dumps({"found": False}, ensure_ascii=False)
    return json.dumps({"error": f"unknown tool: {name}"})


# ============================================================
# 工具实现:异步版(asyncio.sleep 让出事件循环)
# ============================================================


async def execute_tool_async(name: str, args: dict) -> str:
    """异步版工具 — 用 asyncio.sleep 模拟 API 延迟,可被并发调度"""
    await asyncio.sleep(TOOL_LATENCY_SECONDS)  # ⭐ 不阻塞,让出事件循环

    if name == "search_products":
        max_price = args.get("max_price", 99999)
        results = [
            {"name": k, **v}
            for k, v in PRODUCTS.items()
            if v["price"] <= max_price
        ]
        return json.dumps(
            {"count": len(results), "products": results}, ensure_ascii=False
        )
    elif name == "get_product_info":
        product_name = args.get("name", "")
        if product_name in PRODUCTS:
            return json.dumps(
                {"found": True, "name": product_name, **PRODUCTS[product_name]},
                ensure_ascii=False,
            )
        return json.dumps({"found": False}, ensure_ascii=False)
    return json.dumps({"error": f"unknown tool: {name}"})


# ============================================================
# Test 任务
# ============================================================

USER_QUERY = "500 元以内有什么礼物推荐?"

SYSTEM_PROMPT = (
    "你是导购助手。当用户问商品时:\n"
    "1. 先用 search_products 搜索\n"
    "2. 再用 get_product_info 查感兴趣的详情\n"
    "3. 最后给推荐"
)


# ============================================================
# 跑同步版
# ============================================================


def run_sync() -> dict:
    print("\n" + "=" * 70)
    print("🔵 同步版 ReActAgent(工具串行执行)")
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
        tool_executor=execute_tool_sync,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=5,
    )

    start = time.perf_counter()
    result = agent.run(user_message=USER_QUERY, verbose=True)
    elapsed = time.perf_counter() - start

    return {
        "label": "同步串行",
        "elapsed": elapsed,
        "iterations": result["iterations"],
        "tool_calls": len(result["tool_calls"]),
        "total_tokens": result["total_tokens"],
        "answer_preview": result["answer"][:120],
    }


# ============================================================
# 跑异步版
# ============================================================


async def run_async() -> dict:
    print("\n" + "=" * 70)
    print("🟢 异步版 AsyncReActAgent(工具 asyncio.gather 并发)")
    print("=" * 70)

    client = AsyncLLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        thinking=settings.openai_thinking,
    )

    agent = AsyncReActAgent(
        client=client,
        tools=TOOLS,
        tool_executor=execute_tool_async,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=5,
    )

    start = time.perf_counter()
    result = await agent.run(user_message=USER_QUERY, verbose=True)
    elapsed = time.perf_counter() - start

    return {
        "label": "异步并发",
        "elapsed": elapsed,
        "iterations": result["iterations"],
        "tool_calls": len(result["tool_calls"]),
        "total_tokens": result["total_tokens"],
        "answer_preview": result["answer"][:120],
    }


# ============================================================
# 主流程
# ============================================================


def print_comparison(sync_data: dict, async_data: dict):
    print("\n" + "=" * 70)
    print("📊 对比结果")
    print("=" * 70)
    print(
        f"{'指标':<20} {'同步串行':>12} {'异步并发':>12} {'差异':>10}"
    )
    print("-" * 70)
    print(
        f"{'耗时(秒)':<20} "
        f"{sync_data['elapsed']:>12.2f} "
        f"{async_data['elapsed']:>12.2f} "
        f"{sync_data['elapsed'] - async_data['elapsed']:>10.2f}"
    )
    print(
        f"{'迭代轮数':<20} "
        f"{sync_data['iterations']:>12} "
        f"{async_data['iterations']:>12} "
        f"{'同' if sync_data['iterations'] == async_data['iterations'] else '不同':>10}"
    )
    print(
        f"{'工具调用次数':<20} "
        f"{sync_data['tool_calls']:>12} "
        f"{async_data['tool_calls']:>12}"
    )
    print(
        f"{'Token 总消耗':<20} "
        f"{sync_data['total_tokens']:>12} "
        f"{async_data['total_tokens']:>12}"
    )

    speedup = sync_data["elapsed"] / async_data["elapsed"]
    saved = sync_data["elapsed"] - async_data["elapsed"]
    print(f"\n💡 加速比: {speedup:.2f}x")
    print(f"💰 节省时间: {saved:.2f}s")
    print(f"🛠️ 工具延迟设定: {TOOL_LATENCY_SECONDS}s")
    print(
        f"   (LLM 时间相同,差异主要来自工具{'串行' if speedup > 1 else '本身就快'}vs"
        f"并发)"
    )

    print("\n📌 结论:")
    if speedup > 1.5:
        print("   ✅ 异步并发显著加速 — 工具有 IO 延迟时,asyncio.gather 必须用")
    elif speedup > 1.1:
        print("   🟡 异步有边际收益 — 工具延迟较小,加速不明显")
    else:
        print(
            "   ⚠️ 异步收益不足 — 可能 LLM 时间占大头,或工具本身很快"
        )


async def main():
    sync_data = run_sync()
    async_data = await run_async()
    print_comparison(sync_data, async_data)


if __name__ == "__main__":
    print(f"🎯 Week 2 Day 1 — 异步并发工具调用实测")
    print(f"模型: {settings.openai_model}")
    print(f"thinking: {settings.openai_thinking}")
    print(f"工具模拟延迟: {TOOL_LATENCY_SECONDS}s/次")
    print(f"任务: {USER_QUERY}")

    asyncio.run(main())
