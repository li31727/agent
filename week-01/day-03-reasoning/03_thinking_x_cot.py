"""
Week 1 - Day 3 - Step 3: Thinking 模式 × CoT Prompt 矩阵实验
=============================================================

🎯 学习目标:
  在同一个模型(deepseek-v4-flash)上对比 4 种组合:

    A. thinking=disabled, Direct prompt    ← 纯 fast 模式
    B. thinking=disabled, CoT prompt       ← prompt 救场
    C. thinking=enabled,  Direct prompt    ← 模型自带思考
    D. thinking=enabled,  CoT prompt       ← 双 buff 叠加

  数据点:正确性 / token 消耗 / 耗时

📚 核心问题:
  - 思考型模型已经"在心里 CoT 了",外部 CoT prompt 还有用吗?
  - 如果模型自带 thinking,我们是不是该关掉它,用 prompt 自己控制?
  - 成本最优组合是哪个?

▶️ 运行:
   uv run week-01/day-03-reasoning/03_thinking_x_cot.py

📝 课后思考(知识回顾 Q3.8-Q3.10):
  - Q3.8 4 种组合中,token 最便宜+准确率最高的是哪个?
  - Q3.9 thinking 内化的"思考"和 prompt 引导的"思考",输出有差异吗?
  - Q3.10 生产环境的"复杂度路由"应该按 task 怎么切换?
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "day-01-llm-basics"))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402


# ============================================================
# 测试题(用 Demo 1 中最难的那道)
# ============================================================

HARD_PROBLEM = (
    "原价 380 元,会员 88 折,再叠加 30 元优惠券,最后用 50 元积分抵扣。实际付多少?"
)
EXPECTED = 254.4

DIRECT_PROMPT = (
    f"直接给出答案数字,不要解释或推理:\n\n"
    f"题目: {HARD_PROBLEM}\n\n"
    f"答案:"
)

COT_PROMPT = f"""请一步步计算下题。

要求:
1. 列出所有优惠
2. 按生效顺序逐步算
3. 最后一行严格格式: 答案 = X.XX

题目: {HARD_PROBLEM}
"""


# ============================================================
# 调用 + 答案提取
# ============================================================


def call(prompt: str, thinking_mode: str) -> tuple[str, dict, float]:
    """返回 (text, usage_dict, elapsed_seconds)"""
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    extra_kwargs = {}
    if thinking_mode != "auto":
        extra_kwargs["extra_body"] = {
            "thinking": {"type": thinking_mode}
        }

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=600,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        **extra_kwargs,
    )
    elapsed = time.perf_counter() - start

    return (
        response.choices[0].message.content,
        {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
        elapsed,
    )


def extract_answer(text: str) -> float | None:
    match = re.search(r"答案\s*[=::]\s*([0-9]+\.?[0-9]*)", text)
    if match:
        return float(match.group(1))
    numbers = re.findall(r"[0-9]+\.?[0-9]*", text)
    return float(numbers[-1]) if numbers else None


# ============================================================
# 4 种实验组合
# ============================================================

EXPERIMENTS = [
    ("A. thinking=disabled, Direct", "disabled", DIRECT_PROMPT),
    ("B. thinking=disabled, CoT   ", "disabled", COT_PROMPT),
    ("C. thinking=enabled,  Direct", "enabled", DIRECT_PROMPT),
    ("D. thinking=enabled,  CoT   ", "enabled", COT_PROMPT),
]


if __name__ == "__main__":
    print("🧪 实验:Thinking 模式 × CoT Prompt 4×1 矩阵")
    print(f"题目:{HARD_PROBLEM}")
    print(f"标准答案:{EXPECTED}\n")

    results = []
    for name, thinking, prompt in EXPERIMENTS:
        print(f"⏳ {name}...")
        text, usage, elapsed = call(prompt, thinking)
        answer = extract_answer(text)
        is_correct = answer is not None and abs(answer - EXPECTED) < 0.01
        results.append(
            {
                "name": name,
                "answer": answer,
                "correct": is_correct,
                "input": usage["input"],
                "output": usage["output"],
                "total": usage["input"] + usage["output"],
                "elapsed": elapsed,
                "preview": text[:80].replace("\n", " "),
            }
        )

    # ============================================================
    # 表格汇总
    # ============================================================

    print(f"\n{'=' * 88}")
    print(
        f"{'实验':<32} {'答案':>10} {'对':<4} "
        f"{'in':>5} {'out':>5} {'总':>5} {'耗时':>7}"
    )
    print("=" * 88)
    for r in results:
        icon = "✅" if r["correct"] else "❌"
        print(
            f"{r['name']:<32} {str(r['answer']):>10} {icon:<4} "
            f"{r['input']:>5} {r['output']:>5} {r['total']:>5} "
            f"{r['elapsed']:>5.2f}s"
        )

    # ============================================================
    # 分析
    # ============================================================

    print(f"\n{'=' * 70}")
    print("📊 数据分析")
    print("=" * 70)

    correct_count = sum(1 for r in results if r["correct"])
    print(f"✅ 正确组合: {correct_count}/4")

    # 找最便宜的正确组合
    correct_ones = [r for r in results if r["correct"]]
    if correct_ones:
        cheapest = min(correct_ones, key=lambda r: r["total"])
        print(
            f"💰 最便宜的正确组合: {cheapest['name'].strip()} "
            f"(total={cheapest['total']} tokens, {cheapest['elapsed']:.1f}s)"
        )

    # 找最快的正确组合
    if correct_ones:
        fastest = min(correct_ones, key=lambda r: r["elapsed"])
        print(
            f"⚡ 最快的正确组合: {fastest['name'].strip()} "
            f"({fastest['elapsed']:.1f}s, {fastest['total']} tokens)"
        )

    # thinking 多花的 token
    a_total = results[0]["total"]
    c_total = results[2]["total"]
    print(f"\n🔍 thinking=enabled 比 disabled 多消耗 {c_total - a_total} tokens (Direct prompt 下)")

    b_total = results[1]["total"]
    d_total = results[3]["total"]
    print(f"🔍 CoT prompt 在 thinking=disabled 下额外 {b_total - a_total} tokens")
    print(f"🔍 CoT prompt 在 thinking=enabled 下额外 {d_total - c_total} tokens")

    print("\n💡 写到 notes 的关键问题:")
    print("  1. A vs B:CoT 在 fast 模式下能\"救\"准确率吗?")
    print("  2. A vs C:thinking 内化 vs prompt 引导,谁更强?")
    print("  3. C vs D:已经 thinking 了,再加 CoT 是否冗余甚至有害?")
    print("  4. 综合最优组合是哪个?生产环境怎么按任务复杂度路由?")
