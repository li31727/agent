"""
Week 1 - Day 3 - Step 2: Self-Consistency(自一致性)
======================================================

🎯 学习目标:
  1. 理解 Self-Consistency = 多次采样 + 多数投票
  2. 看 temperature=0.7 + N 次采样 vs 单次 temp=0 的稳定性差异
  3. 体会"集体投票"如何从概率上提升准确率(Wisdom of Crowds for LLM)

📚 概念:
  - 单次 temp=0:确定性输出,但可能"系统性错误"
  - 多次 temp=0.7 + 投票:每次有随机性,多数票更可能命中真值
  - 适合:数学/分类/事实(有明确正确答案)
  - 不适合:写作/创意(无单一正确答案)

  研究:Wang et al. 2022 在 GSM8K 数学题上,
  CoT + Self-Consistency 比 CoT 单次再涨 17%

▶️ 运行:
   uv run week-01/day-03-reasoning/02_self_consistency.py

📝 课后思考(写到知识回顾.md Q3.5-Q3.7):
  - Q3.5 N=5 vs N=11,投票稳定性差多少?
  - Q3.6 Self-Consistency 的成本是 N 倍 token,值得吗?
  - Q3.7 思考型模型还需要 Self-Consistency 吗?
"""

import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "day-01-llm-basics"))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402


# ============================================================
# 选 1 道难题(Demo 1 测试集中最复杂的)
# ============================================================

HARD_PROBLEM = (
    "原价 380 元,会员 88 折,再叠加 30 元优惠券,最后用 50 元积分抵扣。实际付多少?"
)
EXPECTED = 254.4

N_SAMPLES = 5  # 改成 11 试试


# ============================================================
# 调用与答案提取
# ============================================================


def call_with_temp(temperature: float) -> str:
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    extra_kwargs = {}
    if settings.openai_thinking != "auto":
        extra_kwargs["extra_body"] = {
            "thinking": {"type": settings.openai_thinking}
        }

    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=400,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": (
                    f"请一步步计算: {HARD_PROBLEM}\n"
                    f"最后一行严格格式: 答案 = X.XX"
                ),
            },
        ],
        **extra_kwargs,
    )
    return response.choices[0].message.content


def extract_answer(text: str) -> float | None:
    match = re.search(r"答案\s*[=::]\s*([0-9]+\.?[0-9]*)", text)
    if match:
        return float(match.group(1))
    # 兜底
    numbers = re.findall(r"[0-9]+\.?[0-9]*", text)
    return float(numbers[-1]) if numbers else None


def majority_vote(answers: list) -> float | None:
    """取出现次数最多的答案"""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("🧪 实验:Self-Consistency")
    print(f"题目: {HARD_PROBLEM}")
    print(f"标准答案: {EXPECTED}")
    print(f"采样次数: {N_SAMPLES}\n")

    # ============================================================
    # 基线 1:单次 temp=0(确定性)
    # ============================================================
    print("⏳ 基线 1:单次 temp=0(确定性)...")
    start = time.perf_counter()
    single_response = call_with_temp(0.0)
    single_time = time.perf_counter() - start
    single_answer = extract_answer(single_response)
    single_correct = single_answer is not None and abs(single_answer - EXPECTED) < 0.01

    print(f"  答案: {single_answer}")
    print(f"  耗时: {single_time:.2f}s")
    print(f"  {'✅ 正确' if single_correct else '❌ 错误'}\n")

    # ============================================================
    # 基线 2:单次 temp=0.7(随机性 baseline)
    # ============================================================
    print("⏳ 基线 2:单次 temp=0.7(随机)...")
    start = time.perf_counter()
    rand_response = call_with_temp(0.7)
    rand_time = time.perf_counter() - start
    rand_answer = extract_answer(rand_response)
    rand_correct = rand_answer is not None and abs(rand_answer - EXPECTED) < 0.01

    print(f"  答案: {rand_answer}")
    print(f"  耗时: {rand_time:.2f}s")
    print(f"  {'✅ 正确' if rand_correct else '❌ 错误'}\n")

    # ============================================================
    # 实验:多次 temp=0.7 + 多数票
    # ============================================================
    print(f"⏳ 实验:Self-Consistency(temp=0.7,采样 {N_SAMPLES} 次)...")
    start = time.perf_counter()
    answers = []
    for i in range(N_SAMPLES):
        response = call_with_temp(0.7)
        answer = extract_answer(response)
        answers.append(answer)
        print(f"  样本 {i + 1}/{N_SAMPLES}: {answer}")
    multi_time = time.perf_counter() - start

    voted = majority_vote(answers)
    voted_correct = voted is not None and abs(voted - EXPECTED) < 0.01

    print("\n  📊 答案分布:")
    for ans, cnt in Counter(answers).most_common():
        bar = "█" * cnt
        print(f"    {str(ans):>10} : {bar} {cnt}")
    print(f"\n  🗳️ 多数票答案: {voted}")
    print(f"  耗时: {multi_time:.2f}s ({N_SAMPLES} 次)")
    print(f"  {'✅ 正确' if voted_correct else '❌ 错误'}")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n{'=' * 70}")
    print("📊 三种方法对比")
    print("=" * 70)
    print(
        f"{'方法':<25} {'答案':>10} {'正确':<6} {'耗时':>8}  {'成本':<10}"
    )
    print("-" * 70)
    print(
        f"{'单次 temp=0':<25} {str(single_answer):>10} "
        f"{'✅' if single_correct else '❌':<6} "
        f"{single_time:>6.2f}s  {'1x':<10}"
    )
    print(
        f"{'单次 temp=0.7':<25} {str(rand_answer):>10} "
        f"{'✅' if rand_correct else '❌':<6} "
        f"{rand_time:>6.2f}s  {'1x':<10}"
    )
    print(
        f"{'Self-Consistency (N=' + str(N_SAMPLES) + ')':<25} {str(voted):>10} "
        f"{'✅' if voted_correct else '❌':<6} "
        f"{multi_time:>6.2f}s  {f'{N_SAMPLES}x':<10}"
    )

    print("\n💡 观察要点(写到 notes):")
    print(f"  1. {N_SAMPLES} 次采样里有几次正确?(看分布)")
    print("  2. 多数票答案是不是 majority 而不是 plurality?")
    print(f"  3. {N_SAMPLES} 倍成本换来稳定性提升,什么场景值得?")
