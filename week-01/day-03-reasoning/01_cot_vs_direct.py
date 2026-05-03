"""
Week 1 - Day 3 - Step 1: CoT vs Direct 实测
=============================================

🎯 学习目标:
  1. 看 CoT(Chain-of-Thought)在多步推理任务上的威力
  2. 理解为什么 "Let's think step by step" 是 NLP 史上最便宜的优化
  3. 学会用 regex 从 LLM 输出中提取结构化答案 + 容差比较

📚 概念:
  Direct: 直接问答案 — "原价 X 元,叠加优惠后多少?"
  CoT:    "请一步步推理后给出答案"
  研究:Wei et al. 2022 在数学/常识/符号推理上,CoT 提升 30-60%

▶️ 运行:
   uv run week-01/day-03-reasoning/01_cot_vs_direct.py

📝 课后思考(写到「知识回顾.md」Q3.1-Q3.4):
  - Q3.1 CoT 为什么对推理任务有效?(底层机制)
  - Q3.2 CoT 在哪些任务上无效甚至有害?
  - Q3.3 "step by step"和"请详细解释"哪个 prompt 更好?
  - Q3.4 思考型模型(如 DeepSeek-R1)是不是把 CoT 内化了?
"""

import re
import sys
from pathlib import Path

# 复用 day-01 的 config
sys.path.insert(0, str(Path(__file__).parent.parent / "day-01-llm-basics"))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402


# ============================================================
# 测试集 — 5 道电商优惠计算题(标准答案预先验证)
# ============================================================

TEST_PROBLEMS = [
    {
        "problem": "某商品原价 299 元,满 200 减 30,会员额外 9 折。实际付多少?",
        "answer": 242.1,  # (299 - 30) × 0.9 = 242.1
    },
    {
        "problem": "购物车有 3 件商品,单价分别是 89、119、56 元。满 2 件 9 折。实际付多少?",
        "answer": 237.6,  # (89 + 119 + 56) × 0.9 = 264 × 0.9 = 237.6
    },
    {
        "problem": "原价 199 元,满 100 减 20,叠加 8 折优惠券,再用 15 元店铺红包。实际付多少?",
        "answer": 128.2,  # (199 - 20) × 0.8 - 15 = 179 × 0.8 - 15 = 143.2 - 15 = 128.2
    },
    {
        "problem": "A 商品 49 元,B 商品 79 元,买 A 加 1 元换购 B(B 减为 1 元)。共付多少?",
        "answer": 50.0,  # 49 + 1 = 50
    },
    {
        "problem": "原价 380 元,会员 88 折,再叠加 30 元优惠券,最后用 50 元积分抵扣。实际付多少?",
        "answer": 254.4,  # 380 × 0.88 - 30 - 50 = 334.4 - 80 = 254.4
    },
]


# ============================================================
# Prompt 模板
# ============================================================

DIRECT_PROMPT_TEMPLATE = """请计算下题,直接给出最终金额数字,不要任何解释或推理过程。
格式严格要求:只回答一个数字(可带小数),例如:123.45

题目:{problem}

答案:"""


COT_PROMPT_TEMPLATE = """请一步步计算下题,展示完整推理过程。

要求:
1. 先列出所有优惠条件
2. 按优惠生效顺序逐步计算(注意优惠叠加顺序)
3. 最后一行单独给出最终答案,严格格式:`答案 = X.XX`

题目:{problem}
"""


# ============================================================
# 调用 + 答案提取
# ============================================================


def call(prompt: str) -> str:
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    # thinking 配置(沿用 Day 2 的写法)
    extra_kwargs = {}
    if settings.openai_thinking != "auto":
        extra_kwargs["extra_body"] = {
            "thinking": {"type": settings.openai_thinking}
        }

    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=600,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        **extra_kwargs,
    )
    return response.choices[0].message.content


def extract_number(text: str, prefer_marker: bool = True) -> float | None:
    """
    从 LLM 输出中提取数字答案

    prefer_marker=True:  优先找"答案 = X"格式(CoT 输出)
    prefer_marker=False: 直接取最后一个数字(Direct 输出)
    """
    if prefer_marker:
        # CoT 模式:找"答案 = 123.45"
        match = re.search(r"答案\s*[=:：]\s*([0-9]+\.?[0-9]*)", text)
        if match:
            return float(match.group(1))

    # 兜底:取文本中最后一个数字(Direct 模式或 CoT 标记缺失)
    numbers = re.findall(r"[0-9]+\.?[0-9]*", text)
    if numbers:
        return float(numbers[-1])
    return None


# ============================================================
# 评估器
# ============================================================


def evaluate(method_name: str, prompt_template: str, prefer_marker: bool):
    correct = 0
    results = []
    for case in TEST_PROBLEMS:
        prompt = prompt_template.format(problem=case["problem"])
        response = call(prompt)
        predicted = extract_number(response, prefer_marker=prefer_marker)
        # 容差 0.01(浮点比较)
        is_correct = (
            predicted is not None and abs(predicted - case["answer"]) < 0.01
        )
        results.append(
            {
                "problem": case["problem"][:30] + "...",
                "expected": case["answer"],
                "predicted": predicted,
                "correct": is_correct,
                "preview": response[:60].replace("\n", " "),
            }
        )
        if is_correct:
            correct += 1

    return {
        "method": method_name,
        "accuracy": correct / len(TEST_PROBLEMS),
        "correct": correct,
        "total": len(TEST_PROBLEMS),
        "results": results,
    }


def print_report(report):
    print(f"\n{'=' * 75}")
    print(
        f"📊 {report['method']}: "
        f"{report['correct']}/{report['total']} = {report['accuracy']:.0%}"
    )
    print("=" * 75)
    for r in report["results"]:
        icon = "✅" if r["correct"] else "❌"
        print(
            f"{icon} 期望:{r['expected']:>7} | 预测:{str(r['predicted']):>8} | {r['problem']}"
        )
        if not r["correct"]:
            print(f"   ↳ 输出预览: {r['preview']}")


if __name__ == "__main__":
    print("🧪 实验:CoT vs Direct (电商优惠计算)")
    print(f"模型:{settings.openai_model}")
    print(f"测试题数:{len(TEST_PROBLEMS)}\n")

    print("⏳ 跑 Direct (直接问答案)...")
    direct_report = evaluate(
        "Direct (直接问)", DIRECT_PROMPT_TEMPLATE, prefer_marker=False
    )

    print("⏳ 跑 CoT (一步步推理)...")
    cot_report = evaluate(
        "CoT (Chain-of-Thought)", COT_PROMPT_TEMPLATE, prefer_marker=True
    )

    print_report(direct_report)
    print_report(cot_report)

    delta = (cot_report["accuracy"] - direct_report["accuracy"]) * 100
    print(f"\n💡 CoT 相比 Direct 准确率变化: {delta:+.0f}%")

    if delta > 20:
        print("   → CoT 强烈推荐(任务推理复杂度高)")
    elif delta > 0:
        print("   → CoT 有边际价值")
    elif delta == 0:
        print("   → 持平 — 可能因为模型已\"内化\"CoT(thinking 模式)")
    else:
        print("   → CoT 反而下降 — 可能是答案提取问题或任务太简单")
