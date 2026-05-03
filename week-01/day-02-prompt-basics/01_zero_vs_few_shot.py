"""
Week 1 - Day 2 - Step 1: Zero-shot vs Few-shot 实测
=====================================================

🎯 学习目标:
  1. 体验 Zero-shot 在简单任务下的表现
  2. 看 Few-shot 如何"教会"模型新的输出模式
  3. 用真实业务场景(导购评论分类)实测准确率差异

📚 概念速查:
  Zero-shot: 不给例子,直接问 — "判断这条评论的情感"
  Few-shot:  给 N 个例子让模型学模式 — "这是 3 个例子,现在分类:..."
  注意:Few-shot 不是"学习"而是 In-Context Learning(上下文学习),
       模型权重不变,只是从你的例子里"看"出模式。

▶️ 运行(在仓库根目录):
   uv run week-01/day-02-prompt-basics/01_zero_vs_few_shot.py

📝 课后思考(写到 notes.md):
  - Few-shot 的例子越多越好吗? 成本和效果如何平衡?
  - 例子的"代表性"和"多样性"哪个更重要?
  - 如果 Zero-shot 已经准确率 100%,Few-shot 还有意义吗?
"""

import sys
from pathlib import Path

# 复用 day-01 的 config 和 utils(Day 5 会重构成共享模块)
sys.path.insert(0, str(Path(__file__).parent.parent / "day-01-llm-basics"))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402

# ============================================================
# 测试数据集 — 10 条人工标注的商品评论
# ============================================================

TEST_CASES = [
    {"text": "包装精美,送货很快,客服态度也好,值得推荐", "label": "正面"},
    {"text": "用了一个月就坏了,客服各种推脱,千万别买", "label": "负面"},
    {"text": "外观还行,功能一般,价格略贵", "label": "中性"},
    {"text": "孩子很喜欢,质量也不错,会回购", "label": "正面"},
    {"text": "宣传图和实物差距大,有点失望", "label": "负面"},
    {"text": "可以用,没什么特别亮点", "label": "中性"},
    {"text": "买了三个,效果都很满意,推荐给闺蜜了", "label": "正面"},
    {"text": "材质很差,刚拆封就有异味,不能接受", "label": "负面"},
    {"text": "尺码偏小一码,穿着还可以", "label": "中性"},
    {"text": "性价比超高,完全超出预期,五星好评", "label": "正面"},
]


# ============================================================
# 共享 LLM 调用
# ============================================================


def call_llm(prompt: str) -> str:
    """简单调用 — temperature=0 保证可复现"""
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=10,        # 只要分类标签,不浪费 token
        temperature=0,        # 零温度,确定性输出
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": settings.openai_thinking}}

    )
    return response.choices[0].message.content.strip()


# ============================================================
# Prompt 1: Zero-shot
# ============================================================


def zero_shot_classify(text: str) -> str:
    """直接问,不给任何例子"""
    prompt = f"""判断以下商品评论的情感倾向(只回答:正面/中性/负面,不要解释)

评论:{text}
情感:"""
    return call_llm(prompt)


# ============================================================
# Prompt 2: Few-shot (3 个示例)
# ============================================================


def few_shot_classify(text: str) -> str:
    """给 3 个示例,让模型学输出模式"""
    prompt = f"""判断商品评论的情感倾向(只回答:正面/中性/负面)

示例 1:
评论:这个手机很流畅,拍照清晰,值得购买
情感:正面

示例 2:
评论:用了几天就闪退,卡顿严重,不推荐
情感:负面

示例 3:
评论:外观一般,功能够用,没什么惊喜
情感:中性

现在分类:
评论:{text}
情感:"""
    return call_llm(prompt)


# ============================================================
# 评估器
# ============================================================


def evaluate(method_name: str, classify_fn) -> dict:
    """跑测试集,返回准确率"""
    correct = 0
    results = []
    for case in TEST_CASES:
        pred = classify_fn(case["text"])
        # "包含即对" — 模型可能输出 "正面" 或 "正面 - 因为..."
        is_correct = case["label"] in pred
        results.append(
            {
                "text": case["text"][:25] + "...",
                "expected": case["label"],
                "predicted": pred,
                "correct": is_correct,
            }
        )
        if is_correct:
            correct += 1

    return {
        "method": method_name,
        "accuracy": correct / len(TEST_CASES),
        "correct": correct,
        "total": len(TEST_CASES),
        "results": results,
    }


def print_report(report: dict):
    print(f"\n{'=' * 70}")
    print(
        f"📊 {report['method']}: "
        f"{report['correct']}/{report['total']} = {report['accuracy']:.0%}"
    )
    print("=" * 70)
    for r in report["results"]:
        icon = "✅" if r["correct"] else "❌"
        print(
            f"{icon} 期望:{r['expected']:4s} | 预测:{r['predicted']:8s} | {r['text']}"
        )


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("🧪 实验:Zero-shot vs Few-shot 商品评论情感分类")
    print(f"模型:{settings.openai_model}")
    print(f"测试样本:{len(TEST_CASES)} 条\n")

    print("⏳ 跑 Zero-shot...")
    zero_report = evaluate("Zero-shot (无示例)", zero_shot_classify)

    print("⏳ 跑 Few-shot (3 examples)...")
    few_report = evaluate("Few-shot (3 个示例)", few_shot_classify)

    print_report(zero_report)
    print_report(few_report)

    delta = (few_report["accuracy"] - zero_report["accuracy"]) * 100
    print(f"\n💡 Few-shot 相比 Zero-shot 准确率变化: {delta:+.0f}%")

    if delta > 10:
        print("   → Few-shot 显著有效(任务模式不那么直观)")
    elif delta > 0:
        print("   → Few-shot 边际有效(任务比较直观)")
    else:
        print("   → Few-shot 无效甚至有害(模型已经\"懂\",示例可能误导)")
