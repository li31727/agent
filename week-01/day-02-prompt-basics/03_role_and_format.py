"""
Week 1 - Day 2 - Step 3: 角色定义 + 输出格式控制
=================================================

🎯 学习目标:
  1. 体验"角色"如何改变模型的语气、视角和深度
  2. 学会用 Prompt 强制特定输出格式(纯文本 / Markdown / JSON)
  3. 理解前端集成时,JSON 卡片为什么是 Agent 应用首选

📚 角色 × 格式 = 4 种 UX
  纯文本 + 默认角色   → 一段平庸文字
  纯文本 + 资深买手   → 一段有温度的推荐
  Markdown + 任意角色 → 文档级阅读体验
  JSON + 任意角色     → 直接驱动前端组件渲染 ⭐

▶️ 运行:
   uv run week-01/day-02-prompt-basics/03_role_and_format.py

📝 课后思考:
  - 角色描述里加"10 年经验"这种数字,真的会让输出更专业吗?
  - 你们公司的 Agent 应用,前端用的是 Markdown 还是 JSON 卡片?
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "day-01-llm-basics"))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402


# ============================================================
# 用户问题(导购插件场景)
# ============================================================

USER_QUERY = "我想给老婆送个 500 元以内的母亲节礼物,有什么推荐?"


# ============================================================
# 实验组合
# ============================================================

# 3 种角色 × 1 种格式 = 看"角色"对内容的影响
ROLES = {
    "默认无角色": "请回答用户问题。",
    "中立购物顾问": (
        "你是一名理性的购物顾问。基于事实给出客观建议,"
        "不夸张、不煽情。"
    ),
    "10 年经验资深买手": (
        "你是一位有 10 年经验的电商买手,熟悉品牌、性价比、"
        "送礼场合和不同人群的偏好。语气温暖且专业,"
        "推荐时会考虑收礼人的情感价值。"
    ),
}

# 1 种角色 × 3 种格式 = 看"格式"对集成的影响
FORMATS = {
    "纯文本": "请用一段话给出推荐。",
    "Markdown 列表": (
        "请用 Markdown 格式输出 3 个推荐,每项包含:\n"
        "- **商品名**\n"
        "- 价格\n"
        "- 推荐理由(简短)\n"
        "- 适合送给:..."
    ),
    "JSON 卡片(前端友好)": (
        '请返回 JSON 数组,每个元素包含 name, price, reason, target 字段。\n'
        '格式示例:[{"name": "...", "price": 399, "reason": "...", "target": "..."}]\n'
        "只返回 JSON,不要其他文字,不要 Markdown 代码块。"
    ),
}


# ============================================================
# 调用
# ============================================================


def call_with_system(system: str, user: str) -> str:
    """用 system + user 双消息调用"""
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=600,
        temperature=0.7,  # 适度多样,但不发散
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


# ============================================================
# 实验 A:角色对比(固定格式 = 纯文本)
# ============================================================


def experiment_roles():
    print("\n" + "=" * 70)
    print("🧪 实验 A:同一问题 + 不同角色(格式固定为纯文本)")
    print("=" * 70)
    print(f"用户问题:{USER_QUERY}\n")

    for role_name, role_def in ROLES.items():
        system = role_def
        user = USER_QUERY
        print(f"\n--- 角色:{role_name} ---")
        response = call_with_system(system, user)
        print(response)
        print()


# ============================================================
# 实验 B:格式对比(固定角色 = 资深买手)
# ============================================================


def experiment_formats():
    print("\n" + "=" * 70)
    print("🧪 实验 B:同一问题 + 不同输出格式(角色固定为资深买手)")
    print("=" * 70)

    role = ROLES["10 年经验资深买手"]

    for fmt_name, fmt_instr in FORMATS.items():
        system = role
        user = f"{USER_QUERY}\n\n{fmt_instr}"
        print(f"\n--- 格式:{fmt_name} ---")
        response = call_with_system(system, user)

        # JSON 格式额外做解析验证
        if "JSON" in fmt_name:
            try:
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = "\n".join(cleaned.split("\n")[1:-1])
                parsed = json.loads(cleaned)
                print("✅ JSON 解析成功:")
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解析失败: {e}")
                print(f"原始: {response[:300]}")
        else:
            print(response)
        print()


if __name__ == "__main__":
    print(f"🎯 模型:{settings.openai_model}")
    print(f"📌 用户问题:{USER_QUERY}")

    experiment_roles()
    experiment_formats()

    print("\n" + "=" * 70)
    print("💡 观察记录(填到 notes.md):")
    print("=" * 70)
    print("  1. 三个角色的输出,风格差异有多大? 你觉得哪个最好?")
    print("  2. JSON 输出能直接给前端用吗? 哪些字段值得加?")
    print("  3. 如果让你做导购插件,你会选哪种角色 + 哪种格式?")
