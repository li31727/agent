"""
Week 1 - Day 2 - Step 2: 模糊指令 vs 清晰指令
==============================================

🎯 学习目标:
  1. 看到"提取属性"这种模糊指令的输出有多自由发挥
  2. 学会写"角色 + 任务 + 格式 + 约束 + 异常处理"的工业级 Prompt
  3. 理解为什么生产代码必须强制 JSON 输出

📚 清晰指令 5 要素:
  1. 角色(Role)         — "你是电商数据分析师"
  2. 任务(Task)         — "从商品描述中提取结构化属性"
  3. 格式(Format)       — "返回合法 JSON"
  4. 字段(Schema)       — "必含 brand/model/.../price_range"
  5. 异常处理(Fallback)— "缺失字段填 null"

▶️ 运行:
   uv run week-01/day-02-prompt-basics/02_clear_instructions.py

📝 课后思考:
  - 同样的 token 数,清晰版花了 5x 的字数,值得吗?
  - 工业级 Prompt 应该多用 markdown / xml / json 哪种结构化?
  - 如果 LLM 偶尔不返回 JSON,你的代码有兜底吗?
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "day-01-llm-basics"))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402


# ============================================================
# 测试场景:从商品描述提取属性
# ============================================================

PRODUCT_DESC = """
【新品上市】小米 Redmi Note 13 Pro+ 5G 智能手机
颜色:子夜黑 / 时光蓝 / 阳光金
存储:8+256GB,12+256GB,12+512GB
屏幕:6.67 英寸 2.5K 曲面屏,120Hz 高刷
摄像头:2 亿像素主摄 + 800 万超广角
电池:5000mAh,120W 闪充
价格:1899-2399 元
"""


# ============================================================
# Prompt A:模糊指令(教程级)
# ============================================================

VAGUE_PROMPT = f"""提取这个商品的属性:

{PRODUCT_DESC}"""


# ============================================================
# Prompt B:清晰指令(工业级)
# ============================================================

CLEAR_PROMPT = f"""你是一位电商数据分析师,负责从商品描述中提取结构化属性数据。

# 任务
从下方商品描述中提取关键属性,返回**合法的 JSON 对象**。

# 输出格式要求
- 必须是合法的 JSON,不要包含任何 Markdown 代码块标记(如 ```json)
- 不要任何额外说明文字,只返回 JSON

# 必需字段
{{
  "brand": "品牌名",
  "model": "型号",
  "colors": ["颜色 1", "颜色 2"],
  "storage_options": ["8+256GB", ...],
  "screen_size": "6.67 英寸",
  "refresh_rate": "120Hz",
  "main_camera": "2 亿像素",
  "battery_capacity": "5000mAh",
  "charging_power": "120W",
  "price_range": {{"min": 1899, "max": 2399, "currency": "CNY"}}
}}

# 异常处理
- 如果某字段在描述中不存在,填 null(不要省略字段)
- 数值带单位时保留单位字符串

# 商品描述
{PRODUCT_DESC}

# JSON 输出:"""


# ============================================================
# 调用与对比
# ============================================================


def call(prompt: str) -> str:
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=600,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def try_parse_json(text: str) -> tuple[bool, dict | str]:
    """
    尝试解析 JSON。LLM 偶尔会用 ```json 包裹,这里做兜底。
    返回:(是否成功, 解析结果或错误信息)
    """
    # 兜底:去掉 markdown 代码块标记
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # 去掉首行和末行的 ```
        cleaned = "\n".join(cleaned.split("\n")[1:-1])

    try:
        return True, json.loads(cleaned)
    except json.JSONDecodeError as e:
        return False, f"{e}"


def report(label: str, prompt: str, response: str):
    print(f"\n{'=' * 70}")
    print(f"🧪 {label}")
    print(f"   Prompt 字符数: {len(prompt)} | 输出字符数: {len(response)}")
    print("=" * 70)

    # 试着解析 JSON
    ok, parsed = try_parse_json(response)
    if ok:
        print("✅ JSON 解析成功!")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        print(f"❌ JSON 解析失败: {parsed}")
        print("\n原始输出:")
        print(response[:500] + ("..." if len(response) > 500 else ""))


if __name__ == "__main__":
    print(f"模型:{settings.openai_model}\n")

    print("⏳ 跑模糊指令...")
    vague_resp = call(VAGUE_PROMPT)
    report("Prompt A:模糊指令(教程级)", VAGUE_PROMPT, vague_resp)

    print("\n⏳ 跑清晰指令...")
    clear_resp = call(CLEAR_PROMPT)
    report("Prompt B:清晰指令(工业级)", CLEAR_PROMPT, clear_resp)

    print(f"\n{'=' * 70}")
    print("💡 观察要点:")
    print("  1. 模糊指令的输出结构稳定吗? 能直接交给下游消费吗?")
    print("  2. 清晰指令多花的 prompt token 值不值?(下游不用写正则)")
    print("  3. 如果 100 个商品都要提取,哪种 Prompt 维护成本低?")
