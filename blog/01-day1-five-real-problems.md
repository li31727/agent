---
title: 5 年 Go 后端转 Agent · Day 1：200 行代码踩中的 5 个 LLM 工程化真问题
date: 2026-05-02
tags: [LLM, Agent, Python, Go, DeepSeek, Anthropic, 转型]
description: 5 年 Go 后端工程师转型 Agent 工程师 Day 1 踩坑实录。从 hardcode 到 pydantic-settings、从 ThinkingBlock AttributeError 到思考模式开关、从协议字段差异到 1.80x 并发加速比，全程公开学习。
cover: https://...
---

# 5 年 Go 后端转 Agent · Day 1：200 行代码踩中的 5 个 LLM 工程化真问题

> 这是一篇**公开学习日记**。我是一名 5 年 Go 后端工程师，正在用 12 周时间转型 Agent 工程师。今天是 Day 1。
>
> 原计划很简单：2 小时跑通 Anthropic + OpenAI 的 Python SDK，做点同步、异步、流式调用。**实际花了 4 小时，撞了 5 次墙**。
>
> 这篇文章把每一次踩坑的过程、原因、解决方案都摊开来写。
>
> 比起那些 *"15 分钟带你入门 LLM"* 的教程，我更想分享：**当你真把代码扔到生产之前，会撞到哪些没人提的问题**。

## TL;DR — 5 个真问题

1. **配置外置**：第一行代码就该用 `pydantic-settings`，不是 `os.getenv()`
2. **ThinkingBlock**：思考型模型的 `content[0]` 不一定是文本，会崩
3. **思考开关**：DeepSeek/o1 类模型的 thinking 是**可关的**，关掉省 50%+ token
4. **协议差异**：同模型 Anthropic 协议 vs OpenAI 协议，**字段位置完全不同**
5. **并发实测**：`asyncio.gather` 实测 1.80x 加速（理论 2x，效率 90%），TTFT 差 1.85x

---

## 为什么是 12 周？

故事很简单。

我是一个 5 年的 Go 后端，主业做 C 端导购插件。看到 Agent 工程师方向越来越火，我扒了 4 份目标公司的 JD（携程 ×2 / 智慧树 / 拓竹），把要求的能力做了一个交集：

- ✅ 我已有的：Python / Go / K8s / Redis / 微服务 / 分布式
- ❌ 我要补的：LLM 实战 / RAG / Agent 框架 / Multi-Agent / Eval 体系 / MCP

差距很明确。我给自己定了 **12 周转型计划**，平均每周 26 小时投入，**全程公开学习**：

- 每周 1 篇博客（你看的就是 #1）
- 12 周 5 个开源项目
- 边学边投递

仓库地址：[github.com/<your-github>/agent](https://github.com/) （Star 一下我会更有动力 🥺）

今天是 Day 1。任务：跑通 LLM API 的基础调用。

---

## 真问题 1：第一行代码，就该考虑配置外置

90% 的 LLM 教程是这样开头的：

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key="sk-ant-xxx-把你的 key 贴这里")

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好"}]
)
```

但**这是糟糕代码**。理由：

1. API Key 写代码里，一旦 commit 就泄露
2. 模型名硬编码，想换 GPT-4o / DeepSeek 要全文搜索改
3. `base_url` 写不进去，无法走代理 / Cloudflare AI Gateway / 国内中转
4. 没有类型校验，key 拼错到运行时才报错

我直接上了 `pydantic-settings`：

```python
# config.py
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    anthropic_thinking: Literal["auto", "enabled", "disabled"] = Field(
        default="auto", alias="ANTHROPIC_THINKING"
    )

    # OpenAI
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")


settings = Settings()
```

业务代码统一这么写：

```python
from config import settings

client = Anthropic(
    api_key=settings.anthropic_api_key,
    base_url=settings.anthropic_base_url,
)
```

好处一字排开：

- API Key 在 `.env`，**绝不进 git**（`.gitignore` 里写好）
- 切换模型只改 `.env`，代码不动
- `Literal["auto", "enabled", "disabled"]` 让非法值在**启动时**就报错，不到运行时
- 类型提示让 IDE 自动补全，少写错

> 💎 **5 年后端老兵的本能**：任何会变的东西都不写死。这条规则在 LLM 时代依然有效。**用 `pydantic-settings` 替代 `os.getenv()`，是零成本的工程化提升。**

---

## 真问题 2：第一次调用就崩了 — `'ThinkingBlock' object has no attribute 'text'`

我跑了第一个最简单的 demo：

```python
def call_anthropic(prompt: str) -> dict:
    client = Anthropic(...)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": response.content[0].text,  # ⚠️ 这行崩了
        ...
    }
```

报错：

```
AttributeError: 'ThinkingBlock' object has no attribute 'text'
```

什么？Block？ThinkingBlock？

查了一下，这是**思考型模型**（DeepSeek-R1 / Claude 4 / OpenAI o1 等）的标配响应结构：

```python
response.content = [
    ThinkingBlock(thinking="让我想想..."),     # 推理过程
    TextBlock(text="ReAct 是..."),              # 用户看的最终答案
    ToolUseBlock(...),                           # 工具调用（后续会遇到）
]
```

`content[0]` 不一定是文本，可能是模型的"内心独白"。

正确写法：

```python
def extract_text(response) -> str:
    """从响应里提取所有文本块，跳过 thinking / tool_use"""
    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )
```

加一个对偶函数，把 thinking 也单独提出来，调试时用：

```python
def extract_thinking(response) -> str:
    return "\n".join(
        block.thinking
        for block in response.content
        if block.type == "thinking"
    )
```

跑一次，看到模型的"思考过程"被打出来：

```
💭 [模型 thinking 过程]
我们要求用 50 个汉字解释什么是 ReAct（推理+行动）模式。
需要简洁准确。ReAct 是"推理+行动"的循环：
模型先思考推理，再执行行动（如调用工具），
然后观察结果，继续推理，直到完成任务...
─────────────
📤 [最终答案]
ReAct 模式让模型交替进行推理思考与实际行动...
```

**为什么 thinking 单独成 block？** 我查了下，理由很合理：

1. **可控**：开发者可以选择给不给用户看
2. **计费**：thinking 内容也算 output token，但单独标记
3. **缓存**：Anthropic 的 prompt caching 可以单独缓存 thinking
4. **审计**：思考过程可以单独存日志，回查模型决策

> 💎 **教训**：看到 `response.content[0].text` 这种写法的教程，**直接关掉**。它在思考型模型上一定崩。**生产代码必须遍历 + 按 type 过滤**，这是面试常考点。

---

## 真问题 3：思考模式可以关 — 你为什么要付那个钱？

接着看 token 消耗：

```
[tokens] input=20, output=160
```

20 个 token 的输入，160 个 token 的输出 — 但我只要 50 个汉字啊？

剩下那 110 个 token 是什么？**是 thinking 内容**。

我去翻 [DeepSeek 官方文档](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode)，发现：

> 思考模式默认开启。可以通过 `extra_body={"thinking": {"type": "disabled"}}` 关闭。

也就是说，thinking 是**可关的**。我封装了一个 helper：

```python
def build_thinking_kwargs(thinking_mode: str) -> dict:
    """
    根据配置构造 messages.create() 的额外参数

    auto:     不传参数,用模型/网关默认
    enabled:  强制开
    disabled: 强制关
    """
    if thinking_mode == "auto":
        return {}
    return {
        "extra_body": {"thinking": {"type": thinking_mode}}
    }
```

调用时：

```python
response = client.messages.create(
    model=settings.anthropic_model,
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
    **build_thinking_kwargs(settings.anthropic_thinking),
)
```

实测对比同一个 prompt（"用 50 个汉字解释 ReAct"）：

| 模式 | output token | 答案质量 |
|------|:------------:|---------|
| disabled | ~50 | 直接给答案 |
| enabled | ~160 | 含 110 token thinking + 50 token 答案 |

> 💎 **生产规则**：Agent 系统**不要全局开 thinking**，按 task 复杂度切换：
> - 复杂规划 / 数学 / 代码 → enabled
> - 闲聊 / 翻译 / 格式转换 → disabled
>
> 大规模跑下来可以省 50-70% 成本。

---

## 真问题 4：同模型 + 两种协议 = 完全不同的字段位置

我同时用 Anthropic SDK 和 OpenAI SDK 调用了**同一个模型**（网关后端是 deepseek-v4-flash）：

```python
# Anthropic 协议
response_a = anthropic_client.messages.create(...)

# OpenAI 协议
response_o = openai_client.chat.completions.create(...)
```

Token 用量：

```
Anthropic 协议:  input=20, output=160  ← 含 thinking
OpenAI    协议:  input=20, output=66   ← 不含 thinking?
```

什么？**thinking 没了？**

不，thinking 还在，只是字段位置变了：

| 协议 | thinking 位置 | text 位置 |
|------|--------------|----------|
| Anthropic | `content[i]` 里 `type=="thinking"` 的 block | `content[i]` 里 `type=="text"` 的 block |
| OpenAI | `message.reasoning_content` | `message.content` |

我的代码原本只读了 `response.choices[0].message.content`，所以"丢失"了 thinking。但 token 还是按总数计费的 — **我"不知不觉"付了钱却没拿到内容**。

修复一行：

```python
def call_openai(prompt: str) -> dict:
    response = client.chat.completions.create(...)
    msg = response.choices[0].message
    return {
        "text": msg.content,
        "thinking": getattr(msg, "reasoning_content", "") or "",  # ⭐ 加这行
        "tokens": {...},
    }
```

> 💎 **教训**：写"通用"LLM 接入层时，**绝不能假设 SDK 自动统一**。每个协议每个版本都可能有独家字段。**用 `response.dict()` 完整 dump 一次，看一眼，胜过读 10 篇教程**。

---

## 真问题 5：并发不是越多越快 — 实测加速比 1.80x

我跑了 `asyncio.gather` 并发调用：

```python
async def main():
    # 串行
    start = time.perf_counter()
    r1 = await call_anthropic_async(prompt)
    r2 = await call_openai_async(prompt)
    serial_time = time.perf_counter() - start

    # 并发
    start = time.perf_counter()
    r1, r2 = await asyncio.gather(
        call_anthropic_async(prompt),
        call_openai_async(prompt),
    )
    parallel_time = time.perf_counter() - start

    print(f"加速比: {serial_time / parallel_time:.2f}x")
```

实测：

```
串行:3.64s
并发:2.03s
加速比:1.80x
```

理论极限是 2.00x（完全独立的两个网络调用），我拿到 90% 效率。**那 0.20x 损耗在哪?**

- DNS 解析 / TLS 握手
- 网关路由分发延迟
- Python asyncio 调度开销

到生产环境会发生什么？

- W11 旗舰项目里我会跑 4 个 Agent 并发，理论 4x，实测预期 3.5-3.8x
- 多模型 fallback 路由（同时打 Claude / GPT / DeepSeek 取最快）能省 30-50% TTFT
- **这就是 Multi-Agent 的核心价值** — 串行延迟堆叠，并发只看最慢的那个

顺便测了 Streaming 的 **TTFT（首字节时间）**：

| 协议 | TTFT | 总耗时 |
|------|:----:|:-----:|
| Anthropic | **1.77s** | 4.16s |
| OpenAI | 3.28s | 5.33s |

**同模型不同协议，TTFT 差近 2x**！可能原因是网关对 Anthropic 协议优化更好（预热连接、优先级调度等），但具体机制需要进一步排查。

> 💎 **TTFT 是 ChatGPT 体验的灵魂指标**。1.77s 和 3.28s 体感差别巨大。流式输出绝不是为了"炫酷"，是为了**用户感知到的等待时长**这个核心指标。

---

## Day 1 战绩

```
代码量      : ~600 行(Python + Go + 配置)
开源文件    : 6 个(config.py / utils.py / 3 demos / Go main)
跑通模型    : 1(deepseek-v4-flash,双协议)
踩坑解决    : 5 个核心问题
工程化收获  : 4 项(配置外置 / Helper / 类型校验 / 路径鲁棒)
关键数据    : 4 组(token / 加速比 / TTFT / Go usage)
学习投入    : ~4h
```

**所有代码开源**：[github.com/<your-github>/agent](https://github.com/) → `week-01/day-01-llm-basics/`

包含：
- `config.py` — pydantic-settings 配置层
- `utils.py` — `extract_text` / `extract_thinking` / `build_thinking_kwargs`
- `01_sync.py` / `02_async.py` / `03_stream.py` — 三种调用模式
- `go-demo/main.go` — Go 端实现（我的 Eino 路线起点）
- `notes.md` — Day 1 完整数据复盘

---

## 12 周路线图（免得你以为我只学 1 天）

| 周 | 主题 |
|:---:|------|
| W1 | LLM API 基础 ✅ Day 1 |
| W2 | Tool / Function Calling |
| W3-4 | Embedding + RAG 完整流程 |
| W5 | LangChain |
| W6 | LangGraph(Plan-Execute / Reflection)|
| **W7** | **Eino(字节 Go AI 框架)— Go 工程师的杀手锏** |
| W8 | MCP 协议 + 用 Go 实现 MCP Server |
| W9 | Multi-Agent 协作 |
| W10 | Eval 体系(Langfuse + A/B 测试) |
| W11 | 旗舰项目:智能导购 Agent 平台 |
| W12 | 面试冲刺 + 收 Offer |

每周 26 小时（工作日 2h × 5 + 周末 8h × 2）。

---

## 下一篇预告

**Day 2：Prompt Engineering 真的是"玄学"吗?**

会用同一个任务实测：
- Zero-shot vs Few-shot 准确率差多少
- CoT(思维链)对推理类任务的提升
- ReAct 模式手写实现(不用框架)

如果你也是后端 / 全栈，正在或想转 LLM / Agent 方向，**关注一下，我们 12 周一起走**。

GitHub Star 一个，我会更动力。下篇见 👋

---

## 互动 🗣️

留言区聊聊：
1. 你转型 / 想转型遇到的最大痛点是什么?
2. 你踩过哪些 LLM API 的坑(欢迎补充 #6 #7 #8 ...)
3. 你最想看 12 周里的哪一周(我可以提前剧透 / 调整顺序)

---

> *🤖 这是「5 年 Go 后端转 Agent」公开学习系列第 1 篇。*
> *👉 GitHub:[github.com/&lt;your-github&gt;/agent](https://github.com/)*
> *📚 12 周计划完整版:留言或私信发你 Notion 链接*
