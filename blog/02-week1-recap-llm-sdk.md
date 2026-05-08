---
title: 5 年 Go 后端转 Agent · Week 1 复盘:6 天踩了 11 个真坑,Token 直降 67%(SDK 已开源)
date: 2026-05-07
tags: [LLM, Agent, Python, Go, DeepSeek, Anthropic, 转型, Pydantic, Instructor, ReAct]
description: 5 年 Go 后端工程师 12 周转型 Agent 工程师的 Week 1 全程复盘。从撞 ThinkingBlock 报错到封装可复用 SDK,Token 消耗直降 67%。每个数据点都有实测,每个金句都来自真实代码。
---

# 5 年 Go 后端转 Agent · Week 1 复盘:6 天踩了 11 个真坑,Token 直降 67%(SDK 已开源)

> 这是一篇**公开学习日记**。我是 5 年 Go 后端工程师,正在用 12 周时间转型 Agent 工程师。Week 1 已经过完。
>
> 跟那些"15 分钟入门 LLM"的教程不同,这篇文章把**6 天踩到的 11 个真问题**摊开来写,**带数据、带代码、带 debug 过程**。
>
> 最大的收获不是写了多少代码,而是把所有踩坑封装成一个 **可复用 SDK(llm_kit)**,然后用同一份 demo 对比,**Token 消耗直降 67%**。
>
> 全部代码开源:[github.com/&lt;your-github&gt;/agent](https://github.com)

## TL;DR — Week 1 战绩

| 维度 | 数据 |
|------|------|
| 学习时长 | 26h(工作日 2h × 5 + 周末 8h × 2) |
| 跑通 demo | **18 个** |
| 开源代码量 | **~3500 行**(Python + Go + 配置) |
| 知识沉淀 | 6 篇 notes + 56 道 Q&A |
| 踩坑解决 | **11 个真实生产问题** |
| 自己 debug 出来的 | **3 个文档不写的隐藏坑** |
| 整合产出 | **1 个完整 SDK(llm_kit)** |
| 同 demo Token 减少 | **67%**(6759 → 2256) |

---

## 为什么是 12 周?

故事很简单。

我在做 C 端导购插件,最近扒了 4 份目标公司的 Agent 工程师 JD(携程 ×2 / 智慧树 / 拓竹),做了一个能力交集:

- ✅ 我已有的:Python / Go / K8s / Redis / 微服务 / 分布式
- ❌ 我要补的:LLM 实战 / RAG / Agent 框架 / Multi-Agent / Eval 体系 / MCP

差距很明确。我给自己定了 **12 周转型计划**:

- 每周 1 篇博客(你看的就是 #1)
- 12 周 5 个开源项目
- **边学边投递**(W5 起)

Week 1 主题是 **LLM API 基础 + Prompt Engineering**,7 天计划:

| Day | 主题 | 实际产出 |
|:---:|------|---------|
| 1 | LLM API 基础(同步/异步/流式) | 4 个 demo + Go 端起步 |
| 2 | Prompt 基础(Zero/Few-shot / 指令清晰度 / 角色) | 3 个对比 demo |
| 3 | CoT + Self-Consistency | 反直觉发现 |
| 4 | Tool Calling 实战 | 4 个文件 + 完整 Agent |
| 5 | 结构化输出(Pydantic + Instructor) | 3 个递进 demo |
| 6 | **SDK 整合**(本周精华) | llm_kit 包 + 67% 节省 |
| 7 | 写博客 + push GitHub | 你正在读的这篇 |

---

## Day 1:第一行代码就给了我一记重锤

我跑了一个最简单的同步调用 demo:

```python
def call_anthropic(prompt: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "text": response.content[0].text,  # ⚠️ 这一行
        ...
    }
```

报错:

```
AttributeError: 'ThinkingBlock' object has no attribute 'text'
```

什么是 ThinkingBlock?

查了一下,**思考型模型**(Claude 4 / DeepSeek-R1 / o1 等)的响应是这样的:

```python
response.content = [
    ThinkingBlock(thinking="让我想想..."),
    TextBlock(text="ReAct 是..."),
    ToolUseBlock(...),
]
```

`content[0]` 不一定是文本,可能是模型的"内心独白"。

正确写法是**遍历 + 按 type 过滤**:

```python
def extract_text(response) -> str:
    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )
```

> 💎 **教训**:看到 `response.content[0].text` 这种写法的教程,直接关掉。它在思考型模型上一定崩。

Day 1 还撞了几个其他坑(协议字段差异 / 并发加速比 / TTFT 经济学),细节我写在另一篇博客里:[《200 行代码踩中的 5 个 LLM 工程化真问题》](./01-day1-five-real-problems.md)。

**Day 1 实测数据**:
- 异步并发加速比 **1.80x**(理论 2x,90% 效率)
- 流式 TTFT:Anthropic **1.77s** vs OpenAI **3.28s**(同模型不同协议差近 2x)

---

## Day 2:Prompt Engineering 不是玄学,有真数据

跑了三个对比实验,**数据说话**:

### 实验 1:Zero-shot vs Few-shot(10 条评论分类)

| 方法 | 准确率 |
|------|:----:|
| Zero-shot(纯 prompt) | 9/10 = **90%** |
| Few-shot(3 examples) | 10/10 = **100%** |

**唯一错的那条**:"外观还行,功能一般,价格略贵"
- Zero-shot 误判为「负面」(因为"略贵"是负面词)
- Few-shot 救活了它

为什么?Few-shot 给了一个**结构高度对应**的例子("外观一般 + 没什么惊喜 → 中性"),模型立刻 anchor 到"中性"。

> 💎 **金句**:Few-shot 的核心价值不是"教模型分类",是**"给边界 case 一个 anchor"**。

### 实验 2:模糊 vs 清晰指令(token 经济学)

同样的提取任务:

| | Prompt 字符 | 输出字符 | 总字符 | JSON 解析 |
|---|---:|---:|---:|:---:|
| 模糊指令 | 183 | 900(markdown 表格) | 1083 | ❌ |
| 清晰指令 | **716** (+533) | **351** (-549) | **1067** | ✅ |

**多花 533 prompt 字符,省下 549 输出字符**,总字符基本持平,而 JSON 解析率从 0% 提升到 100%。

> 💎 **金句**:多花 prompt token 换 deterministic 输出,**几乎总是划算的**。

### 实验 3:角色 × 格式矩阵

同样问题:"500 元以内的母亲节礼物推荐"

| 角色 | 输出字数 | 推荐数 | 风格 |
|------|:---:|:---:|------|
| 默认无角色 | ~600 | 13 项(撒胡椒) | 不聚焦 |
| 中立购物顾问 | ~500 | 6 + 避坑 | 理性 |
| **10 年资深买手** | **~300** | **3 精选** | **有情感钩子** |

最有趣的是,资深买手主动建议加"手写卡片:感谢你让家如此温暖" — 默认 prompt 不会有这种细节。

> 💎 **金句**:角色 prompt 不是"让模型扮演",而是**"给模型一个语料分布锚点"**。"10 年经验"这种数字描述比"经验丰富"更有效。

---

## Day 3:反直觉 — **关掉 thinking 反而最强**(8.5x 成本陷阱)

跑 CoT vs Direct 对比时,发现一个超出预期的数据。

设置 4 种组合(Thinking × CoT 矩阵):

```
                 Direct prompt    CoT prompt
                 ────────────────────────────
thinking=disabled │  A: 57t,0.6s   B: 244t   │
                 ────────────────────────────
thinking=enabled  │  C: 210t      D: 482t,5.49s
                 ────────────────────────────
```

跑同一道电商优惠题,**4 个组合全部正确**:

| 配置 | tokens | 耗时 | 相对 A |
|------|:----:|:----:|:----:|
| **A. disabled + Direct** ⭐ | **57** | **0.60s** | **1.0x** |
| B. disabled + CoT | 244 | 2.42s | 4.3x |
| C. enabled + Direct | 210 | 2.30s | 3.7x |
| D. enabled + CoT | 482 | 5.49s | **8.5x** |

**A 比 D 便宜 8.5 倍 / 快 9 倍,准确率却完全一样**。

这反直觉的本质是:
- 思考型模型不等于"必须开 thinking"
- thinking 是**安全网**(对难题有用),不是**放大器**(简单题没用)
- **默认开 thinking = 给所有任务交"安全税"**

> 💎 **金句**:**永远先用最便宜配置 baseline,只在准确率不达标时升级**。Demo 3 的 8.5x 成本浪费就是反例。

而 **Self-Consistency 在思考型模型上几乎死了** — 跑 5 次 temp=0.7,**全部 254.4**(标准差 0)。论文里 +17% 的红利,在 2026 年的强模型 + 三步算术上**完全消失**。

> 💎 **金句**:**不要无脑套论文方法**。先测 baseline 的 variance,数据驱动决策。

---

## Day 4:Tool Calling 实战 + 自己 debug 出 2 个隐藏坑

原本计划是"手写 ReAct(不用框架)",但**5 年后端老兵的本能**让我直接问:

> "为什么不用 Tool Calling?直接学 SDK 原生 API 吧。"

跳过手写,直接做 Tool Calling 实战(因为 ReAct 论文 2022 的"手写 prompt 解析",2023+ 已被 SDK 原生 Tool Calling 完全替代)。

### 隐藏坑 1:reasoning_content 续写适配

跑 Tool Calling 时,第一次工具调用成功,但**第二次续写**报 400。

API 报错指向 `messages` 参数。我打印调试发现:DeepSeek 思考型模型首轮返回会带 `reasoning_content` 字段,**续写时必须把它原样带回**,否则 API 拒绝。

```python
# 自己 debug 出来的修复
assistant_turn = {
    "role": "assistant",
    "content": msg.content,
    "tool_calls": [...],
}
reasoning = getattr(msg, "reasoning_content", None)
if reasoning is not None:
    assistant_turn["reasoning_content"] = reasoning  # ⭐ 关键
messages.append(assistant_turn)
```

**这个坑文档里几乎不写**。教程都没提,但实际生产中思考型模型 + Tool Calling 必撞。

### 隐藏坑 2:Tool Schema 不加 enum,模型就敢瞎编

Demo 3 同一个问题,Anthropic 协议下模型给了 `category="智能手环"` —— **我们的数据库里根本没这个类别**(只有"手环")。

修复一行:

```python
"category": {
    "type": "string",
    "enum": ["手环", "智能手表", "香氛", "花艺礼品", "女士手表"]  # ⭐ 必须
}
```

加了 enum 后,模型只能从枚举里选,不会瞎编。

> 💎 **金句**:**Tool Schema 不加 enum,模型就敢瞎编参数**。一行修复一个 production bug。

### ReAct loop 30 行 = LangGraph / Eino 的核心

```python
while not done:
    response = llm.call(messages, tools)
    if response.has_tool_calls:
        results = execute_in_parallel(response.tool_calls)
        messages += [response, *results]
    else:
        return response.content
```

**Day 4 实测**:推荐 500 元礼物任务,4 轮调用 / 11 次工具调用 / 6759 tokens / 21.9 秒。

模型展现了**分治 + 并行**思维:
- Round 1:3 个并行 search(撒网)
- Round 2:4 个并行 get_product_info(聚焦)
- Round 3:4 个并行 get_reviews(验证)
- Round 4:综合输出

> 💎 **金句**:理解了你的 30 行 loop,**LangGraph 文档秒懂** — 它只是在此基础上加了状态机 / Checkpoint / HITL。

---

## Day 5:**LLM 没有"现在"的概念**(超出预期的反直觉)

学结构化输出时,跑了三种方法对比:

| Method | sentiment 字段 | 状态 |
|--------|------|:----:|
| A 纯 prompt | "中性" ✅ | 完美 |
| B json_object | **"mixed"** ❌ | JSON 合法但语义跑偏 |
| C json_schema (strict) | — | ❌ 网关不支持 |

**Method B 的"mixed"翻车** 是 Day 5 最值钱的发现 — `response_format=json_object` 只保证 JSON 合法,**不保证字段符合预期 schema**。

后端代码 `data['sentiment'] in ['正面','中性','负面']` 仍可能崩。

> 💎 **金句**:**json_object 是伪结构化** — JSON 合法但语义随机。

### Pydantic + Instructor 救场

切到 Pydantic + Instructor 后,5 个字段全部命中,0 次重试一次过。

但**踩了第 3 个隐藏坑**:Instructor 默认 `Mode.TOOLS` 在国内网关必死(用 `tool_choice="required"`,DeepSeek 不支持)。

```python
# 自己 debug 出的修复
client = instructor.from_openai(
    base_client,
    mode=instructor.Mode.JSON  # ⭐ 国内网关唯一可用
)
```

> 💎 **金句**:**Mode.JSON 是国内网关唯一可用配置**。这条经验文档不写,但坑过我。

### 最反直觉的发现:嵌套订单测试

复杂订单提取(6 层校验链),3/3 全部一次过。但 `order_id` 字段:

```
今天是 2026-05-04,但模型给了:
  ORD-2024-001  ❌
  ORD-2025-001  ❌
```

**LLM 不知道当前日期**!system prompt 写"YYYY 是当前年份"完全无效。

为什么?
- LLM 没有"现在"的概念
- 训练数据截止决定它认为的"现在"
- system prompt 无法让模型 introspect 当前时间

**生产对策**:
```python
# 必须从外部注入
from datetime import datetime
system_prompt = f"""...
当前日期:{datetime.now().strftime('%Y-%m-%d')}
订单号格式:ORD-{datetime.now().year}-NNN
"""
```

> 💎 **金句**:**LLM 没有"现在"的概念**。任何依赖当前时间/日期的字段,**必须从外部注入**。这条教程几乎不写,但你做日历助手 / 订单系统必踩。

---

## Day 6:**SDK 整合 → Token 直降 67%**(Week 1 真正的高潮)

到 Day 6,我手上的零散资产:
- 17 个 demo 文件
- 多次重复的"修复代码"(reasoning_content / Mode.JSON / extra_body 等)
- 每写新 demo 都要复制粘贴这些 hack

**这就是经典的"散户工程"反模式** — 每个 demo 都是孤岛。

我用 Day 6 的 8 小时把所有 hack 封装成 `llm_kit` 包:

```
src/llm_kit/
├── __init__.py            # 统一导出
├── client.py              # LLMClient(整合 thinking + reasoning_content)
├── agent.py               # ReActAgent(30 行 loop 工程化)
├── structured.py          # StructuredExtractor(Mode.JSON + 重试)
└── exceptions.py          # 自定义异常体系(LLMError + 5 子类)
```

### SDK 内置的 6 个"已踩坑"全部自动化

| # | 踩坑修复 | 原代码痛点 | SDK 化后 |
|---|---------|----------|---------|
| 1 | 思考模式开关 | 每次写 `extra_body` | `thinking=` 参数 |
| 2 | reasoning_content 续写 | 每次手动 `getattr` 拼回 | `make_assistant_message()` 自动 |
| 3 | Mode.JSON 兼容国内网关 | 每次记得切 | StructuredExtractor 默认开 |
| 4 | MAX_ITERATIONS 防死循环 | 每次写 while 防护 | ReActAgent 内置 |
| 5 | Pydantic 失败重试 | 每次写 try/except | extract() 内置 |
| 6 | 统一返回结构 | 每次解 ChatCompletion 嵌套 | dict {text/thinking/tool_calls/tokens} |

### 用 SDK 重写 Day 1 / 4 / 5 demo

```python
# 基础聊天 — 4 行
client = LLMClient(api_key=..., model=..., thinking="disabled")
response = client.chat([{"role": "user", "content": "你好"}])

# ReAct Agent — 5 行
agent = ReActAgent(client, tools=TOOLS, tool_executor=execute_tool)
result = agent.run("帮我推荐礼物", verbose=True)

# 结构化输出 — 4 行
extractor = StructuredExtractor(api_key=..., base_url=..., model=...)
review: ReviewAnalysis = extractor.extract(
    response_model=ReviewAnalysis, messages=[...]
)
```

### 出乎意料的收获:**Token 直降 67%**

跑同一个"推荐 500 元礼物"任务,SDK 版 vs Day 4 原版:

| 指标 | Day 4 原版 | **Day 6 SDK** | 改善 |
|------|:---:|:---:|:---:|
| 实际轮数 | 4 | **3** | -25% |
| 工具调用次数 | 11 | **4** | **-64%** |
| Token 总消耗 | 6759 | **2256** | **-67%** ⭐ |

我以为 SDK 只是"代码组织优化",**没想到顺便把成本砍了 67%**。

为什么?三个因素叠加:

1. **`OPENAI_THINKING=disabled` 通过 SDK 自动透传**(最大功臣)
   - .env 改一行 → SDK 自动应用 → 全项目所有 LLM 调用受益
2. system_prompt 从 4 步简化为 3 步
3. 工具集精简(去掉 reviews 工具)

> 💎 **金句**:**配置外置的复利效应** — `.env` 改一行,SDK 自动放大,成本降 67%。

### 代码量对比

```
Day 1 01_sync.py:        110 行  →  10 行 SDK 调用  (-91%)
Day 4 02_react_loop.py:  170 行  →  25 行 SDK 调用  (-85%)
Day 5 02_pydantic.py:    160 行  →  15 行 SDK 调用  (-91%)
─────────────────────────────────────────────────────
合计 440 行  →  50 行                 -89%
```

**业务代码只关心"问什么 / 答什么",不关心 thinking / 续写 / Mode.JSON 这些"管道工细节"**。

> 💎 **金句**:这就是"工程化"的 ROI — **一次封装,N 次复用**。Day 1-5 的"投资",在 Day 6 开始连本带利收回。

---

## 6 个工程化金句(Week 1 提炼)

1. **配置外置的复利效应** — `.env` 改一行,SDK 自动放大,成本降 67%
2. **永远先用最便宜配置 baseline**,只在准确率不达标时升级
3. **Tool Schema 不加 enum,模型就敢瞎编参数** — 一行修复一个 production bug
4. **LLM 没有"现在"的概念** — 任何依赖时间的字段必须从外部注入
5. **30 行 ReAct loop = LangGraph / Eino 的核心** — 理解底层,框架文档秒懂
6. **Mode.JSON 是国内网关唯一可用配置** — Mode.TOOLS 在 DeepSeek 系网关必死

---

## 11 个 Week 1 真问题清单(直接生产可用)

| # | 真问题 | 修复方式 |
|:---:|------|---------|
| 1 | ThinkingBlock 没有 .text 属性 | `extract_text(response)` 遍历过滤 |
| 2 | OpenAI / Anthropic 协议字段位置完全不同 | 双协议 helper |
| 3 | json_object mode 必须 prompt 含 'json' 字样 | prompt 显式包含关键词 |
| 4 | json_object 是伪结构化(语义随机) | 升级到 Pydantic + Instructor |
| 5 | strict json_schema 国内网关不支持 | 默认假设不支持,客户端补校验 |
| 6 | DeepSeek 思考模型 reasoning_content 续写必带回 | `make_assistant_message()` 自动 |
| 7 | Instructor 默认 Mode.TOOLS 撞 tool_choice 限制 | 切 Mode.JSON |
| 8 | Tool Schema 不加 enum 模型瞎编 | 必须用 Literal / enum |
| 9 | Self-Consistency 红利在强模型上几乎消失 | 先测 variance 再决定 |
| 10 | 默认开 thinking = 8.5x 成本陷阱 | 复杂度路由,默认 disabled |
| 11 | LLM 不知道"今天几号" | 当前日期从外部注入 |

---

## 开源 SDK:llm_kit

把所有 Week 1 的踩坑修复 + 工程化经验封装在一起:

- **GitHub**:[github.com/&lt;your-github&gt;/agent](https://github.com)
- **路径**:`src/llm_kit/`
- **大小**:~600 行核心代码
- **依赖**:openai / pydantic / instructor

**用法**:

```python
from llm_kit import LLMClient, ReActAgent, StructuredExtractor

# 1. 普通聊天
client = LLMClient(api_key="...", base_url="...", model="...", thinking="disabled")
response = client.chat([{"role": "user", "content": "你好"}])

# 2. ReAct Agent
agent = ReActAgent(client, tools=TOOLS, tool_executor=execute_tool)
result = agent.run("帮我推荐礼物")

# 3. 结构化输出
extractor = StructuredExtractor(api_key="...", base_url="...", model="...")
review: ReviewAnalysis = extractor.extract(
    response_model=ReviewAnalysis,
    messages=[...]
)
```

**SDK 自动处理**:
- 思考模式开关(`thinking="enabled" | "disabled" | "auto"`)
- reasoning_content 续写适配(避免 400)
- Mode.JSON 兼容国内网关
- ReAct loop 防死循环
- Pydantic 校验失败自动重试

---

## Week 2 预告

下周主题:**Tool / Function Calling 进阶 + 多工具编排**。

- 多工具并行调用(asyncio.gather → 实测能否压到 5s 以内)
- 工具组合链(搜索 → 分析 → 总结)
- 接入真实 API(替换模拟商品库为真实导购 API)
- 项目 ① 第一个旗舰开源:`go-shopping-agent v0.1`(目标 50+ Star)

如果你也是后端 / 全栈,正在或想转 LLM / Agent 方向:

- ⭐ **GitHub Star** 给一个,我会更动力
- 📌 **关注**,下周 #2 会聊"多工具协作 + 真实场景"
- 💬 **评论区聊聊**,你转型 / 想转型遇到的最大痛点是什么?
- 📨 **简历互助**,Agent 工程师方向的 JD / 经验交换

---

## 12 周路线图(免得你以为我只学 1 周)

| 周 | 主题 | 状态 |
|:---:|------|:---:|
| W1 | LLM API 基础 + Prompt | ✅ 本周 |
| W2 | Tool / Function Calling 进阶 | 下周 |
| W3-4 | RAG 完整流程 | 计划中 |
| W5 | LangChain | 投递起点 |
| W6 | LangGraph(Plan-Execute / Reflection) | |
| **W7** | **Eino(字节 Go AI 框架)** | 🔥 Go 工程师杀手锏 |
| W8 | MCP 协议 + 用 Go 实现 MCP Server | |
| W9 | Multi-Agent 协作 | |
| W10 | Eval 体系(Langfuse + A/B 测试)| |
| W11 | 旗舰项目:智能导购 Agent 平台 | |
| W12 | 面试冲刺 + 收 Offer | 收官 |

---

## 互动 🗣️

留言区聊聊:

1. 你踩过哪些 LLM API 的坑?(欢迎补充 #12 #13 ...)
2. 你转型 / 想转型遇到的最大痛点?
3. SDK 你最想看哪个 v0.2 功能?(异步 / 流式 / Anthropic 原生 / Go 版本?)

---

> *🤖 这是「5 年 Go 后端转 Agent」公开学习系列博客 #1。*
> *👉 GitHub:[github.com/&lt;your-github&gt;/agent](https://github.com)*
> *📚 12 周计划完整版 + 56 道 Q&A 知识回顾:留言或私信发你 Notion 链接*
> *🔔 下周 #2:《多工具编排 + 真实 API 接入,做一个能用的导购 Agent》*
