---
day: W1D1
date: 2026-05-02
topic: LLM API 基础调用(同步 / 异步 / 流式 / Go)
hours: 2
status: done
tags: [agent, day1, llm-basics, deepseek, thinking-mode]
---

# Day 1 学习笔记 — LLM API 基础

> 第一天就撞到了**思考型模型 + 网关协议转换**两个真问题,数据丰富。

---

## 📊 今日实测数据

### 1️⃣ Token 消耗(同 prompt:50 字解释 ReAct)

| 协议 | input | output | 备注 |
|------|:-----:|:------:|------|
| Anthropic | 20 | 160 | 含 thinking |
| OpenAI | 20 | 66 | reasoning_content 字段(初版漏读,已修) |

**结论**:同模型(deepseek-v4-flash)同 prompt,**Anthropic 协议 output 是 OpenAI 的 2.4x**,差异在 thinking 是否被序列化进 content。**两边都付了 thinking 的钱**,只是字段位置不同。

---

### 2️⃣ 异步并发加速比

| 模式 | 耗时 |
|------|:----:|
| 串行(`await` 接力) | 3.64s |
| 并发(`asyncio.gather`) | 2.03s |
| **加速比** | **1.80x** |

**结论**:理论极限 2x,实测 90% 效率。损耗来自 DNS / TLS / 调度。**生产 5 模型并发能接近 5x**,这是 Multi-Agent 的核心价值。

---

### 3️⃣ 流式 TTFT(首字节时间)

| 协议 | TTFT | 总耗时 |
|------|:----:|:------:|
| Anthropic | **1.77s** | 4.16s |
| OpenAI | 3.28s | 5.33s |

**结论**:同模型不同协议,**Anthropic 协议 TTFT 快 1.85x**。可能原因:
- 网关对 Anthropic 协议有特殊优化
- OpenAI 协议要等 reasoning_content 完整生成才开始 stream(待验证)
- 内部路由优先级不同

> 💡 **TTFT 是 ChatGPT 类产品的核心 UX 指标**,1.77s vs 3.28s 体感差别巨大。

---

### 4️⃣ Go 同步调用

```
input=18, output=98(含 thinking)
```

成功遍历 `[]anthropic.ContentBlockUnion`,用 `extractText` / `extractThinking` 按 `block.Type` 过滤。**Go + Agent 路线打通起点**。

---

## 💡 三个核心洞察(写到博客 #1)

### 洞察 1:思考型模型的 content 是数组,不是字符串

```python
# ❌ 旧代码
text = response.content[0].text   # ThinkingBlock 没 .text → AttributeError

# ✅ 正确
text = "\n".join(b.text for b in response.content if b.type == "text")
```

**适用范围**:Claude 3.7+ / Claude 4 / DeepSeek-R1 / o1 / GPT-o3 等所有 reasoning 模型。

---

### 洞察 2:同模型不同协议,字段位置完全不同

| | thinking 位置 | text 位置 |
|---|---|---|
| Anthropic 协议 | `content[i]` 中 type=="thinking" 的 block | `content[i]` 中 type=="text" 的 block |
| OpenAI 协议 | `message.reasoning_content` | `message.content` |

**踩坑教训**:写"通用"LLM 接入层时,**不能假设 SDK 会自动统一**,要分别处理。

---

### 洞察 3:Thinking 可以通过 extra_body 透传开关

```python
# DeepSeek 风格(网关支持透传)
extra_body={"thinking": {"type": "disabled"}}  # 关
extra_body={"thinking": {"type": "enabled"}}   # 开
```

**生产规则**:Agent 系统**不要全局开 thinking**,按 task 复杂度切换。
- 复杂规划 / 数学 / 代码 → enabled
- 闲聊 / 翻译 / 格式转换 → disabled

---

## 🛠️ 工程化收获

| 实践 | 实现 | 价值 |
|------|------|------|
| 配置外置 | `config.py` + `pydantic-settings` | 切模型 / 切代理零代码改动 |
| Helper 抽离 | `utils.py`(`extract_text` / `build_thinking_kwargs`)| 12 周复用 |
| 类型校验 | `Literal["auto", "enabled", "disabled"]` | 启动时报错,不到运行时 |
| 路径鲁棒 | `Path(__file__).parents[2]` 自动定位 .env | 不同目录运行都 OK |

---

## 🔜 留给以后的疑问

1. Thinking 的 token 按 input 还是 output 计费?具体倍数?
2. 网关协议转换的具体机制?(Anthropic in → DeepSeek OpenAI out 怎么映射 ToolUseBlock?)
3. 如果模型本身不支持 thinking 参数,传 disabled 会报错还是被忽略?
4. Anthropic 协议 stream 比 OpenAI 协议快近 2x 是普遍现象还是网关特例?

> 💡 **W2 Tool Calling** 时多半会撞到 1 和 2,W4 RAG 时撞到 3,W7 Eino 时撞到 4。**留着,不急于现在搞懂**。

---

## 📌 博客 #1 素材清单

- **标题候选**:《5 年 Go 后端转 Agent · Day 1:200 行代码踩中 LLM 工程化的 5 个真问题》
- **数据钩子**:1.80x 并发加速 / TTFT 1.77s vs 3.28s / Anthropic 输出比 OpenAI 多 2.4x
- **故事钩子**:从 `'ThinkingBlock' object has no attribute 'text'` 讲起
- **核心知识点**:配置外置 / extract_text / 协议差异 / thinking 开关
- **代码 link**:`week-01/day-01-llm-basics/`(GitHub 公开)

---

## ✅ Day 1 完成清单

- [x] 环境(Python 3.11 + Go 1.26 + uv)
- [x] 仓库结构 + .gitignore
- [x] config.py(pydantic-settings)
- [x] utils.py(extract_text / extract_thinking / build_thinking_kwargs)
- [x] 01_sync.py(Anthropic + OpenAI 协议双向)
- [x] 02_async.py(asyncio.gather)
- [x] 03_stream.py(TTFT 测量)
- [x] go-demo/main.go(Go 杀手锏起点)
- [x] OpenAI reasoning_content 字段读取(自己修)
- [x] Day 1 数据复盘 + 笔记
- [ ] git commit + push
- [ ] Obsidian 勾选 W1D1
