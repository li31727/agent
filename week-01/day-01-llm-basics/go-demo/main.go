// Week 1 - Day 1 - Step 4: Go 同步调用 LLM
// =========================================
//
// 🎯 学习目标:
//   1. 用 Go 调用 LLM API — 这是你 5 年后端转 Agent 的杀手锏
//   2. 携程① JD 明确要求 Go + Agent,这就是入门
//   3. 后续 W7 学的 Eino 框架,底层就是这套调用 + 抽象
//   4. 配置全部从 .env 读取(model / base_url / api_key)
//   5. ⭐ 处理思考型模型的 ThinkingBlock(遍历 content 按 Type 过滤)
//
// ▶️ 运行方式(在仓库根目录 /Users/lileibiao/code/agent):
//    go run ./week-01/day-01-llm-basics/go-demo
//
// 📝 课后思考(写到 Obsidian Day 1 笔记):
//   - Go 的并发(goroutine)对比 Python 的 asyncio,
//     在 Multi-Agent 场景下哪个更合适?
//   - 为什么 Anthropic / OpenAI 直到 2024+ 才推官方 Go SDK?

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/joho/godotenv"
)

// getEnv 读取环境变量,如果不存在或为空则返回默认值
func getEnv(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

// extractText 从 message content 中提取纯文本,过滤掉 thinking / tool_use 等
//
// 为什么需要这个函数?
// 思考型模型(DeepSeek-R1 / o1 / Claude 4)返回的 Content 是多 block:
//   - ThinkingBlock: 推理过程(type="thinking",对应 .Thinking 字段)
//   - TextBlock:     最终答案(type="text",对应 .Text 字段)
//   - ToolUseBlock:  工具调用(type="tool_use")
// 直接取 Content[0].Text 在思考型模型上会拿到空字符串。
func extractText(content []anthropic.ContentBlockUnion) string {
	var sb strings.Builder
	for _, block := range content {
		if block.Type == "text" {
			sb.WriteString(block.Text)
		}
	}
	return sb.String()
}

// extractThinking 提取模型的 thinking 过程(调试/学习用)
func extractThinking(content []anthropic.ContentBlockUnion) string {
	var sb strings.Builder
	for _, block := range content {
		if block.Type == "thinking" {
			sb.WriteString(block.Thinking)
		}
	}
	return sb.String()
}

func main() {
	// 尝试从多个路径加载 .env(无论从哪个目录跑都能找到)
	_ = godotenv.Load(".env", "../.env", "../../.env", "../../../.env")

	// ===== 从 .env 读配置 =====
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	baseURL := os.Getenv("ANTHROPIC_BASE_URL")              // 可选
	model := getEnv("ANTHROPIC_MODEL", "claude-sonnet-4-5") // 可选,有默认值

	if apiKey == "" {
		log.Fatal("❌ ANTHROPIC_API_KEY 未设置,请检查 .env 文件")
	}

	// ===== 构建客户端选项 =====
	opts := []option.RequestOption{option.WithAPIKey(apiKey)}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client := anthropic.NewClient(opts...)

	// ===== 调用 messages API =====
	msg, err := client.Messages.New(context.TODO(), anthropic.MessageNewParams{
		Model:     anthropic.Model(model), // 字符串转换为 Model 类型
		MaxTokens: 1024,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("用 50 个汉字解释什么是 Tool Calling(工具调用)"),
			),
		},
	})
	if err != nil {
		log.Fatalf("❌ Anthropic API 调用失败: %v", err)
	}

	// ===== 输出结果(注意要遍历 content,处理 thinking 模型)=====
	fmt.Printf("=== 🟪 Anthropic [%s] (Go) ===\n", model)

	if thinking := extractThinking(msg.Content); thinking != "" {
		fmt.Println("💭 [模型 thinking 过程]")
		fmt.Println(thinking)
		fmt.Println(strings.Repeat("─", 40))
	}

	fmt.Println("📤 [最终答案]")
	fmt.Println(extractText(msg.Content))

	fmt.Printf("\n=== 📊 Token 用量 ===\n")
	fmt.Printf("Input:  %d\n", msg.Usage.InputTokens)
	fmt.Printf("Output: %d\n", msg.Usage.OutputTokens)
	fmt.Printf("Total:  %d\n", msg.Usage.InputTokens+msg.Usage.OutputTokens)
}
