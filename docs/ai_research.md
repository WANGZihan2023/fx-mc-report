# AI 检索员：模仿人工一条条抓新闻

## 先前约定（回忆）

对话里问过：「可以放一个 AI 在里面人工搜索吗？」  
约定是：流水线步骤 3 加 **AI 检索员**——像人一样按货币对信息需求去搜公开源，把投行/宏观展望收成带真链接的语句，供赋权与 References。不破解付费墙。

## 现在怎么工作

| 角色 | 谁 | 干什么 |
|------|-----|--------|
| **脑** | DeepSeek / Groq / Ollama 等 LLM | 拟下一句搜索词 → 看结果 → 再拟下一句；挑选有用标题；收成展望语句 |
| **手** | Tavily / Brave / NewsAPI / Google News RSS | 真正上网搜，返回带 URL 的结果 |
| **白名单** | ING / MUFG / StoneX 等公开页 | 无搜索 Key 也能直接抓（打不开就跳过） |

流程（迭代，默认最多 4 轮）：

```text
白名单页
  → 脑：下一句 query？
  → 手：执行搜索
  → 脑：这条留不留？
  → … 直到够用或脑说停
  → 脑：收成语句（URL 必须来自刚才搜到的链接）
```

诚实规则：

- **不发明 URL**；References 只保留搜索/白名单返回的链接  
- 只填 DeepSeek、没有 Tavily/Brave/NewsAPI：仍可用白名单 + **免费 Google News RSS**，并在审计里写明限制  
- `template_policy=off` 时仍不静默灌模板证据

## 怎么打开

1. Streamlit 侧栏 **② 抓取与判定** → 勾选 **「AI 检索员」**（默认开）  
2. CLI：默认开；关掉用 `--no-ai-research`  
3. 填 Key（「API 配置」或 Railway Variables / vault `.env`）：

| Key | 作用 |
|-----|------|
| `DEEPSEEK_API_KEY`（或 Groq / Ollama） | 脑 |
| `TAVILY_API_KEY` 或 `BRAVE_SEARCH_API_KEY` | 手（强烈建议） |
| `NEWSAPI_KEY` | 手 + 常规新闻头条 |

推荐组合：**DeepSeek + Tavily**。有 Tavily/Brave 时默认约 **10 轮 / target_keep≈90**，配合侧栏「最多头条证据条数」默认约 **80**（滑块 3–120），Live 报告**上限/目标约 100** 条可展示 References（仍不编造 URL；实际条数取决于源命中与配额，不保证每次满额）。无搜索 Key 时默认约 30，可能很少。历史回放默认省钱：不烧 Tavily、不强制冲到 100。  
代码：`fx_report/news/ai_research.py`，由 `pipeline.step3_collect_and_store_statements` 调用。

## References / 证据库呈现

PDF/HTML **Evidence Base · 证据库** 每条显示：

1. 证据 ID + 强弱标签 + 标题  
2. **支撑引用**：从文章/snippet 里挑出能支撑 Higher/Lower/Context 判定的句子（加息、AUD 走弱、RBA/Fed、铁矿石、中国需求等），**不是** SEO 导语或随便首句  
   - 优先字段 `support_quote`（流水线在摘要层按方向/类别抽取）  
   - 弱匹配或只剩标题时标 **「支撑引用（弱）」**  
   - **不编造**原文没有的句子  
3. 来源链接（优先稳定 URL；Google News 跳转尽量解析；软探测 404 则去掉超链并标「链接可能失效」）

诚实约定不变：不发明 URL、不用模板灌水凑条数。历史 cheap 路径无 Tavily，仍从 GDELT snippet / 标题等已有文本抽取。
