# 多货币对 · 最高日高蒙特卡洛情报报告

## 设计目标

**不同货币对的影响因子不同** → 必须先列出「本对需要什么信息」，再抓取与分析。  
**任意 BASE/QUOTE** 都可走同一套七步（目录精调 + 币种自动推断）。

```text
1. 选择货币对          → 锁定该对的影响因子
2. 评估所需信息        → 按因子展开信息需求表（AUD≠EUR≠CNH）
3. 存储有影响语句      → 官方 RSS + **AI 检索员**（投行公开页/搜索）→ 供赋权 / References / 数学分析
4. 评估对货币对的影响
5. 赋予权重
6. 数学分析（蒙特卡洛分档）
7. 规范格式报告
```

## 目录结构

```text
.
├── app.py                 # Streamlit 入口（Cloud 部署指向此文件）
├── run_cli.py             # 命令行入口
├── requirements.txt
├── runtime.txt
├── fx_report/             # 业务包
│   ├── pipeline.py        # 七步流水线编排
│   ├── config/            # API / vault 配置
│   ├── market/            # 货币对、驱动因子、行情抓取
│   ├── news/              # 新闻抓取 / 分类 / LLM / AI 检索
│   ├── model/             # 权重、强度评分、蒙特卡洛
│   ├── report/            # 报告与 diagnostics 文本
│   └── ui/                # Streamlit 页面与 API 面板
├── scripts/               # 一键启动 / 分享 / 部署
├── docs/                  # 分享说明等
└── output/                # 运行产物（git 忽略）
```

影响因子词表：`fx_report/market/pair_drivers.py`  
流水线：`fx_report/pipeline.py`  
AI 检索员：`fx_report/news/ai_research.py`（`--no-ai-research` 可关）

| 货币对 | 典型影响因子 |
|--------|--------------|
| USD/AUD | 地缘、油、Fed、RBA、铁矿、CPI |
| EUR/USD | Fed、ECB、CPI、地缘、增长 |
| USD/JPY | Fed、BOJ、利差、地缘、CPI |
| USD/CNH | Fed、PBOC、中国增长、地缘、利差 |

## 运行

```bash
# CLI
python run_cli.py --pair USD/AUD --sims 50000 --mode rules
python run_cli.py --pair EUR/USD --sims 50000 --mode hybrid
python run_cli.py --pair EUR/JPY --sims 50000 --mode rules --no-ai-research

# UI
streamlit run app.py
# 或双击 scripts/open.command（macOS）/ scripts/open.bat（Windows）
```

## 数据源

权威优先：ECB 行情 + 央行 RSS；可选 FRED/NewsAPI/inbox。  
步骤3可开 AI 检索员：公开投行页 +（可选）Tavily/Brave/NewsAPI Key，再经 LLM 收成展望语句。

### 网站上填 Key

`streamlit run app.py` → **「API 配置」**：

1. 免费申请指引（FRED / NewsAPI / …）
2. 付费/增强 Key 表（空着跳过）
3. AI API（Ollama / Groq / OpenAI 兼容）
4. 可「保存到本机 vault `.env`」（勿提交 git）

Key 默认读：`/Users/wangzihan/Desktop/工作_汇率/fx_data_apis/.env`（可用环境变量 `FX_API_ENV_PATH` 覆盖）。

## 输出

主交付物为 **Torchcast 风格 PDF / HTML**（WeasyPrint 渲染，封面概率条、上行/下行、执行摘要、证据库、What to Watch），落在 `output/{PAIR}_torchcast.pdf`。  
Markdown 仍写入 `output/{PAIR}_report.md` 作调试副本。

macOS 需 Homebrew 的 pango/gobject（本机一般已有）；脚本会设置 `DYLD_FALLBACK_LIBRARY_PATH`。若 WeasyPrint 不可用，自动回退 ReportLab。

## 相关文档

- 长期分享 / Streamlit 公开设置：`docs/sharing.md`
- **Docker 云部署（WeasyPrint / 澳洲）**：`docs/deploy-docker.md`
- 云端 redeploy 标记：`docs/CLOUD_REDEPLOY.txt`
- 部署推送：`scripts/deploy.sh`
- 临时隧道分享：`scripts/share.sh`
