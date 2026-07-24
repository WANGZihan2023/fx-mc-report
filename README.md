# 多货币对 · 最高日高蒙特卡洛情报报告

仿 Torchcast：任意 FX 货币对 → 证据强弱规则打分 → 情景加权 → 10 万次 MC → 分档报告。  
侧栏公开全部隐藏参数，以及 **strength 判定清单**。

## 头条 + 大模型精读

侧栏可选：

| 模式 | 行为 |
|------|------|
| `hybrid`（默认） | 大模型精读正文；失败则回退关键词规则 |
| `llm` | 仅大模型（失败仍回退以免报告中断） |
| `rules` | 仅关键词 |

配置 API Key（任选其一）：

1. 侧栏粘贴 **LLM API Key**（当前会话）
2. 环境变量：`OPENAI_API_KEY` 或 `LLM_API_KEY`
3. `.streamlit/secrets.toml`（见 `secrets.toml.example`）

兼容 OpenAI 接口的模型均可，例如 DeepSeek：

```bash
export LLM_API_KEY=sk-...
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
python run_cli.py --pair USD/CNH --mode llm
```

大模型会尽量抓取文章 HTML 正文摘录再判定方向/类别/强弱分项。

## 给别人看（一键）

详见 **[给别人看.md](./给别人看.md)**：

- **最省事**：部署 Streamlit Cloud，只发网址  
- **发文件夹**：对方双击 `一键打开.command`（Mac）或 `一键打开.bat`（Windows）

## 运行

```bash
cd "/Users/wangzihan/Desktop/工作_汇率/usd_aud_mc_report"
pip install -r requirements.txt
streamlit run app.py

# 命令行
python run_cli.py --pair USD/AUD --sims 100000
python run_cli.py --pair EUR/USD --sims 100000
python run_cli.py --pair USD/JPY --sims 100000
# 自定义
python run_cli.py --pair EUR/GBP --ticker EURGBP=X --sims 50000
```

## 已内置货币对

USD/AUD, AUD/USD, EUR/USD, GBP/USD, USD/JPY, USD/CNH, USD/CAD, NZD/USD, USD/CHF  
也可在侧栏「自定义」填 Yahoo ticker。

分档默认按 **现价相对百分比**（如 0%/2%/4%/6%），跨报价水平通用。

## 信息强弱怎么判断？

**不是模型“感觉”强弱**，而是清单计分（见 `strength.py`）：

```
strength = min(3, 来源分 + 意外分 + 范围分)
contrib  = direction × strength × freshness × unpriced
```

| 维度 | 含义 | 例子 |
|------|------|------|
| 来源 | 一手官方 > 交易所数据 > 一线通讯社 > 二手媒体 > 社交 | BLS CPI=1.2；博客=0.15 |
| 意外 | 相对预期/旧路径偏了多少 | 封锁/紧急加息=large~extreme |
| 范围 | 只影响本币对 vs G10 vs 系统性 | 战争封锁=systemic |
| freshness | `0.5**(年龄/半衰期)` | 地缘~5日；CPI~8日 |
| unpriced | 价格是否已反应 | 已大跳→下调，防双计 |

映射到 Torchcast 标签：≤1 SLIGHT｜≤2 MODERATE｜>2 STRONG。

侧栏每条证据可选「自动打分」或手动覆盖 strength。

## 方向约定

`+1` = 推高 **当前分析口径** 报价的路径最高值。  
例如 USD/AUD 上 +1 = 美元更强/澳元更弱；AUD/USD 上同一基本面往往是 −1。

## 模块

| 文件 | 作用 |
|------|------|
| `pairs.py` | 货币对目录与相对分档 |
| `strength.py` | 强弱判定规则 |
| `fetch_data.py` | Yahoo 抓取 + 波动率 |
| `weights.py` | 情景/证据/映射 |
| `monte_carlo.py` | 10万次混合 MC |
| `app.py` | Streamlit（侧栏权重） |
| `run_cli.py` | 命令行 |

不构成投资建议。
