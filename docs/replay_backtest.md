# 历史时点回放 backtest

## 目标

给定历史时点 `T`，用**当时可见的信息集**重跑完整流水线：

1. 历史价格截面
2. 历史新闻 / 证据
3. 情景映射
4. 蒙特卡洛分桶

然后把预测分桶概率与 `T` 之后 `horizon` 窗口内的**实际最高价**对比。

## 当前实现

命令行：

```bash
python run_cli.py replay-backtest \
  --pair USD/AUD \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --step 7 \
  --sims 800 \
  --days 66
```

输出：

- `output/replay_backtest_<PAIR>_<SPAN>.csv`
- `output/replay_backtest_<PAIR>_<SPAN>.json`

核心字段：

- `as_of`
- `pred_bucket`
- `true_bucket`
- `brier`
- `skill_brier`
- `evidence_n`
- `historical_news_quality`

## 什么是“真冻结”

### 价格端：较精确

价格历史会按 `as_of` 截断：

- `fetch_market(..., as_of_date=T)` 只使用 `T` 及以前的价格
- 波动率窗口也只来自 `T` 及以前
- 实现值 `realized_max` 来自 `T` 之后 `horizon` 窗口的实际最高价

因此价格 / 波动 / 分桶边界这一段，属于**真实历史冻结**。

### 新闻端：best-effort，分质量标记

历史新闻是最难完美复现的部分。当前实现遵循“宁可保守，也不假装完整”：

- 历史回放**不使用**当前 RSS 流、Google News RSS、Finnhub category feed
- 历史回放**不使用** AI researcher 的实时搜索 / 白名单网页
- 仅使用：
  - `NewsAPI everything` 的 `from/to` 日期过滤
  - 本地 `inbox` 中 `mtime <= as_of` 的文件

质量字段：

- `historical_news_quality=date_filtered`
  - 至少使用了真实日期过滤的 NewsAPI 历史检索
- `historical_news_quality=limited`
  - 没有可用的日期过滤新闻源，或只有本地 inbox
  - 这代表证据链**不是完整历史新闻快照**

## 仍然是近似的地方

即使 `historical_news_quality=date_filtered`，也仍不是“完美当日信息集”，原因包括：

- NewsAPI 可检索范围、收录源和排序并不等于真实市场终端
- 某些历史文章正文可能现在不可访问
- LLM 证据判定仍是今天的模型在读历史文本
- 模板证据 / prior 策略若启用，仍会按当前配置参与

因此建议把 replay backtest 用于：

- 验证流程是否具备“历史冻结”能力
- 粗看 bucket 概率是否有方向性 / 校准价值
- 比较 price-only、limited-news、date-filtered-news 三种路径

而不是把它当成完美的机构级新闻回放数据库。

## UI

Streamlit 中新增了 `历史时点回放` 小节：

- 只建议跑 2-5 个日期的小样本
- 若新闻保真度有限，会弹出警告
- 结果表会展示 `historical_news_quality`

## 建议用法

1. 先用小样本（2-3 个日期）做 smoke
2. 优先关闭大规模 `ai_research` 期待，因为历史模式里它会被诚实禁用
3. 如果需要更高新闻保真度，后续可接入真正支持历史归档检索的供应商
