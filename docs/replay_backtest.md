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
- `python run_cli.py replay-summary --out output`

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
- 仅使用（按尝试顺序）：
  1. 本地 `inbox` 中 `published/mtime <= as_of` 的文件
  2. `NewsAPI everything` 的 `from/to` 日期过滤（需 KEY；开发者档约近 29 天）
  3. **GDELT DOC 2.0 ArtList** 的 `startdatetime/enddatetime`（**免费无 KEY**；约近 90 天）

质量字段：

- `historical_news_quality=date_filtered`
  - 至少有一处日期约束命中：NewsAPI **或** GDELT **或** 带日期的 inbox
- `historical_news_quality=limited`
  - 上述来源均无命中（或窗口外 / 限流失败且无 inbox）
  - 这代表证据链**不是完整历史新闻快照**

`limitation` / meta 会写明各源命中数与错误（含 NewsAPI/GDELT 429），不会静默吞掉失败。

更严格的验收口径：

- **历史新闻真正工作**
  - 至少一个回放时点同时满足：
    - `historical_news_quality=date_filtered`
    - `evidence_n > 0`
- **历史新闻仍未真正工作 / 仍受限**
  - 全部样本都还是 `limited`
  - 或虽然出现 `date_filtered`，但 `evidence_n = 0`
  - 这说明“日期过滤接口可用”与“真的产出可用历史证据”仍有差距

## 仍然是近似的地方

即使 `historical_news_quality=date_filtered`，也仍不是“完美当日信息集”，原因包括：

- NewsAPI / GDELT 可检索范围、收录源和排序并不等于真实市场终端
- GDELT `seendate` 是监测见到时间，不一定等于原文精确发布时间
- 某些历史文章正文可能现在不可访问
- LLM 证据判定仍是今天的模型在读历史文本
- 模板证据 / prior 策略若启用，仍会按当前配置参与

因此建议把 replay backtest 用于：

- 验证流程是否具备“历史冻结”能力
- 粗看 bucket 概率是否有方向性 / 校准价值
- 比较 price-only、limited-news、date-filtered-news 三种路径

而不是把它当成完美的机构级新闻回放数据库。

## UI

Streamlit 中新增了 `历史时点回放` 卡片：

- 只建议跑 2-5 个日期的小样本
- 若新闻保真度有限，会弹出警告
- 结果表会展示 `historical_news_quality`

另外新增 `历史冻结回放总览`：

- 自动汇总 `output/` 下已有 replay JSON
- 展示 pair / n / hit_rate / mean_brier / mean_skill_brier
- 同时展示 `evidence_mean`、`evidence_max`、`date_filtered` / `limited` 次数
- `历史新闻是否工作=yes` 的判定标准：
  - 至少一个时点 `historical_news_quality=date_filtered`
  - 且同一时点 `evidence_n > 0`

## 建议用法

1. 先用小样本（2-3 个日期）做 smoke
2. 无 NewsAPI KEY 时可先跑：`python scripts/smoke_gdelt_historical.py`
3. 先看 `历史冻结回放总览` 或 `python run_cli.py replay-summary --out output`
4. 如果结果仍是 `历史新闻是否工作=no`，不要把“回放能跑通”误当成“历史新闻已接通”
5. 优先关闭大规模 `ai_research` 期待，因为历史模式里它会被诚实禁用
6. 可选：每日把 RSS 归档进 inbox，减轻对在线源的依赖：
   `python scripts/daily_inbox_snapshot.py --dry-run`
7. 如果需要更高新闻保真度，后续可再接入更长归档供应商

## 引擎 A vs C 自动对比

先 `source railway-variables.env`（或保证进程里已有 `NEWSAPI_KEY` 等），再跑：

```bash
set -a && source railway-variables.env && set +a
python run_cli.py replay-engine-compare --pair USD/AUD
# 等价：python scripts/auto_replay_engine_compare.py --pair USD/AUD
```

默认行为：

- 扫描近窗（约 `today-25` → `today-(days+1)`），步长 `--step 3`
- 候选条件：`evidence_n>0` **或** `historical_news_quality=date_filtered`
- 最多 `--max-dates 3`（控 NewsAPI 配额；命中 429 提前停；GDELT 可作无 KEY 补充）
- 每个候选各跑一次引擎 **A**（`path_max+merton+antithetic`）与 **C**（`brownian_bridge+none+antithetic`）
- 小样本默认：`--sims 800 --days 20 --lookback 14 --mode rules`
- 输出：`output/engine_compare/A_{asof}.json`、`C_{asof}.json`、`summary.json`，并打印中文汇总表

跳过扫描、指定日期：

```bash
python run_cli.py replay-engine-compare --dates 2026-07-10,2026-07-15 --max-dates 2
```
