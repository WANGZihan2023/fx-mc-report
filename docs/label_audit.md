# 证据人工标注（label_audit）说明

面向非机器学习用户：在网页里给每条新闻证据打方向/类别标签，用于核对模型是否判对。

详细字段与例子也写在 UI 折叠区「怎么填？」。

## 你在标什么？

模型跑完后会用到若干条**证据语句**（标题/摘要）。你对每一条判断：

> 这条新闻对**分析报价**意味着什么？  
> （看涨货币走强 = 分析报价升高 = 方向 `up`）

## 列说明

| 列 | 谁填 | 含义 |
|---|---|---|
| `statement_id` | 系统 | 语句编号 |
| `title` | 系统 | 标题/摘要 |
| `url` | 系统 | 原文链接（如有） |
| `model_direction` | 模型（只读） | `up` / `down` / `neutral` |
| `model_category` | 模型（只读） | 驱动类别，如 `rba`、`fed` |
| `human_direction` | **你** | 你的方向判断 |
| `human_category` | **你** | 你选的类别（与模型同一词表） |
| `agree` | **你**（可自动） | 是否与模型方向一致 |

## 允许值

### `human_direction` / `model_direction`

| 值 | 含义 |
|---|---|
| `up` | 推高分析报价路径峰值（看涨货币走强） |
| `down` | 压制峰值（看涨货币走弱） |
| `neutral` | 中性 |
| `unclear` | 无法判断（**仅人工**；模型一般不会给） |

与代码里 `EvidenceItem.direction` 对应：`+1→up`，`-1→down`，`0→neutral`。

### `human_category` / `model_category`

与 `VALID_CATEGORY` / `CATEGORY_RULES` 一致，例如：

`geopolitics` · `oil` · `cpi` · `fed` · `ecb` · `boe` · `boj` · `rba` · `rbnz` · `boc` · `snb` · `pboc` · `china_growth` · `china_iron` · `yields` · `growth` · `positioning` · `other` · `unclassified` · `dairy`

### `agree`

| 值 | 含义 |
|---|---|
| `yes` | 人工方向 = 模型方向 |
| `no` | 不一致 |
| `unsure` | 不确定（含人工选了 `unclear`） |

两边方向填好后，UI 会自动给出 `agree`，你仍可手动改。

## 中文例子（看涨=AUD，报价=AUD/USD）

1. 「RBA 偏鹰」→ 澳元走强 → **`up`**，类别 `rba`
2. 「美联储偏鹰」→ 美元走强 → **`down`**，类别 `fed`
3. 「铁矿石大涨」→ 利多澳元 → **`up`**，类别 `china_iron`
4. 「仅假期休市、无政策信息」→ **`neutral`** 或 **`unclear`**

若分析口径是 **USD/AUD**（看涨仍为 AUD），同一条「RBA 偏鹰」应为 **`down`**（报价升高 = 美元相对澳元更强，与 AUD 走强相反）。

## 怎么用（网页）

1. `streamlit run app.py`，选货币对与看涨货币，运行分析  
2. 在 **「本次分析审计」正下方** 找到 **「证据人工标注」**（完整报告之上；也可开侧栏 **⑨ 证据人工标注**）  
3. 展开 **「怎么填？」** 看例子  
4. （可选）点 **「一键按模型预填再改」**，只改不同意的几条  
5. 每条选 `human_direction` / `human_category`；`agree` 一般会自动填  
6. 有标注后页面会显示 **抽检准确率**（= 同意率 yes / yes+no）；练习样例会标明非正式  
7. （可选）勾选 **「用人工标注重算权重」** → **「应用人工标注重算」**：用人工方向覆盖模型方向，重算 S 与情景权重（并尽量重跑 MC）  
8. 点 **「保存标注到 output/」** → 文件  
   `output/label_audit_{PAIR}_{YYYY-MM-DD}.csv`  
   同时写入 `output/label_spotcheck_{PAIR}_{日期}.json`（抽检准确率）  
9. 或点 **下载当前标注 CSV**

### 没有证据时

标注区**仍会显示**（不藏在 PDF 后面），并说明原因（未配新闻 Key / 抓取为空等），并提供：

- **怎么填？** — 字段说明与例子（默认展开）  
- **加载练习样例** — 用演示语句练手（主按钮）  
- **Railway / 环境变量检查清单** — `NEWSAPI_KEY` / `FINNHUB_API_KEY` / LLM / `APP_PASSWORD` 等  
- 无 Key 时流水线仍会试 **央行 RSS + Google News 公开 RSS**；相关度不够则 evidence_n=0（诚实空证据）

## Stage 3 · 标签学习强度（脚手架）

当 `output/label_audit_*.csv` 合计有 **≥ 20** 条带 `human_direction`（up/down/neutral）的标注时：

1. 系统拟合 **类别 → 方向先验** 与 **强度倍率**（`fx_report.model.label_learn`）  
2. 侧栏 **② 抓取与判定** 勾选 **「使用标签学习到的强度」** 后，下次运行会把倍率应用到证据 `strength`  
3. 若不足 20 条：UI / 审计框显示 **「标注不足，需至少 20 条」**——代码路径已就绪，等你标够即可  

拟合结果可落盘：`output/label_learned_params.json`。

这不是端到端深度学习，只是可开关的最小闭环，方便清单「可学习证据」结构上完成。

## 代码位置

- 词表与存盘 / 同意率 / 抽检 / 标注重算：`fx_report.model.label_audit`
- Stage 3 强度学习：`fx_report.model.label_learn`
- CSV 导出：`fx_report.model.backtest` → `evidence_to_label_audit`
- UI：`fx_report.ui.streamlit_app` → `render_label_audit_section`
- 跨对 OOS 看板：同 UI → `render_cross_pair_quality_board`
- 校准日常闭环：`docs/calibration_loop.md`
