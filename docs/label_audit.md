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
2. 滚到 **「证据人工标注」**  
3. 展开 **「怎么填？」** 看例子  
4. （可选）点 **「一键按模型预填再改」**，只改不同意的几条  
5. 每条选 `human_direction` / `human_category`；`agree` 一般会自动填  
6. 点 **「保存标注到 output/」** → 文件  
   `output/label_audit_{PAIR}_{YYYY-MM-DD}.csv`  
7. 或点 **下载当前标注 CSV**

### 没有证据时

页面会说明原因（未配新闻 Key / 抓取为空等），并提供：

- **加载练习样例** — 用演示语句练手  
- 或去侧栏 **API 配置** 填写 `NEWSAPI_KEY` / `FINNHUB_API_KEY` 后重跑

## 代码位置

- 词表与存盘：`fx_report/model/label_audit.py`
- CSV 导出：`fx_report/model/backtest.py` → `evidence_to_label_audit`
- UI：`fx_report/ui/streamlit_app.py` → `render_label_audit_section`
