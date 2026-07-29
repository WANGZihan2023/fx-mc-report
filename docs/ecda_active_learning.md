# ECDA 扩展：Active Learning · 漂移监控 · 证据摘要

在事件聚类（`docs/event_clustering.md`）之外的三项诚实性升级。

## 1. Active Learning 优先排序（HITL）

赋权前对不确定证据打分，并按**信息增益 + 簇多样性**选 top-N 给人看。

| 组件 | 路径 |
|---|---|
| 检测 / 打分 / 多样性选点 | `fx_report/model/human_review.py` |
| 产品说明 | `docs/human_review.md` |

新增原因码：`near_neutral_margin`（方向中性但强度不弱）。  
选点：`prioritize_for_hitl` — 同簇后续候选 ×0.35，避免队列全挤在同一 `EVT-*`。  
UI 展示 `AL优先#k` 与不确定度分数。

## 2. 轻量漂移监控

| 组件 | 路径 |
|---|---|
| TV 距离 + 基线 EMA | `fx_report/news/drift.py` |
| 基线文件 | `output/evidence_drift_baseline_{PAIR}.json` |

每次 `step4` 后比较类别/方向分布 vs 滚动基线；TV≥0.40 → 写入 `cluster_warnings` 风格中文告警（审计面板黄色框）。首跑只播种基线，不告警。

**默认：仅告警，不改 S**（`soft_adapt=False` / `step4(..., soft_adapt_drift=False)`）。

### 可选 soft adaptation（默认关）

开启后（`soft_adapt=True`）：若 TV 告警，对**相对基线过表示**的类别/方向条目做 strength soft-shrink（向中性拉低 |contrib|），并设 floor（默认不低于原 strength 的 55%）。

- **不发明证据、不改方向、不加假标题**
- 审计字段：`drift_adapted=true|false`、`adapt_changes[]`（id / before / after / shrink）、`adapt_note`
- Streamlit「本次分析审计」会显示 `drift_adapted` 与适应说明

## 3. 证据摘要层

| 组件 | 路径 |
|---|---|
| 抽取式压缩（离线） | `fx_report/news/summarize.py` |
| 字段 | `EvidenceItem.summary` |

在聚类 / HITL / 赋权前，把标题+headline.summary+长 note 压成 ≤220 字可审计 blurb。默认 **extractive**；有 LLM key 时可 `prefer_llm=True`，失败自动回退。HITL `snippet` 优先用 `summary`。
