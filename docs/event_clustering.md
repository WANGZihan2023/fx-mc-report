# 事件聚类 / 同主题去重（ECDA-style）

对应 ECDA 2014 论文集精神：**同主题新闻不应线性叠加入证据分 S**，否则情景权重会被同事件重复报道抬高。

## 落点

| 步骤 | 代码 |
|---|---|
| 标题 token Jaccard 单链接聚类 | `fx_report/news/cluster.py` |
| `cluster_id` / `cluster_role` 挂在证据上 | `EvidenceItem`；语句侧 `StoredStatement.cluster_id` |
| 赋权前默认开启 | `pipeline.step4_evaluate_impact(..., cluster_events=True)` |
| 计入 S 时簇内去重 | `evidence_score` → `keep_strongest`（dup 贡献=0） |

## 行为（生产默认）

1. 对新闻证据标题做轻量 tokenization（无 sklearn）。
2. Jaccard ≥ 0.45 且类别兼容 → 同一 `EVT-xx` 簇。
3. 簇内 **代表（rep）** = `|direction×strength×freshness×unpriced|` 最大者；其余标 `dup`。
4. 求和 S 时 **仅 rep / solo 计入**；UI/审计仍展示全部证据条。
5. `news_meta` / 审计面板：`cluster_n` vs `evidence_raw_n`，`cluster_dedup_applied`。

可选模式（测试/调参）：`soft_avg`、`soft_sqrt`、`off`（见 `cluster_score_mult`）。

## 诚实性

- 不发明证据；只对已有标题聚类。
- 去重只影响权重聚合，不删除证据卡片。
- 不同类别（如 geopolitics vs china_iron）默认不合并，降低误伤。

## 异常告警（`cluster_warnings`）

聚类后自动检测并写入 `news_meta` / `diag["cluster_warnings"]`（中文），**不编造证据**：

| 现象 | 触发条件（摘要） |
|---|---|
| 可能过度合并 | 同簇内方向冲突（±），或紧类别不一致 |
| 可能漏合 | 标题 Jaccard 贴近阈值但未合簇；或 Jaccard≥阈值但被类别阻隔 |
| 去重过重 | `dup/raw ≥ 0.5`（且 raw≥3） |
| 去重后无有效证据 | raw>0 但 keep_strongest 下无计入 S 的条 |
| 新闻为空 / 429 | 与 fetch `limitation` 同风格的附带说明 |

展示：`pipeline` 审计字段 → Streamlit「本次分析审计」+ `st.warning` → 报告 meta 一行 `⚠N告警`。

## UI

结果页 **「本次分析审计」**：`evidence_n` / `cluster_n` / `raw` / `dedup=True|False`；若发生去重会有中文说明；有 `cluster_warnings` 时额外黄色告警框。
