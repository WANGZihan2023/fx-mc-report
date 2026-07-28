# 不确定证据 · 人机协同（HITL）+ Active Learning

赋权与蒙特卡洛**之前**，对置信度不足的证据暂停，请人选择方向后再继续。不发明证据。

选点策略按 **Active Learning** 排序：不确定度打分 + **簇多样性**（避免同主题重复询问），详见 `docs/ecda_active_learning.md`。

## 何时暂停？

最多 **3–5** 条（按 AL 优先序），触发条件可叠加：

| 原因码 | 中文 | 含义 |
|---|---|---|
| `low_confidence` | 置信度偏低 | 强度偏弱（SLIGHT / strength≤1） |
| `rules_llm_conflict` | 规则与 LLM 冲突 | LLM 结论与关键词规则方向不一致 |
| `cluster_direction_conflict` | 同簇方向冲突 | 事件簇内已有 ± 对立 |
| `unclear_category` | 类别不清 | `unclassified` / `other` |
| `near_neutral_margin` | 边界中性 | 方向=0 但强度≥1.2 |

打分权重（高→低）：规则/LLM 冲突 > 簇冲突 > 类别不清 > 边界中性 > 低置信度；多原因叠加有 bonus；同簇 `rep` 略优先于 `dup`。

选点后对同簇剩余候选乘以 `cluster_diversity=0.35`，使 HITL 队列覆盖更多主题。

## 流水线两阶段

1. **Phase A** `run_pipeline_phase_a`：抓取 → 摘要 → 分类 → 聚类 → 漂移检查 → `pending_reviews`
2. **Phase B** `run_pipeline_phase_b(checkpoint, review_overrides=…)`：应用人工选择 → S → MC → 报告

CLI 默认 `human_review_mode=auto_skip`：只记日志，不弹窗，保留模型方向继续。  
可用 `--auto-skip-uncertain` / `--max-uncertain N` 显式声明。

## Streamlit 怎么用

1. 侧栏 **② 抓取与判定** 勾选 **「不确定证据先人工确认」**（默认开）
2. 点主区 **运行分析**
3. 若有待确认项：页面出现表单（标题/摘要/模型猜测/原因/`AL优先#`）
4. 每条选 **利多 / 利空 / 中性 / 跳过** → **确认并继续生成报告**
5. 选择会写入 `output/label_audit_*.csv`（跳过不写人工方向）
6. 刷新页面时，未提交的检查点尽量从 `session_state["hitl_checkpoint"]` 恢复

关闭勾选则等同 CLI：自动跳过不确定项。

## 代码入口

- 检测 / AL 选点 / 覆盖：`fx_report/model/human_review.py`
  - `detect_uncertain_evidence` / `active_learning_prioritize`
  - `prioritize_for_hitl`
- 检查点：`fx_report/pipeline.PipelineCheckpoint`
- UI：`fx_report/ui/streamlit_app.py` → `render_hitl_uncertain_form`
