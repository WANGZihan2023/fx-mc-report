# Goal E / J — 恰当评分与可审计对错（Gneiting ↔ 本仓库）

本地参考 PDF（**勿提交 Git**）：`/Users/wangzihan/Desktop/工作_汇率/书籍/Gneiting-StrictlyProperScoring-2007.pdf`  
（Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, 2007。下文只做概念 ↔ 代码映射，**不摘录论文原文**。）

## 概念对照

| Gneiting 概念 | 含义（复述） | 代码落点 |
|---|---|---|
| Strictly proper scoring rule | 只有报出真实条件分布时期望分最优；激励诚实概率 | `fx_report/model/scoring.py` |
| Brier score（多类） | \(\sum_k (p_k - y_k)^2\)，越低越好 | `brier_multiclass`；校准 `sample_loss(..., loss="brier")`；回测 `eval_split_metrics` |
| Logarithmic score / log-loss | \(-\sum_k y_k \log p_k\)，严格恰当 | `log_loss_multiclass`；`loss="logloss"` |
| Skill score | \(1 - S_{\mathrm{model}} / S_{\mathrm{baseline}}\)，相对朴素基准的改进（越高越好） | `skill_score`；摘要字段 `skill_brier` / `skill_logloss` |
| Climatology / frequency baseline | 用历史分档频率（或均匀）作基准预报 | `frequency_baseline` / `uniform_probs`；holdout 用 **train** 气候频率 |
| Reliability / calibration | 预测概率与经验命中是否对齐 | `reliability_by_prob_bins`、`reliability_per_bucket`、`reliability_ece` |
| CRPS（连续） | 连续分布 vs 实现值的恰当分 | **未实现**（需 path-max 样本分布）；见文末 |

## 当前实现假设

1. **预报对象**：季度路径峰值所落分档（多类 one-hot），不是点估计。
2. **Brier / log-loss**：与校准循环、回测 CSV 行内分一致；OOS 摘要在 `calib_oos_summary_*.json` 的 `train` / `holdout` 下写入（旧键保留，新增 skill / reliability）。
3. **Skill 基准**：默认 `frequency`（样本内气候频率）；holdout 的基准向量取自 **train** 频率，避免用 holdout 自身泄漏。
4. **可靠性**：分档级（mean \(p\) vs emp rate）+ argmax 概率分箱 + ECE；UI「跨对质量」/ 回测区展示。
5. **可审计对错（Goal J）**：回测表 `hit` / `pred_bucket` / `true_bucket` + 上述评分，便于核对「对在哪、错在哪、概率是否校准」。

## API / 摘要兼容

- 旧 `calib_oos_summary_*.json` 无 `skill_brier` 时 UI 显示「—」，不报错。
- 重新跑 `calibrate` / `backtest` 会写入完整字段。
- 主入口：`summarize_forecast_scores`；回测/校准经 `eval_split_metrics` 调用。

## UI

- 「跨对质量」：Holdout Brier、**Skill（Brier）**、可靠性 ECE；当前对可展开分档可靠性表。
- 信任条 / 回测结果：同步展示 Skill；回测后可看可靠性表。

## 未做（刻意）

- **CRPS** 连续峰值：需要每次预报保留 max 样本（或 CDF），再与 `realized_max` 比较；接口清晰后再加。
- 重写校准优化器本身（仍用 Brier/log-loss 搜索）；本目标是 **评估与审计面**。
- 把版权 PDF 拷进仓库（`.gitignore` 已忽略 `docs/refs/*.pdf`）。
