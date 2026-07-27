# Goal D — Brownian bridge 连续峰值（Shreve II ↔ 本仓库）

本地参考 PDF（**勿提交 Git**；`.gitignore` 已忽略 `*.pdf`）：  
`…/书籍/Stochastic Calculus for Finance II.pdf`  
（Steven Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*）  
扫描件无文字层；章节页码来自本机目录 OCR（英文版书页）。下文只做章节 ↔ 代码映射与公式要点复述，**不摘录书中原文**。

## 章节对照

| Shreve II 主题 | 书页（TOC OCR） | 代码落点 |
|---|---|---|
| §3.6 First Passage Time Distribution | ~p.108 | 连续峰 / 触碰水平的背景；与障碍式生存函数同族 |
| §3.7 Reflection Principle（含 3.7.1–3.7.3：反射等式、首达时、BM 与其最大值联合分布） | ~p.111–113 | `sample_bridge_log_maxima` 所用条件最大值分布 / 逆 CDF |
| §3.4.3 Volatility of Geometric Brownian Motion | ~p.106 | 与 Goal A 一致：在 `X=log S` 上做扩散；`gbm_log_step_params` |
| §4.7 Brownian Bridge（4.7.1–4.7.5：高斯过程、标度积分、条件 BM） | ~p.172–182 | 给定日端点 `(X_t, X_{t+Δt})` 后，段内路径为布朗桥 |
| §4.7.5 Brownian Bridge as a Conditioned Brownian Motion | ~p.182 | 漂移已吸入端点 ⇒ 条件桥可用无漂移桥极值公式 |
| Ch 11 Jump Processes（对照） | 后章 | **本引擎刻意不含跳跃**；见 Goal B / `bb_jumps_caveat_message` |

> 若中译/影印版页码偏移，以章名与公式语义为准。

## 公式（工程用）

记日度对数汇率 `X = ln S`，一步方差 `v = σ_ann² Δt = σ_daily²`（Goal A / Hull 对齐）。

1. **端点**：`ΔX = (μ − ½σ_ann²)Δt + σ_daily Z`，`Z~N(0,1)`（`gbm_log_step_params`）。
2. **条件生存函数**（反射原理；`m ≥ max(a,b)`）：  
   `P(M ≥ m | X_0=a, X_Δt=b) = exp(−2(m−a)(m−b)/v)`。
3. **逆 CDF 抽样**（`U~Unif(0,1)`）：  
   `M = ½(a + b + √((a−b)² − 2 v log U))`，再与 `max(a,b)` 取大（数值保险）。
4. **多日路径峰**：对每个交易日桥段独立抽 `M_i`，路径对数峰 = `max_i M_i`（含起点），再 `exp(·)` 回现价单位。

一行版：`给定 (a,b)，M = ½(a+b+√((a−b)²−2v log U))`（反射桥最大；GBM 在 log 空间；跳跃不计）。

## 当前实现假设（Goal D）

1. **对数空间一致**：峰值在 `X=ln S` 上采样，最后 `exp`；与 Hull Goal A 日步长共用 `gbm_log_step_params`。
2. **多日合成**：非单步 naive；`trading_days` 段桥最大值取 max（已覆盖默认 66 日窗口）。
3. **数值**：`U` clip 到 `(1e-12, 1]`；`v≤0` 或退化端点时退回 `max(a,b)`；判别式 `≥0`。
4. **反变量（Goal C）**：端点 `Z` 与桥 `U` 均可 antithetic 配对。
5. **诚实边界**：连续峰只加厚**扩散**路径内极值；复合泊松跳跃仍忽略（UI/审计 caveat）。

## API / 校准兼容

- `peak_engine="brownian_bridge"` 行为加强但不改签名；默认仍为 `path_max`。  
- 校准 JSON 缺键 / `peak_engine=path_max` 时与改前一致；**未重跑过夜校准**。

## 验证

- 单元：`tests/test_brownian_bridge_max.py`（生存函数身份、逆 CDF、退化、多日 ≥ 端点 max）。  
- Smoke：`python scripts/smoke_peak_engines.py`。

## 未做（刻意）

- 跳跃桥 / 带跳连续监控  
- 亚式等其他路径泛函  
- 重新过夜校准
