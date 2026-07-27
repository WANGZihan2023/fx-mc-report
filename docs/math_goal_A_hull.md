# Goal A — 日度路径与波动（Hull ↔ 本仓库）

本地参考 PDF（**勿提交 Git**）：`docs/refs/Hull_Options_Futures_Other_Derivatives.pdf`  
版本：清华社中译 **第 8 版**（扫描件，无文字层；章节对照来自封面/目录 OCR + 英文页 OCR）。  
下文只做章节 ↔ 代码映射与公式要点复述，**不摘录书中原文**。

## 章节对照

| Hull（8e）主题 | OCR 核对 | 代码落点 |
|---|---|---|
| Ch13 Wiener / Itô / GBM：`dS = μ S dt + σ S dz`；离散 `ΔS/S = μ Δt + σ √Δt Z`；风险中性下 μ→r | 英文页约 PDF 321–323（书页 ~286–288） | `fx_report/model/gbm_vol.py` → `gbm_log_step_params`；`monte_carlo.py` / `brownian_bridge_max.py` 日步长 |
| 对数正态精确离散：`Δ ln S = (μ − ½σ²)Δt + σ√Δt Z` | 同上 + 工程标准 GBM Euler–Maruyama on log | 同上（`drift + diffusion·Z`） |
| Ch14 历史波动：`u_i = ln(S_i/S_{i−1})`；年化用交易日 √N，N≈252 | PDF ~341（Table 14.1 / Trading Days vs Calendar Days） | `estimate_vol(..., estimator="window")`；`fetch_data._snapshot_from_series`；`history_peaks._sigma_daily` |
| Ch5 / Ch15 外汇远期 / 风险中性漂移 ≈ 利率差（carry） | Ch5 OCR ~PDF 149–151；定价测度与真实世界 μ 分开 | `drift_mode="carry"` + `carry_mu_annual` 占位（不拉实时利率） |
| Ch22 EWMA：`σ_n² = λ σ_{n−1}² + (1−λ) u_{n−1}²`，λ≈0.94 | 目录 OCR 见 Ch22 EWMA 条目 | `estimate_vol(..., estimator="ewma")` |

> PDF 为整书扫描图，全文检索不可用；上表页码以本机 OCR 抽样为准。若页码偏移，以章名/公式语义为准。

## 当前实现假设（Goal A 之后）

1. **时间单位**：每个交易日 `Δt = 1/252` 年；路径长度 `trading_days`（默认 66 ≈ 一季交易日）。
2. **波动输入**：行情与峰值样本给出 **日波动** `σ_daily`；MC 内部用 `σ_ann = σ_daily · √252`，扩散项用 `σ_daily · Z`（等价于 `σ_ann · √Δt · Z`）。
3. **漂移 μ（年化）**  
   - 默认 `drift_mode="scenario"`：情景 `mu_annual` + 证据 `mu_annual_shift`（真实世界情景混合，非定价测度）。  
   - `zero`：μ=0 + shift（仍保留 Itô −½σ²）。  
   - `carry`：用 `carry_mu_annual` 占位利率差 + shift（Hull FX RN 漂移形状；需外部填入，默认 0）。
4. **跳跃**：`path_max` 引擎在 GBM 上叠加复合泊松跳跃；`brownian_bridge` **不含跳跃**（连续峰近似）。
5. **波动估计器**：默认窗口样本标准差（Hull Ch14）；可选 EWMA（Ch22）。校准 JSON 可带 `vol_estimator` / `ewma_lambda` / `drift_mode` / `carry_mu_annual`，缺省向后兼容。

## API / 校准兼容

- `run_mixture_monte_carlo(..., drift_mode=..., carry_mu_annual=...)` 新参有默认值。  
- `ModelWeights` / `pack_params` / `apply_calibrated_params` 同步新键；旧 JSON 无这些键时行为与改前一致。  
- UI 未暴露新控件时仍走 dataclass 默认（`window` + `scenario`）。

## 未做（刻意）

- Goal C 以外的更多 VR（控制变量等）  
- 实时利率曲线拉 carry  
- GARCH(1,1) 全估计（仅 EWMA 轻量选项）

> Goal B（Cont–Tankov / Merton 跳跃）见 `docs/math_goal_B_cont_tankov.md`。  
> Goal D（Shreve II 布朗桥连续峰）见 `docs/math_goal_D_shreve.md`。
