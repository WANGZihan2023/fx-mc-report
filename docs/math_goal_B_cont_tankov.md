# Goal B — 跳跃过程（Cont & Tankov ↔ 本仓库）

本地参考 PDF（**勿提交 Git**；`.gitignore` 已忽略 `*.pdf`）：  
`…/书籍/Financial Modelling with Jump Processes (Peter Tankov)_9781135437947.pdf`  
（Cont & Tankov, *Financial Modelling with Jump Processes*, CRC 2004）  
下文只做章节 ↔ 代码映射与公式要点复述，**不摘录书中原文**。

## 章节对照

| Cont–Tankov 主题 | 书页（TOC） | 代码落点 |
|---|---|---|
| Ch 2.5 Poisson / 补偿泊松 `Ñ_t = N_t − λt` | ~p.58–67 | `jumps.py` 文档；补偿子开关 `jump_compensate` |
| Ch 3.2 复合泊松：`X_t = Σ_{i=1}^{N_t} Y_i` | ~p.83 | `sample_merton_jumps`；`monte_carlo._simulate_path_max_mixture` |
| Ch 4.3 Merton 跳扩散：对数跳跃 `Y_i ~ N(μ, δ²)`，强度 λ | ~p.124–125 | `jump_model="merton"`；情景 `jump_mean`/`jump_std` = μ_J / σ_J |
| Ch 6.1 复合泊松 / 跳扩散模拟（扩散与跳跃独立） | ~p.181–182 | 日步 Bernoulli 近似 + 正态跳跃尺寸；与 GBM 增量独立 |
| Ch 10.1 Merton 定价测度下的指数跳扩散形式 | ~p.327 | 本仓库为**真实世界情景混合**峰值 MC，非定价 RN；补偿子可选对齐均值 |

> PDF 可全文抽取；上表页码来自本机 TOC。若版次偏移，以章名/公式语义为准。

## 当前实现假设（Goal B）

1. **情景参数**  
   - `expected_jumps` = 窗口内期望跳跃次数 `E[N_T]`（默认 horizon ≈ `trading_days`），**不是**年化 λ。  
   - `jump_mean` / `jump_std` = 对数跳跃 `J ~ N(μ_J, σ_J²)`（Merton）。  
2. **年化一致**  
   - `λ_ann = expected_jumps · 252 / trading_days`  
   - `λ_daily = λ_ann · Δt = expected_jumps / trading_days`，`Δt = 1/252`  
3. **日步抽样**：`P(当日跳跃) = λ_daily`（Bernoulli / 稀有跳跃近似）；发生则加 `J`。  
4. **路径**：`Δ ln S = (μ − ½σ²)Δt + σ√Δt Z + J·1_{jump}`（`path_max`）。  
5. **补偿子（可选，默认关）**：`jump_compensate=True` 时日漂移再减 `λ_ann · (E[e^J]−1) · Δt`，其中 `E[e^J] = exp(μ_J + ½σ_J²)`。默认 `False` → 与 Goal B 前均值行为一致。  
6. **`jump_model`**：`merton`（默认）| `none`（关闭跳跃，忽略 `expected_jumps`）。  
7. **Brownian bridge**：仍为**纯扩散连续峰**；`expected_jumps>0` 且 `jump_model=merton` 时在 pipeline / UI 审计 / `MCResult.bb_jumps_caveat` 给出明确警告。需要跳跃加厚尾部时用 `path_max`。

## API / 校准兼容

- `run_mixture_monte_carlo(..., jump_model=..., jump_compensate=...)` 新参有默认值。  
- `ModelWeights.jump_model` / `jump_compensate`；`pack_params` / `apply_calibrated_params` 同步；旧 JSON 缺键时行为与改前一致。  
- CLI：`--jump-model merton|none`，`--jump-compensate` / `--no-jump-compensate`。  
- UI：侧栏「③ 蒙特卡洛」选择跳跃模型与补偿子；情景滑块仍编辑 E[jumps]/μ_J/σ_J。

## 未做（刻意）

- Kou 双指数跳跃、无限活动 Lévy（VG/NIG 等）  
- 精确 Poisson 到达时刻模拟（日内连续跳时刻）  
- 风险中性定价 / 测度变换（Ch 9–11）  
- 重新跑过夜校准（默认保持旧行为）
