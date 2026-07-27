# Goal C：峰值（path-wise maxima）估计的方差缩减（Glasserman 对应关系）

> 说明：当前代码用到的“相关方法”来自 Glasserman 的经典章节（MC 误差/估计 + 方差缩减 + 反射原理/布朗运动极值）。本仓库内不提交 PDF 原文，只做章节到实现的映射说明。

## 1) MC 误差与稳定性（Monte Carlo error）

- 章节要点（Glasserman）：估计量的方差/标准误来自抽样噪声；对同一目标函数，方差缩减会直接降低置信区间宽度。
- 代码落点：
  - `fx_report/model/monte_carlo.py`：`run_mixture_monte_carlo(..., variance_reduction=...)` 控制不同抽样方案（默认保持现有行为）。
  - 新增字段：`MCResult.variance_reduction` 用于 UI/日志审计。

## 2) 方差缩减：反变量（Antithetic variates）

- 章节要点（Glasserman）：若目标函数对某些潜变量单调/近似凸，则使用反变量配对能降低方差（即对随机输入做成对负相关）。
- 代码落点：
  - `variance_reduction="antithetic"`：
    - `path_max` 引擎：对扩散部分的正态增量 `z` 做反变量配对；并复用同一对的跳跃抽样（使配对仍“在路径增量层面”相关）。
    - `brownian_bridge` 引擎：对端点对应的正态增量 `z` 做反变量配对；并对布朗桥“反射原理”用到的 `U~Unif(0,1)` 做 `U` 与 `1-U` 配对，从而让桥段极值抽样也保持配对一致性。
  - 入口参数：
    - `run_mixture_monte_carlo(... variance_reduction=...)`
    - CLI：`--variance-reduction none|antithetic`
    - UI：侧栏 `③ 蒙特卡洛 → 方差缩减 variance_reduction`

## 3) 峰值/极值：Brownian-bridge 与反射原理（Path-dependent maxima）

- 章节要点（Glasserman）：布朗桥的连续监控最大值可用反射原理转为一个“可逆的极值分布采样”（涉及对数极值与 `log U` 的逆变换）。
- 代码落点：
  - `fx_report/model/brownian_bridge_max.py`：
    - `sample_bridge_log_maxima(...)`：使用反射原理/条件极值的逆-CDF 采样桥段连续最大值。
    - `simulate_bb_path_maxima(...)`：把逐日桥段最大值拼成整段 path 最大值。

## 4) 当前实现的验证方式（可编码、可测试）

- 单元测试：
  - `tests/test_peak_variance_reduction.py`
    - `test_antithetic_is_deterministic_with_seed`：同种子下 antithetic 输出可复现。
    - `test_antithetic_reduces_empirical_variance_on_diffusion_toy`：在“跳跃被关掉的扩散 toy 场景”上，对比 bucket 概率估计的经验方差。

## 5) 运行/回测如何开关

- CLI（示例）：
  - `python run_cli.py run --variance-reduction antithetic ...`
  - `python run_cli.py calibrate --variance-reduction antithetic ...`
  - `python run_cli.py backtest --variance-reduction antithetic ...`

