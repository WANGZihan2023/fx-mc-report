# Auto-tune USD/AUD & AUD/USD (2026-07-28)

零人工标注；网格 holdout 选引擎组合后，用胜出设置重校准并 sync 到 deploy bundle。

## Grid

| Combo | peak_engine | jump_model | variance_reduction |
|---|---|---|---|
| A | path_max | merton | antithetic |
| B | path_max | none | antithetic |
| C | brownian_bridge | none | antithetic |

Backtest flags: `--max-rows 40 --n-sims 2000 --seed 42`（沿用当时 `output/calibrated_params_*.json` 情景权重）。

Selection: maximize holdout `skill_brier`; tie-break lower holdout `brier`, then higher table hit rate.

## Holdout results (selection table)

| Pair | Combo | holdout skill_brier | holdout brier | holdout logloss | table hit | table brier |
|---|---|---:|---:|---:|---:|---:|
| USD/AUD | A | 0.0059 | 0.7332 | 1.3414 | 0.275 | 0.7473 |
| USD/AUD | B | -0.1065 | 0.8161 | 1.5247 | 0.200 | 0.7641 |
| USD/AUD | C | -0.1582 | 0.8542 | 1.6183 | 0.200 | 0.7784 |
| AUD/USD | A | -0.1285 | 0.8323 | 1.6043 | 0.350 | 0.7105 |
| AUD/USD | B | -0.2537 | 0.9246 | 1.9694 | 0.375 | 0.7388 |
| AUD/USD | C | -0.2123 | 0.8941 | 1.9060 | 0.350 | 0.7199 |

## Winners

- **USD/AUD → Combo A**: `path_max` + `merton` + `antithetic` (holdout skill_brier=0.0059, brier=0.7332)
- **AUD/USD → Combo A**: `path_max` + `merton` + `antithetic` (holdout skill_brier=-0.1285, brier=0.8323)

两对胜出组合相同，共享 Combo A。

## Recalibrate

```bash
python run_cli.py calibrate --pair USD/AUD --peak-engine path_max --jump-model merton \
  --variance-reduction antithetic --n-sims 4000 --n-iters 40 --max-rows 200 --loss brier
python run_cli.py calibrate --pair AUD/USD --peak-engine path_max --jump-model merton \
  --variance-reduction antithetic --n-sims 4000 --n-iters 40 --max-rows 200 --loss brier
```

CLI 已支持 calibrate/backtest 的 `--peak-engine` / `--jump-model`（此前 calibrate 不透传 peak/jump，本次补齐）。

### Post-calib OOS

- USD/AUD: train_brier=0.7124, holdout_brier=0.7401, holdout_skill_brier=-0.0035, holdout_hit=0.300
- AUD/USD: train_brier=0.7264, holdout_brier=0.8416, holdout_skill_brier=-0.1411, holdout_hit=0.250

## How UI / runtime picks params

1. `resolve_calibrated_params_path(pair)` → `output/` 或 bundled `fx_report/data/calibrated/`.
2. `apply_calibrated_params` 写入 `ModelWeights.peak_engine` / `jump_model`（及 `recommended_variance_reduction`）。
3. Streamlit 侧栏 selectbox 默认跟 weights；方差缩减默认优先 `recommended_variance_reduction=antithetic`。
4. Pipeline `run` 同样从校准 JSON 覆盖引擎字段。

## Residual risks

- Holdout skill 仍接近 0 / 负：对气候频率基线优势有限，尤其 AUD/USD。
- 网格用重校准前权重比选引擎；重校准后 holdout 可能漂移。
- `max_rows=40` / `n_sims=2000` 为可完成小网格，非全量隔夜搜索。
- brownian_bridge 不含跳跃；与 Merton 情景不完全可比。
- 未跑 8 对 daily；未依赖新闻 API。

Log: `output/auto_tune_USDAUD_AUDUSD_20260728.log`
