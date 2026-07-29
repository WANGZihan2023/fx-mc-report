# 算法自动推荐（简洁模式）

默认路径只需 **货币对 + 看涨货币**（可选上传单子 PDF/图片）；系统自动挑选
`peak_engine` / `jump_model` / `variance_reduction` / `cluster_method` /
`use_calibrated` / `human_review`，并在审计面板写明中文理由。

## 推荐优先级

1. **校准 JSON**（`output/calibrated_params_{PAIR}.json` 或内置
   `fx_report/data/calibrated/`）若含 `peak_engine` / `jump_model` /
   `recommended_variance_reduction`（或 `calibration.recommended_variance_reduction`）
   → 采用这些字段；缺省字段用产品默认补齐。
2. 否则若 `output/engine_compare/summary.json` 的 `pair` 匹配，且能解析出胜者
   （顶栏 `overall_winner` / `winner`，或行内 `winner` 多数票）→ 映射
   `ENGINE_COMBOS`（A=`path_max+merton+antithetic`，C=`brownian_bridge+none+antithetic`）。
3. 否则 **产品默认**：`path_max` + `merton` + `antithetic`，聚类 `jaccard`，
   有校准文件则 `use_calibrated=true`，`human_review=true`。

实现：`fx_report/model/algo_recommend.py` → `recommend_algorithms(pair)`。

## UI

| 模式 | 开始设置必选 | 算法项 |
|---|---|---|
| **简洁（推荐）** | 货币对、看涨（运行时还须分档方式，除非 PDF 已填） | 隐藏；确认时写入 `start_cfg` + `algo_recommend` |
| **专家** | 另加峰值引擎 / 是否校准 / 人工确认 | 手选；侧栏可继续选 jump / VR |

侧栏①在简洁模式下显示「改用专家设置」一键解锁。

结果页「本次分析审计」在简洁模式展示：

> **本次算法由系统推荐** + 来源 + 字段 + 理由列表

## 测试

```bash
pytest tests/test_algo_recommend.py -q
```
