# 校准日常闭环（overnight → sync → Railway）

本机跑完 Stage 0–1 校准后，用同步脚本把 JSON 打进仓库，再 push 让 Railway 重建镜像。

## 标准流程

```bash
cd /Users/wangzihan/Desktop/工作_汇率/usd_aud_mc_report

# 1) 过夜 / 加强校准（任选其一）
./scripts/overnight_calibrate.sh          # 单对或脚本内默认
# 或
./scripts/overnight_calibrate_full.sh     # 多对加强

# 2) 把 output/ 校准产物拷进部署目录（已跟踪在 git）
./scripts/sync_calibrated_to_deploy.sh

# 3) 提交并推送 → Railway 自动 redeploy
git add fx_report/data/calibrated/
git commit -m "Refresh bundled calibrated params"
git push origin main
```

## 云端如何确认

打开公网 URL → 选货币对 → 页顶「已加载校准参数」+ Holdout；侧栏勾选「使用校准参数」；跑一次后「本次分析审计」参数来源应为校准 JSON 文件名。

详见 `docs/deploy-docker.md`「校准参数如何进云镜像」。

## 可选：本机 cron 提醒（不强制）

不想每天手敲时，可加一条 cron（仅示例；路径按本机改）：

```cron
# 每周日 02:00 跑多对校准并同步（需本机醒着且有网络）
0 2 * * 0 cd /Users/wangzihan/Desktop/工作_汇率/usd_aud_mc_report && ./scripts/overnight_calibrate_full.sh && ./scripts/sync_calibrated_to_deploy.sh >> /tmp/fx_calib_cron.log 2>&1
```

推送仍建议人工 `git commit && git push`（避免 cron 自动改 git）。也可只跑校准，第二天上班再 sync + push。

## 相关文件

| 文件 | 作用 |
|------|------|
| `scripts/overnight_calibrate.sh` | 过夜校准入口 |
| `scripts/overnight_calibrate_full.sh` | 多对加强 |
| `scripts/sync_calibrated_to_deploy.sh` | `output/` → `fx_report/data/calibrated/` |
| `fx_report/data/calibrated/` | 打进 Docker / Railway 的 JSON |
| `docs/deploy-docker.md` | 部署与 `APP_PASSWORD` |
