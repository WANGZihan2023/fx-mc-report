# 校准日常闭环（overnight / daily → sync → Railway）

本机跑完 Stage 0–1 校准后，用同步脚本把 JSON 打进仓库，再 push 让 Railway 重建镜像。

## 标准流程（手动）

```bash
cd /Users/wangzihan/Desktop/工作_汇率/usd_aud_mc_report

# 1) 过夜 / 加强校准（任选其一）
./scripts/overnight_calibrate.sh          # 默认 USD/AUD + AUD/USD
# 或
./scripts/overnight_calibrate_full.sh     # 多对加强（更重）

# 2) 把 output/ 校准产物拷进部署目录（已跟踪在 git）
./scripts/sync_calibrated_to_deploy.sh

# 3) 提交并推送 → Railway 自动 redeploy
git add fx_report/data/calibrated/
git commit -m "Refresh bundled calibrated params"
git push origin main
```

## 每日自动校准（本机 launchd）— 已安装

本机用 **LaunchAgent** 每天 **03:00（Asia/Shanghai）** 跑中等强度校准，成功后 sync，并尝试 `git commit` + `push`（消息 `chore: daily calibrated params`），便于 Railway 拉取。

| 项 | 值 |
|----|-----|
| 计划 | `~/Library/LaunchAgents/com.fxreport.daily-calibrate.plist` |
| 时刻 | 每天 03:00，`TZ=Asia/Shanghai`（与系统 CST 一致时即本地 3 点） |
| 脚本 | `scripts/daily_calibrate.sh`（`caffeinate -i` 防休眠） |
| 默认货币对 | 产品 8 对：`USD/AUD` `AUD/USD` `EUR/USD` `GBP/USD` `USD/JPY` `USD/CNH` `USD/CAD` `NZD/USD` |
| 参数 | `history=2000` `step=5` `n_sims=4000` `n_iters=40` `max_rows=200`（约 overnight 窗口可跑完；比 full 轻） |
| 仅主对 | `MODE=primary` → 只跑 USD/AUD + AUD/USD |
| 日志 | `output/daily_calib_YYYYMMDD.log`、`output/daily_calib_latest.log` |
| 状态 | `output/daily_calib_status.txt` |
| 装/卸 | `./scripts/install_daily_calibrate.sh` / `./scripts/uninstall_daily_calibrate.sh` |

```bash
# 安装（已对本机执行过则可复查）
./scripts/install_daily_calibrate.sh
launchctl list | grep fxreport

# 卸装
./scripts/uninstall_daily_calibrate.sh

# 手动冒烟（不跑 MC）
DRY_RUN=1 ./scripts/daily_calibrate.sh

# 手动跑一轮（会真校准；AUTO_GIT=0 可跳过 push）
AUTO_GIT=0 ./scripts/daily_calibrate.sh
```

说明：`RunAtLoad=false`，安装后**不会立刻全量校准**，等到当天/次日 03:00。push 失败会写进日志与 status，不静默吞掉。

## 云端如何确认

打开公网 URL → 选货币对 → 页顶「已加载校准参数」+ Holdout；侧栏勾选「使用校准参数」；跑一次后「本次分析审计」参数来源应为校准 JSON 文件名。

详见 `docs/deploy-docker.md`「校准参数如何进云镜像」。

## 相关文件

| 文件 | 作用 |
|------|------|
| `scripts/daily_calibrate.sh` | 每日中等强度校准 + sync + 可选 git push |
| `scripts/install_daily_calibrate.sh` | 安装 launchd（03:00） |
| `scripts/uninstall_daily_calibrate.sh` | 卸载 launchd |
| `scripts/com.fxreport.daily-calibrate.plist.template` | LaunchAgent 模板 |
| `scripts/overnight_calibrate.sh` | 过夜校准入口（更重 sims/iters） |
| `scripts/overnight_calibrate_full.sh` | 多对加强 |
| `scripts/sync_calibrated_to_deploy.sh` | `output/` → `fx_report/data/calibrated/` |
| `fx_report/data/calibrated/` | 打进 Docker / Railway 的 JSON |
| `docs/deploy-docker.md` | 部署与 `APP_PASSWORD` |
