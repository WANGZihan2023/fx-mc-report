# 多货币对 · 最高日高蒙特卡洛情报报告

## 设计目标

**不同货币对的影响因子不同** → 必须先列出「本对需要什么信息」，再抓取与分析。  
**任意 BASE/QUOTE** 都可走同一套七步（目录精调 + 币种自动推断）。

```text
1. 选择货币对          → 锁定该对的影响因子
2. 评估所需信息        → 按因子展开信息需求表（AUD≠EUR≠CNH）
3. 存储有影响语句      → 官方 RSS + **AI 检索员**（投行公开页/搜索）→ 供赋权 / References / 数学分析
4. 评估对货币对的影响
5. 赋予权重
6. 数学分析（蒙特卡洛分档）
7. 规范格式报告
```

## 目录结构

```text
.
├── app.py                 # Streamlit 入口（Cloud 部署指向此文件）
├── run_cli.py             # 命令行入口
├── requirements.txt
├── runtime.txt
├── fx_report/             # 业务包
│   ├── pipeline.py        # 七步流水线编排
│   ├── config/            # API / vault 配置
│   ├── market/            # 货币对、驱动因子、行情抓取
│   ├── news/              # 新闻抓取 / 分类 / LLM / AI 检索
│   ├── model/             # 权重、强度评分、蒙特卡洛、校准
│   ├── data/calibrated/   # **入库**的 Stage-1 JSON（云部署用；优先仍读 output/）
│   ├── report/            # 报告与 diagnostics 文本
│   └── ui/                # Streamlit 页面与 API 面板
├── scripts/               # 一键启动 / 分享 / 部署 / sync_calibrated_to_deploy.sh
├── docs/                  # 分享说明等
└── output/                # 运行产物（git 忽略）
```

影响因子词表：`fx_report/market/pair_drivers.py`  
流水线：`fx_report/pipeline.py`  
AI 检索员：`fx_report/news/ai_research.py`（`--no-ai-research` 可关）

校准参数搜索顺序：`output/calibrated_params_{PAIR}.json`（本地过夜刷新）→ `fx_report/data/calibrated/`（镜像内置）。过夜跑完后同步到入库目录：

```bash
./scripts/sync_calibrated_to_deploy.sh
git add fx_report/data/calibrated/ && git commit -m "Refresh bundled calibrated params" && git push
```
| 货币对 | 典型影响因子 |
|--------|--------------|
| USD/AUD | 地缘、油、Fed、RBA、铁矿、CPI |
| EUR/USD | Fed、ECB、CPI、地缘、增长 |
| USD/JPY | Fed、BOJ、利差、地缘、CPI |
| USD/CNH | Fed、PBOC、中国增长、地缘、利差 |

## 运行

```bash
# CLI（七步流水线）
python run_cli.py --pair USD/AUD --sims 50000 --mode rules
python run_cli.py --pair EUR/USD --sims 50000 --mode hybrid
python run_cli.py --pair EUR/JPY --sims 50000 --mode rules --no-ai-research
# 峰值引擎：path_max（默认，GBM+跳跃路径最大）或 brownian_bridge（连续布朗桥峰值，不含跳跃）
python run_cli.py --pair USD/AUD --sims 50000 --peak-engine brownian_bridge --mode rules

# Stage 0：历史峰值样本 → output/peak_samples_USDAUD.csv
python run_cli.py build-peaks --pair USD/AUD
# 或：python scripts/build_peak_dataset.py --pair USD/AUD

# Stage 1：校准 MC 参数（S=0）→ output/calibrated_params_USDAUD.json
# 同时写 output/calib_oos_summary_USDAUD.json（train vs holdout Brier/logloss/skill）
python run_cli.py calibrate --pair USD/AUD --n-iters 40 --max-rows 80

# 校准档位：
#   smoke  — CLI 默认小样本（如 --n-iters 40 --max-rows 80）用于联调
#   daily  — 每日中等强度（launchd 03:00）：./scripts/daily_calibrate.sh
#               默认 8 对 history=2000 step=5 sims=4000 iters=40；见 docs/calibration_loop.md
#   overnight — 单/双对过夜：./scripts/overnight_calibrate.sh
#               默认 USD/AUD + AUD/USD；HISTORY_DAYS=2000 STEP=5 N_SIMS=8000 N_ITERS=80 MAX_ROWS=250
#               日志 output/overnight_calib_YYYYMMDD.log  PID output/overnight_calib.pid
#   full   — 多货币对加强数据：./scripts/overnight_calibrate_full.sh
#               （或 MODE=full ./scripts/overnight_calibrate.sh）
#               默认 8 对：USD/AUD AUD/USD EUR/USD GBP/USD USD/JPY USD/CNH USD/CAD NZD/USD
#               HISTORY_DAYS=3500 STEP=3 N_SIMS=7000 N_ITERS=70 MAX_ROWS=350
#               日志 output/overnight_calib_full_YYYYMMDD.log  PID output/overnight_calib_full.pid
# 启动示例（防休眠）：
#   caffeinate -i nohup ./scripts/overnight_calibrate_full.sh >/dev/null 2>&1 &

# 历史回测（argmax hit / Brier / log-loss / skill / 可靠性）
python run_cli.py backtest --pair USD/AUD --max-rows 40
# → output/backtest_USDAUD.csv + backtest_USDAUD_summary.json + calib_oos_summary_USDAUD.json
# 评分规则说明：docs/math_goal_E_gneiting.md

# 流水线使用校准参数
python run_cli.py run --pair USD/AUD --calibrated-params output/calibrated_params_USDAUD.json

# 峰值引擎冒烟对比
python scripts/smoke_peak_engines.py

# Goal B：Merton 跳跃 / BB 警告冒烟
python scripts/smoke_jumps_goal_b.py

# 证据链诚实性（无静默模板回退）
python scripts/smoke_evidence_honesty.py

# UI（产品名 FX Analyse）
streamlit run app.py
# 或双击 scripts/open.command（macOS）/ scripts/open.bat（Windows）
```

新闻证据为空时默认 `template_policy=off`（不静默填模板）；`prior_only` / `fallback_warn` 可显式启用。UI 结果页有「本次分析审计」。

**UI 快捷操作**（现价 + 看涨 + 分档就绪后）：
- **双引擎对比**：`path_max` vs `brownian_bridge` 同现价/分档并排概率与 Δ（降采样 MC）
- **历史回测**折叠区：最多 30 行小回测，展示 hit rate / Brier / 明细表
- 运行后在 **「本次分析审计」正下方** 的 **「证据人工标注」** 填写方向/类别（侧栏 ⑨；不必手改 CSV）；说明见 `docs/label_audit.md`
- 标注保存为 `output/label_audit_{PAIR}_{日期}.csv`，亦可下载

## 数据源

权威优先：ECB 行情 + 央行 RSS；可选 FRED/NewsAPI/inbox。  
步骤3可开 AI 检索员（默认开）：LLM 当「脑」迭代拟搜索词，Tavily/Brave/NewsAPI/Google News 当「手」抓真链接；详见 `docs/ai_research.md`。

### 网站上填 Key

`streamlit run app.py` → **「API 配置」**：

1. 免费申请指引（FRED / NewsAPI / …）
2. 付费/增强 Key 表（空着跳过）
3. AI API（Ollama / Groq / OpenAI 兼容）
4. **本机** Streamlit：
   - `仅本次会话`：立刻生效，但刷新会丢
   - `保存到本机 .env`：写入 vault + 仓库 `.env`（gitignore）
5. **Railway 公网站**：
   - 一次性先在 Railway Variables 里设置 `ADMIN_SAVE_TOKEN`
   - 普通访客可继续 `仅本次会话` 使用，不必拿管理员口令
   - 管理员在网页输入 **管理员保存口令** 后，才可点 `保存并持久化到 Railway Variables`
   - `保存并持久化到 Railway Variables`：只会处理 API / LLM 相关白名单变量，优先尝试直接写 **Railway Variables**
   - 若页面检测到当前宿主无法直写 Variables，会自动退回成最省事的兜底：
     - 直接生成可复制的 **Railway Variables `KEY=VALUE` 块**
     - 可下载 `railway-variables.env`
     - 本机一条命令推送：`./scripts/push_env_to_railway.sh railway-variables.env`
   - `仅本次会话`：只对当前网页进程生效，刷新会丢
   - `写入服务器临时磁盘`：只写容器临时盘，**redeploy 后丢失**
   - 为避免泄露，网页导出的持久化块 / `.env` 只包含**本次新填写或上传**的值，不会回显服务器里已有 secrets
   - 启动时若已从服务器环境变量读到 Key，页面会显示 **已永久保存到 Railway Variables**
   - 临时恢复：点「下载 .env」存 Mac → 下次打开网页用「从本机 .env 上传」恢复本会话

Key 本机默认读：`/Users/wangzihan/Desktop/工作_汇率/fx_data_apis/.env` 与仓库 `.env`（`FX_API_ENV_PATH` / `FX_API_ROOT` 可覆盖）。  
云端启动时从 `os.environ`（Railway Variables）加载，表格「已配置」显示「已从环境变量加载」。

## 输出

主交付物为 **FX Analyse 风格 PDF / HTML**（WeasyPrint 渲染，封面概率条、上行/下行、执行摘要、证据库、What to Watch），落在 `output/{PAIR}_fx_analyse.pdf`。  
Markdown 仍写入 `output/{PAIR}_report.md` 作调试副本。

macOS 需 Homebrew 的 pango/gobject（本机一般已有）；脚本会设置 `DYLD_FALLBACK_LIBRARY_PATH`。若 WeasyPrint 不可用，自动回退 ReportLab。

## 相关文档

- 证据人工标注（label_audit）+ Stage 3 标签学习：`docs/label_audit.md`
- 校准日常闭环（overnight / **每日 launchd 03:00** → sync → Railway）：`docs/calibration_loop.md`（本机已可 `./scripts/install_daily_calibrate.sh`）
- 长期分享 / Streamlit 公开设置：`docs/sharing.md`
- **AI 检索员（脑+手迭代搜新闻）**：`docs/ai_research.md`
- **Docker 云部署（WeasyPrint / 澳洲）**：`docs/deploy-docker.md`（含 **`APP_PASSWORD`**）
- 云端 redeploy 标记：`docs/CLOUD_REDEPLOY.txt`
- 部署推送：`scripts/deploy.sh`
- 临时隧道分享：`scripts/share.sh`
- 同步校准到镜像：`scripts/sync_calibrated_to_deploy.sh` → `fx_report/data/calibrated/`

### 云部署访问密码（产品共享密钥）

所有访客打开页面都需输入密码。默认密钥为 **`uniocean`**（代码与 Docker 镜像均如此）。Railway **Variables** 设置 `APP_PASSWORD=你的密码`（或 `FX_REPORT_PASSWORD`）可覆盖默认。