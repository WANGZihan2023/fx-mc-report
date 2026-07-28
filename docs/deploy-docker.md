# 部署：Streamlit + Docker + WeasyPrint（澳洲同事）

目标：云上也能下载 FX Analyse 风格 PDF（WeasyPrint），不必挂自己电脑。

推荐平台：**Railway**（简单）或 **Render**（类似）。区域尽量选 **Sydney / Singapore**。

---

## 0. 本地先验证 Docker（可选）

本机已装 Docker Desktop 时：

```bash
cd /Users/wangzihan/Desktop/工作_汇率/usd_aud_mc_report
docker build -t fx-mc-report .
docker run --rm -p 8501:8501 \
  -e GROQ_API_KEY=你的key \
  fx-mc-report
```

浏览器打开 http://127.0.0.1:8501 ，跑一次报告并下载 PDF。

---

## 1. 代码推到 GitHub

仓库若已是 `fx-mc-report`，把含 `Dockerfile` 的改动 push 到 `main`：

```bash
cd /Users/wangzihan/Desktop/工作_汇率/usd_aud_mc_report
git add Dockerfile .dockerignore railway.toml docs/deploy-docker.md requirements.txt
git status
# 确认无 .env / secrets 后：
git commit -m "Add Docker image with WeasyPrint for cloud deploy"
git push
```

**不要**提交：`.env`、`.streamlit/secrets.toml`、API Key。

---

## 2. Railway 部署（推荐）

1. 打开 https://railway.app ，用 GitHub 登录  
2. **New Project** → **Deploy from GitHub repo** → 选 `fx-mc-report`  
3. 若未自动识别 Docker：Settings → Build → Builder = Dockerfile  
4. **Settings → Networking → Generate Domain**（得到 `https://xxxx.up.railway.app`）  
5. **Variables** 里按需添加（有就填，没有可空）：

> **重要：** 在 Railway 网站上点「保存到本机 .env」**不会**写到你的 Mac，只写容器临时磁盘，**redeploy 后丢失**。云端请把 Key 配进 Variables；或本机 `streamlit run` 后再保存；或用页面「下载 .env」。

| 变量 | 说明 |
|------|------|
| `APP_PASSWORD`（或 `FX_REPORT_PASSWORD`） | **产品共享访问密钥**；镜像默认 `uniocean`。平台 Variables 可覆盖。 |
| `NEWSAPI_KEY` / `FINNHUB_API_KEY` | **References / 证据条数**主来源 |
| `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY` | AI 检索员搜索「手」（强烈建议） |
| `GROQ_API_KEY` / `DEEPSEEK_API_KEY` / `LLM_API_KEY` | LLM「脑」：迭代拟搜索词 + 判定（**不**虚构 URL） |
| `LLM_BASE_URL` | DeepSeek 须 `https://api.deepseek.com/v1` |
| `FRED_API_KEY` | 行情增强 |
| `FX_PDF_ENGINE` | 默认已是 `weasyprint`，一般不用改 |

6. 区域：若控制台可选 Region，优先 **Sydney** 或东南亚  
7. 等 Deploy 变绿 → 打开公网域名 → 输访问密码 → 跑报告 → 下载 PDF  

分享给澳洲同事：把这个 `https://….up.railway.app` 发给他们，并告知共享密钥（默认 `uniocean`；若你在 Variables 改过则以你设的为准）。

### 校准参数如何进云镜像

`output/` 被 `.gitignore` / `.dockerignore` 排除，Railway **不会**自动带上过夜校准结果。仓库内已跟踪：

`fx_report/data/calibrated/calibrated_params_*.json` + `calib_oos_summary_*.json`（8 对）

运行时优先读本地 `output/`，否则读该内置目录。过夜校准后刷新并 redeploy：

```bash
./scripts/sync_calibrated_to_deploy.sh
git add fx_report/data/calibrated/
git commit -m "Refresh bundled calibrated params"
git push   # Railway 自动重建镜像
```

更完整的「过夜 / **每日 launchd** → sync → push → 确认」步骤见 **`docs/calibration_loop.md`**（本机可 `./scripts/install_daily_calibrate.sh`，默认每天 03:00 Asia/Shanghai）。

**如何确认云端已加载校准**：打开公网 URL → 选货币对 → 页顶应显示「已加载校准参数」+ Holdout hit rate / Brier；侧栏勾选「使用校准参数」；跑一次分析后「本次分析审计」里参数来源应为校准 JSON 文件名（非「默认先验」）。

### 访问密码（产品共享密钥）

| 变量 | 说明 |
|------|------|
| **`APP_PASSWORD`** | 产品共享访问密钥。Docker 镜像默认 `uniocean`；未设 env 时应用代码同样默认 `uniocean` |
| `FX_REPORT_PASSWORD` | 与上等效的别名 |

所有访客打开页面都需先输密码。可用平台 **Variables** 覆盖默认值；不设则使用产品默认密钥。

---

## 3. Render 部署（备选）
1. https://render.com → New → **Web Service** → 连 GitHub 仓库  
2. Runtime: **Docker**  
3. Instance：Free 可试；正式用 Starter  
4. Region：尽量靠近澳洲（Singapore 常见）  
5. Environment 同上填 Key  
6. 生成 URL 后测试 PDF 下载  

---

## 4. 安全注意

- 链接等于半公开：所有访客需输入产品共享密钥（默认 `uniocean`；可用 `APP_PASSWORD` / `FX_REPORT_PASSWORD` 覆盖）  
- Key 只放平台 **Environment Variables**，不要写进代码  
- 不构成投资建议；内部分享即可  

---

## 5. 常见问题

| 现象 | 处理 |
|------|------|
| 构建失败找不到包 | 看 Build Log；确认 `requirements.txt` 已含 `weasyprint` |
| PDF 下载失败 / 回退丑陋版 | 进容器确认有 pango；镜像已含 `libpango`，一般不用改 |
| 页面能开但行情失败 | 云上填 `FRED_API_KEY` 等；ECB 无 Key 通常仍可用 |
| 休眠/冷启动慢 | 免费档会睡；付费常开即可 |

仓库内文件：

- `Dockerfile` — 镜像定义  
- `.dockerignore` — 减小构建体积  
- `railway.toml` — Railway 使用 Dockerfile  
