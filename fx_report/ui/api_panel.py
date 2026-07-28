"""
Streamlit：API 申请指引 + 表格填 Key（行情 / 搜索 / AI）。

- 免费源：展示逐步申请说明 + 填入 Key
- 付费/增强：表格填写
- AI：单独表格（Ollama / Groq / 自定义 OpenAI 兼容）
填写后写入 session，并注入 os.environ，供 pipeline 立即使用。
可选「保存到本机 vault .env」（不进 git）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from fx_report.config.api_config import (
    PROVIDERS,
    apply_runtime_overrides,
    env_path,
    is_set,
    load_config,
    project_env_path,
    save_keys_to_local,
    status_text,
)

# ---------------------------------------------------------------------------
# 目录：免费申请指引
# ---------------------------------------------------------------------------

FREE_SIGNUP_GUIDES: list[dict[str, str]] = [
    {
        "name": "FRED（强烈推荐）",
        "env_key": "FRED_API_KEY",
        "cost": "完全免费",
        "why": "宏观序列、部分汇率；增强行情与 CPI/利率证据",
        "url": "https://fred.stlouisfed.org/docs/api/api_key.html",
        "steps": (
            "1. 打开链接 → Request API Key\n"
            "2. 用工作邮箱注册/登录圣路易斯联储账号\n"
            "3. 填用途（写 FX research / internal tool 即可）\n"
            "4. 邮件里拿到 Key → 粘贴到下方「免费 Key」表"
        ),
    },
    {
        "name": "NewsAPI（开发者免费档）",
        "env_key": "NEWSAPI_KEY",
        "cost": "免费档（有日限额）",
        "why": "财经新闻标题；AI 检索员补搜",
        "url": "https://newsapi.org/register",
        "steps": (
            "1. Register 注册（个人/开发者）\n"
            "2. 登录后在 Account → API Key 复制\n"
            "3. 注意：免费档对生产域名有限制，本机/内网试用一般够用\n"
            "4. 粘贴到下方表格"
        ),
    },
    {
        "name": "Alpha Vantage（免费档）",
        "env_key": "ALPHA_VANTAGE_API_KEY",
        "cost": "免费档（有分钟限额）",
        "why": "备用外汇报价（ECB 失败时）",
        "url": "https://www.alphavantage.co/support/#api-key",
        "steps": (
            "1. 打开链接 → 填姓名与邮箱 Claim API key\n"
            "2. 邮件收到 Key\n"
            "3. 粘贴到下方表格（每日请求有限，够做报告）"
        ),
    },
    {
        "name": "Finnhub（免费档）",
        "env_key": "FINNHUB_API_KEY",
        "cost": "免费档",
        "why": "财经日历 / 新闻增强",
        "url": "https://finnhub.io/register",
        "steps": (
            "1. 注册账号\n"
            "2. Dashboard 复制 API Key\n"
            "3. 粘贴到下方表格"
        ),
    },
    {
        "name": "Groq（LLM 免费额度）",
        "env_key": "GROQ_API_KEY",
        "cost": "有免费额度",
        "why": "云端大模型：证据判定 + AI 检索抽取（本机无 Ollama 时用）",
        "url": "https://console.groq.com/keys",
        "steps": (
            "1. 打开 Groq Console 注册登录\n"
            "2. API Keys → Create API Key\n"
            "3. 复制后填到「AI API」表的 API Key，通道选 Groq\n"
            "4. 模型可用 llama-3.1-8b-instant"
        ),
    },
    {
        "name": "Ollama（本机，无需 Key）",
        "env_key": "",
        "cost": "免费",
        "why": "本机大模型，Key 不外传、无云费用",
        "url": "https://ollama.com/download",
        "steps": (
            "1. 安装 Ollama（Mac/Windows/Linux）\n"
            "2. 终端执行：ollama pull llama3.1\n"
            "3. 保持 Ollama 运行；网站「AI API」通道选 Ollama\n"
            "4. Base URL 默认 http://127.0.0.1:11434/v1，Key 填 ollama"
        ),
    },
]

# 付费 / 增强（网站表格填写；空=跳过）
PAID_OR_OPTIONAL: list[dict[str, str]] = [
    {
        "env_key": "TWELVE_DATA_API_KEY",
        "name": "Twelve Data",
        "tier": "增强/付费档更稳",
        "why": "外汇 OHLC",
        "signup": "https://twelvedata.com/",
    },
    {
        "env_key": "FMP_API_KEY",
        "name": "Financial Modeling Prep",
        "tier": "多为付费",
        "why": "行情/基本面",
        "signup": "https://site.financialmodelingprep.com/developer",
    },
    {
        "env_key": "POLYGON_API_KEY",
        "name": "Polygon.io",
        "tier": "付费为主",
        "why": "市场数据",
        "signup": "https://polygon.io/",
    },
    {
        "env_key": "OPENEXCHANGERATES_APP_ID",
        "name": "Open Exchange Rates",
        "tier": "免费档很窄/付费",
        "why": "汇率中间价",
        "signup": "https://openexchangerates.org/",
    },
    {
        "env_key": "TAVILY_API_KEY",
        "name": "Tavily Search",
        "tier": "有试用额度，超量付费",
        "why": "AI 检索员网页搜索",
        "signup": "https://tavily.com/",
    },
    {
        "env_key": "BRAVE_SEARCH_API_KEY",
        "name": "Brave Search API",
        "tier": "有免费档/付费",
        "why": "AI 检索员搜索",
        "signup": "https://brave.com/search/api/",
    },
    {
        "env_key": "BROKER_REST_BASE_URL",
        "name": "券商 REST Base URL",
        "tier": "公司采购",
        "why": "企业网关地址（非 Key）",
        "signup": "",
    },
    {
        "env_key": "BROKER_REST_TOKEN",
        "name": "券商 REST Token",
        "tier": "公司采购",
        "why": "企业网关令牌",
        "signup": "",
    },
]


def _mask(v: str) -> str:
    v = (v or "").strip()
    if not v:
        return ""
    if len(v) <= 8:
        return "••••"
    return v[:4] + "…" + v[-4:]


def _session_keys() -> dict[str, str]:
    return dict(st.session_state.get("api_keys_ui") or {})


def _set_session_keys(keys: dict[str, str]) -> None:
    cleaned = {k: (v or "").strip() for k, v in keys.items() if k}
    st.session_state["api_keys_ui"] = cleaned
    apply_runtime_overrides(cleaned)


def _seed_from_vault() -> dict[str, str]:
    cfg = load_config()
    out: dict[str, str] = {}
    for k in list(PROVIDERS) + [
        "BROKER_REST_BASE_URL",
        "GROQ_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
    ]:
        if is_set(cfg, k):
            out[k] = cfg[k]
    sess = _session_keys()
    for k, v in sess.items():
        if v:
            out[k] = v
    return out


def _editor_table(
    rows: list[dict[str, str]],
    key_col: str,
    value_map: dict[str, str],
    *,
    editor_key: str,
) -> dict[str, str]:
    """Show editable table; return env_key -> value from editor."""
    data = []
    for r in rows:
        ek = r[key_col]
        data.append(
            {
                "名称": r.get("name", ek),
                "环境变量": ek,
                "档位": r.get("tier", r.get("cost", "")),
                "用途": r.get("why", ""),
                "申请页": r.get("signup", r.get("url", "")) or None,
                "API Key / 值": value_map.get(ek, ""),
                "已配置": "是" if value_map.get(ek, "").strip() else "否",
            }
        )
    df = pd.DataFrame(data)
    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        disabled=["名称", "环境变量", "档位", "用途", "申请页", "已配置"],
        column_config={
            "申请页": st.column_config.LinkColumn("申请页", display_text="打开"),
            "API Key / 值": st.column_config.TextColumn(
                "API Key / 值",
                help="粘贴 Key；留空=不启用该源",
                width="medium",
            ),
        },
        key=editor_key,
    )
    out: dict[str, str] = {}
    for _, row in edited.iterrows():
        ek = str(row["环境变量"])
        out[ek] = str(row["API Key / 值"] or "").strip()
    return out


def render_api_settings_panel() -> dict[str, Any]:
    """
    主区：免费申请教学 + 付费表格 + AI 表格。
    返回当前生效的 llm 相关选项，供流水线使用。
    """
    if "api_keys_ui" not in st.session_state:
        st.session_state["api_keys_ui"] = _seed_from_vault()

    st.subheader("API 配置")
    st.caption(
        "免费按指引申请；付费/AI 有 Key 再填。空=跳过。"
        "「应用到本会话」刷新即丢；点「保存 API 到本机 .env」才会落盘（已 gitignore，勿提交）。"
    )
    vault_cfg = load_config()
    disk_on = [k for k in PROVIDERS if is_set(vault_cfg, k)]
    if not disk_on:
        st.warning(
            "本机 `.env` / vault 里目前没有行情/新闻 Key。"
            "填表后务必点 **保存 API 到本机 .env**，否则刷新或重启 Streamlit 会全部丢失。"
        )

    tab_free, tab_paid, tab_ai, tab_status = st.tabs(
        ["① 免费申请指引", "② 付费/增强 Key 表", "③ AI API", "④ 当前状态"]
    )

    free_vals: dict[str, str] = {}
    paid_vals: dict[str, str] = {}
    ai_vals: dict[str, str] = {}

    with tab_free:
        st.markdown("#### 怎么申请（按顺序做即可）")
        guide_df = pd.DataFrame(
            [
                {
                    "服务": g["name"],
                    "费用": g["cost"],
                    "用途": g["why"],
                    "申请链接": g["url"],
                    "环境变量": g["env_key"] or "（无需 Key）",
                }
                for g in FREE_SIGNUP_GUIDES
            ]
        )
        st.dataframe(
            guide_df,
            hide_index=True,
            use_container_width=True,
            column_config={"申请链接": st.column_config.LinkColumn("申请链接")},
        )

        for g in FREE_SIGNUP_GUIDES:
            with st.expander(f"步骤详解 · {g['name']}", expanded=False):
                st.markdown(f"**为什么要：** {g['why']}")
                st.markdown(f"**申请页：** {g['url']}")
                st.code(g["steps"], language=None)

        st.markdown("#### 免费 Key 填写表（申请到后粘贴）")
        free_rows = [
            {
                "env_key": g["env_key"],
                "name": g["name"],
                "tier": g["cost"],
                "why": g["why"],
                "signup": g["url"],
            }
            for g in FREE_SIGNUP_GUIDES
            if g.get("env_key")
        ]
        free_vals = _editor_table(
            free_rows, "env_key", _session_keys(), editor_key="free_keys_editor"
        )

    with tab_paid:
        st.markdown(
            "没有采购就**全部留空**——系统会继续用 ECB + 央行 RSS + 白名单公开页。"
            "公司若已买终端/数据商，把 Key 填进表即可增强。"
        )
        paid_vals = _editor_table(
            PAID_OR_OPTIONAL, "env_key", _session_keys(), editor_key="paid_keys_editor"
        )

    with tab_ai:
        st.markdown(
            "用于步骤 3（AI 检索抽取）与步骤 4（证据判定）。"
            "推荐：本机 **Ollama**；或云端 **Groq** 免费额度；也可用公司 OpenAI 兼容网关。"
        )
        keys = _session_keys()
        channel = st.selectbox(
            "AI 通道",
            options=["ollama", "groq", "openai_compatible"],
            format_func=lambda x: {
                "ollama": "Ollama 本机（免费，Key=ollama）",
                "groq": "Groq 云端（填 GROQ / LLM Key）",
                "openai_compatible": "OpenAI 兼容（公司网关/DeepSeek/…）",
            }[x],
            index=0,
            key="ai_channel",
        )

        if channel == "ollama":
            defaults = {
                "LLM_API_KEY": keys.get("LLM_API_KEY") or "ollama",
                "LLM_BASE_URL": keys.get("LLM_BASE_URL") or "http://127.0.0.1:11434/v1",
                "LLM_MODEL": keys.get("LLM_MODEL") or "llama3.1:latest",
            }
        elif channel == "groq":
            defaults = {
                "LLM_API_KEY": keys.get("GROQ_API_KEY") or keys.get("LLM_API_KEY") or "",
                "LLM_BASE_URL": "https://api.groq.com/openai/v1",
                "LLM_MODEL": keys.get("LLM_MODEL") or "llama-3.1-8b-instant",
            }
        else:
            defaults = {
                "LLM_API_KEY": keys.get("LLM_API_KEY") or keys.get("OPENAI_API_KEY") or "",
                "LLM_BASE_URL": keys.get("LLM_BASE_URL") or "https://api.openai.com/v1",
                "LLM_MODEL": keys.get("LLM_MODEL") or "gpt-4o-mini",
            }

        ai_df = pd.DataFrame(
            [
                {
                    "字段": "API Key",
                    "环境变量": "LLM_API_KEY",
                    "说明": "Groq/OpenAI/兼容网关的密钥；Ollama 填 ollama",
                    "值": defaults["LLM_API_KEY"],
                },
                {
                    "字段": "Base URL",
                    "环境变量": "LLM_BASE_URL",
                    "说明": "OpenAI 兼容接口根路径（…/v1）",
                    "值": defaults["LLM_BASE_URL"],
                },
                {
                    "字段": "Model",
                    "环境变量": "LLM_MODEL",
                    "说明": "模型名，如 llama3.1:latest / llama-3.1-8b-instant",
                    "值": defaults["LLM_MODEL"],
                },
            ]
        )
        ai_edited = st.data_editor(
            ai_df,
            hide_index=True,
            use_container_width=True,
            disabled=["字段", "环境变量", "说明"],
            column_config={
                "值": st.column_config.TextColumn("值", width="large"),
            },
            key="ai_api_editor",
        )
        ai_vals = {
            str(r["环境变量"]): str(r["值"] or "").strip() for _, r in ai_edited.iterrows()
        }
        if channel == "groq" and ai_vals.get("LLM_API_KEY"):
            ai_vals["GROQ_API_KEY"] = ai_vals["LLM_API_KEY"]

        st.info(
            "Key **不会**写进报告正文，也不会自动上传。"
            "若部署到公网 Streamlit，请改用服务端 Secrets，避免访客互抢额度。"
        )

    with tab_status:
        st.code(status_text(), language=None)
        sess = _session_keys()
        if sess:
            st.markdown("#### 本会话已填（脱敏）")
            st.dataframe(
                pd.DataFrame(
                    [{"环境变量": k, "值": _mask(v)} for k, v in sorted(sess.items()) if v]
                ),
                hide_index=True,
                use_container_width=True,
            )

    merged = dict(_session_keys())
    merged.update(free_vals)
    merged.update(paid_vals)
    merged.update(ai_vals)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("应用到本会话", type="primary", use_container_width=True):
            _set_session_keys(merged)
            st.success("已注入本会话环境，可直接跑流水线。注意：未点保存则刷新会丢。")
    with c2:
        if st.button("保存 API 到本机 .env", use_container_width=True):
            nonempty = {k: v for k, v in merged.items() if (v or "").strip()}
            if not nonempty:
                st.error("表格里没有非空 Key，无法保存。请先粘贴再点保存。")
            else:
                _set_session_keys(merged)
                paths = save_keys_to_local(merged)
                st.success(
                    "已写入本机（gitignore，勿提交）：\n\n"
                    + "\n".join(f"- `{p}`" for p in paths)
                )
    with c3:
        st.caption(
            f"落盘位置：vault `{env_path()}` 与仓库 `{project_env_path()}`。"
            "只点「应用到本会话」不会写盘。"
        )

    _set_session_keys(merged)

    return {
        "llm_key": merged.get("LLM_API_KEY") or merged.get("GROQ_API_KEY") or "",
        "llm_base": merged.get("LLM_BASE_URL") or "",
        "llm_model": merged.get("LLM_MODEL") or "",
        "api_keys": merged,
    }
