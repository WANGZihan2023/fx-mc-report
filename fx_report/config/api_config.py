#!/usr/bin/env python3
"""Load API keys from the shared FX vault (fill key → fetch; empty → skip)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_VAULT = Path("/Users/wangzihan/Desktop/工作_汇率/fx_data_apis")
DEFAULT_ENV = DEFAULT_VAULT / ".env"

PROVIDERS = {
    "FRED_API_KEY": "FRED 宏观/汇率序列",
    "ALPHA_VANTAGE_API_KEY": "Alpha Vantage 汇率",
    "TWELVE_DATA_API_KEY": "Twelve Data FX OHLC",
    "FINNHUB_API_KEY": "Finnhub 日历/新闻",
    "FMP_API_KEY": "FMP 行情",
    "POLYGON_API_KEY": "Polygon",
    "OPENEXCHANGERATES_APP_ID": "Open Exchange Rates",
    "NEWSAPI_KEY": "NewsAPI 新闻",
    "TAVILY_API_KEY": "Tavily 网页搜索（AI 检索员）",
    "BRAVE_SEARCH_API_KEY": "Brave Search（AI 检索员）",
    "BROKER_REST_TOKEN": "券商/企业 REST",
}

MARKET_KEYS = (
    "TWELVE_DATA_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "FRED_API_KEY",
    "FMP_API_KEY",
    "POLYGON_API_KEY",
    "OPENEXCHANGERATES_APP_ID",
    "BROKER_REST_TOKEN",
)

NEWS_KEYS = ("NEWSAPI_KEY", "FINNHUB_API_KEY")
SEARCH_KEYS = ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY", "NEWSAPI_KEY")


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def env_path() -> Path:
    return Path(os.environ.get("FX_API_ENV_PATH", DEFAULT_ENV))


def _repo_root() -> Path:
    """usd_aud_mc_report/ （fx_report/config/api_config.py → parents[2]）"""
    return Path(__file__).resolve().parents[2]


def load_config() -> dict[str, str]:
    """Merge vault .env with process env (process wins)."""
    cfg = _parse_env_file(env_path())
    # Also accept project-root .env（仓库根目录，勿提交）
    local = _repo_root() / ".env"
    for k, v in _parse_env_file(local).items():
        if v and not cfg.get(k):
            cfg[k] = v
    keys = list(PROVIDERS) + [
        "FX_API_ROOT",
        "BROKER_REST_BASE_URL",
        "FX_API_DRY_RUN",
        "FX_API_TIMEOUT",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "TAVILY_API_KEY",
        "BRAVE_SEARCH_API_KEY",
    ]
    for k in keys:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    cfg.setdefault("FX_API_ROOT", str(DEFAULT_VAULT))
    cfg.setdefault("FX_API_TIMEOUT", "20")
    cfg.setdefault("FX_API_DRY_RUN", "0")
    return cfg


def is_set(cfg: dict[str, str], key: str) -> bool:
    return bool(cfg.get(key, "").strip())


def apply_runtime_overrides(overrides: dict[str, str] | None) -> None:
    """Inject UI-filled keys into process env so load_config() / fetchers see them."""
    if not overrides:
        return
    for k, v in overrides.items():
        val = (v or "").strip()
        if not val:
            continue
        os.environ[k] = val


def save_keys_to_env(overrides: dict[str, str], path: Path | None = None) -> Path:
    """
    Merge non-empty keys into vault .env (create if missing).
    Does not delete existing keys that are absent from overrides.
    """
    path = path or env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _parse_env_file(path) if path.is_file() else {}
    for k, v in overrides.items():
        val = (v or "").strip()
        if val:
            existing[k] = val

    # Preserve a short header; rewrite known keys in stable order
    order = [
        "FX_API_ROOT",
        "FRED_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "TWELVE_DATA_API_KEY",
        "FINNHUB_API_KEY",
        "FMP_API_KEY",
        "POLYGON_API_KEY",
        "OPENEXCHANGERATES_APP_ID",
        "NEWSAPI_KEY",
        "TAVILY_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "BROKER_REST_BASE_URL",
        "BROKER_REST_TOKEN",
        "GROQ_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "FX_API_DRY_RUN",
        "FX_API_TIMEOUT",
    ]
    lines = [
        "# FX Data APIs — 由网站「API 配置」写入；勿提交 git",
        f"FX_API_ROOT={existing.get('FX_API_ROOT', str(DEFAULT_VAULT))}",
        "",
    ]
    seen: set[str] = {"FX_API_ROOT"}
    for k in order:
        if k == "FX_API_ROOT":
            continue
        if k in existing:
            lines.append(f"{k}={existing[k]}")
            seen.add(k)
    for k, v in sorted(existing.items()):
        if k not in seen:
            lines.append(f"{k}={v}")
    if "FX_API_DRY_RUN" not in existing:
        lines.append("FX_API_DRY_RUN=0")
    if "FX_API_TIMEOUT" not in existing:
        lines.append("FX_API_TIMEOUT=20")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    apply_runtime_overrides(overrides)
    return path


def timeout_s(cfg: dict[str, str] | None = None) -> int:
    cfg = cfg or load_config()
    try:
        return max(5, int(cfg.get("FX_API_TIMEOUT", "20")))
    except ValueError:
        return 20


def vault_paths(cfg: dict[str, str] | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    root = Path(cfg.get("FX_API_ROOT", DEFAULT_VAULT))
    return {
        "root": root,
        "env": env_path(),
        "inbox": root / "inbox",
        "cache": root / "cache",
    }


def enabled_providers(cfg: dict[str, str] | None = None) -> list[str]:
    cfg = cfg or load_config()
    return [k for k in PROVIDERS if is_set(cfg, k)]


def has_market_api(cfg: dict[str, str] | None = None) -> bool:
    cfg = cfg or load_config()
    return any(is_set(cfg, k) for k in MARKET_KEYS)


def has_news_api(cfg: dict[str, str] | None = None) -> bool:
    cfg = cfg or load_config()
    return any(is_set(cfg, k) for k in NEWS_KEYS)


def inbox_files(cfg: dict[str, str] | None = None) -> list[Path]:
    inbox = vault_paths(cfg)["inbox"]
    if not inbox.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(inbox.iterdir()):
        if p.name.startswith(".") or p.name == "README.md":
            continue
        if p.suffix.lower() in {".pdf", ".csv", ".md", ".txt"}:
            out.append(p)
    return out


def status_text(cfg: dict[str, str] | None = None) -> str:
    cfg = cfg or load_config()
    paths = vault_paths(cfg)
    lines = [
        f"API vault: {paths['root']}",
        f"env: {paths['env']} ({'exists' if paths['env'].is_file() else 'MISSING'})",
        "",
        "Market: ECB/Frankfurter → FRED → Twelve → Alpha",
        "News: Fed/RBA/ECB/BOE RSS → inbox → NewsAPI/Finnhub",
        "AI research: whitelist bank pages → Tavily/Brave/NewsAPI → LLM extract",
        "",
        "Providers:",
    ]
    for key, label in PROVIDERS.items():
        flag = "ON " if is_set(cfg, key) else "off"
        lines.append(f"  [{flag}] {label} ({key})")
    files = inbox_files(cfg)
    lines += ["", f"inbox files: {len(files)}"]
    for f in files[:8]:
        lines.append(f"  - {f.name}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(status_text())
