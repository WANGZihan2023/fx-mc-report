#!/usr/bin/env python3
"""Load API keys from the shared FX vault (fill key → fetch; empty → skip)."""

from __future__ import annotations

import os
from pathlib import Path

# Dev Mac shared vault (local Streamlit). On Railway/cloud this path does not exist —
# we fall back to a writable in-container dir (ephemeral; prefer platform Variables).
MAC_DEFAULT_VAULT = Path("/Users/wangzihan/Desktop/工作_汇率/fx_data_apis")
DEFAULT_VAULT = MAC_DEFAULT_VAULT  # kept for backwards-compatible imports
DEFAULT_ENV = MAC_DEFAULT_VAULT / ".env"

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

# Keys we may persist from the API 配置 UI (order used when writing .env).
PERSIST_KEYS = (
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
    "DEEPSEEK_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "FX_API_DRY_RUN",
    "FX_API_TIMEOUT",
)


def is_cloud_runtime() -> bool:
    """True on Railway / Render / Fly / Streamlit Community Cloud (not the user's Mac)."""
    if os.environ.get("FX_FORCE_CLOUD_RUNTIME", "").strip() in {"1", "true", "yes"}:
        return True
    if os.environ.get("FX_FORCE_LOCAL_RUNTIME", "").strip() in {"1", "true", "yes"}:
        return False
    markers = (
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_STATIC_URL",
        "RENDER",
        "RENDER_SERVICE_ID",
        "FLY_APP_NAME",
        "FLY_MACHINE_ID",
        "STREAMLIT_SHARING_MODE",
    )
    return any(os.environ.get(k) for k in markers)


def _repo_root() -> Path:
    """usd_aud_mc_report/ （fx_report/config/api_config.py → parents[2]）"""
    return Path(__file__).resolve().parents[2]


def _cloud_vault_root() -> Path:
    override = (os.environ.get("FX_API_ROOT") or "").strip()
    if override:
        return Path(override)
    return _repo_root() / ".fx_vault"


def default_vault_root() -> Path:
    """Writable vault root for this runtime (Mac shared path or cloud fallback)."""
    override = (os.environ.get("FX_API_ROOT") or "").strip()
    if override:
        return Path(override)
    if is_cloud_runtime() or not MAC_DEFAULT_VAULT.parent.is_dir():
        return _cloud_vault_root()
    return MAC_DEFAULT_VAULT


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
    override = (os.environ.get("FX_API_ENV_PATH") or "").strip()
    if override:
        return Path(override)
    return default_vault_root() / ".env"


def project_env_path() -> Path:
    """仓库根目录 .env（已 gitignore）。"""
    return _repo_root() / ".env"


def merge_nonempty(existing: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    """
    Merge overrides into existing.
    Never overwrite a non-empty stored value with a blank form field.
    """
    out = dict(existing)
    for k, v in overrides.items():
        if not k:
            continue
        val = (v or "").strip()
        if not val:
            continue
        out[k] = val
    return out


def mask_secret(v: str) -> str:
    v = (v or "").strip()
    if not v:
        return "(empty)"
    if len(v) <= 8:
        return "••••"
    return f"{v[:4]}…{v[-4:]} (len={len(v)})"


def load_config() -> dict[str, str]:
    """Merge vault .env + project .env with process env (process wins)."""
    cfg: dict[str, str] = {}
    # Vault first, then project .env fills gaps (same paths used by save).
    for path in (env_path(), project_env_path()):
        for k, v in _parse_env_file(path).items():
            if v and not cfg.get(k):
                cfg[k] = v
    keys = list(PROVIDERS) + [
        "FX_API_ROOT",
        "BROKER_REST_BASE_URL",
        "FX_API_DRY_RUN",
        "FX_API_TIMEOUT",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "TAVILY_API_KEY",
        "BRAVE_SEARCH_API_KEY",
    ]
    for k in keys:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    cfg.setdefault("FX_API_ROOT", str(default_vault_root()))
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


def _write_env_file(path: Path, existing: dict[str, str]) -> None:
    """Rewrite .env with stable key order. Caller merges overrides into existing first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root_default = str(default_vault_root())
    lines = [
        "# FX Data APIs — 由网站「API 配置」写入；勿提交 git",
        f"FX_API_ROOT={existing.get('FX_API_ROOT', root_default)}",
        "",
    ]
    seen: set[str] = {"FX_API_ROOT"}
    for k in PERSIST_KEYS:
        if k == "FX_API_ROOT":
            continue
        if k in existing:
            # Persist non-empty only — never write blank over a key that was set.
            val = (existing.get(k) or "").strip()
            if val:
                lines.append(f"{k}={val}")
                seen.add(k)
    for k, v in sorted(existing.items()):
        if k in seen:
            continue
        val = (v or "").strip()
        if not val:
            continue
        lines.append(f"{k}={val}")
        seen.add(k)
    if "FX_API_DRY_RUN" not in seen:
        lines.append(f"FX_API_DRY_RUN={existing.get('FX_API_DRY_RUN', '0') or '0'}")
    if "FX_API_TIMEOUT" not in seen:
        lines.append(f"FX_API_TIMEOUT={existing.get('FX_API_TIMEOUT', '20') or '20'}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_keys_to_env(overrides: dict[str, str], path: Path | None = None) -> Path:
    """
    Merge non-empty keys into vault .env (create if missing).
    Does not delete existing keys that are absent from overrides.
    Never overwrites existing non-empty values with blanks.
    """
    path = path or env_path()
    existing = _parse_env_file(path) if path.is_file() else {}
    existing = merge_nonempty(existing, overrides)
    _write_env_file(path, existing)
    apply_runtime_overrides(overrides)
    return path


def verify_env_file(path: Path) -> dict[str, str]:
    """Re-read a .env and return key → masked status for UI confirmation."""
    raw = _parse_env_file(path) if path.is_file() else {}
    return {k: mask_secret(raw.get(k, "")) for k in sorted(raw) if k != "FX_API_ROOT"}


def env_file_download_bytes(keys: dict[str, str]) -> bytes:
    """Build a downloadable .env body (non-empty keys only)."""
    existing = merge_nonempty({}, keys)
    existing.setdefault("FX_API_ROOT", str(default_vault_root()))
    existing.setdefault("FX_API_DRY_RUN", "0")
    existing.setdefault("FX_API_TIMEOUT", "20")
    lines = ["# FX Data APIs — downloaded from UI; keep private, do not commit", ""]
    for k in PERSIST_KEYS:
        val = (existing.get(k) or "").strip()
        if val:
            lines.append(f"{k}={val}")
    for k, v in sorted(existing.items()):
        if k in PERSIST_KEYS:
            continue
        val = (v or "").strip()
        if val:
            lines.append(f"{k}={val}")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def save_keys_to_local(overrides: dict[str, str]) -> list[Path]:
    """
    同时写入 vault .env 与仓库根 .env（双份落盘，刷新/重启不丢）。
    仅合并非空值；绝不把已有非空 Key 用空白表单盖掉。

    Both targets receive the same merged snapshot (vault ∪ project ∪ overrides),
    so load paths stay consistent.

    On Railway/cloud this writes the *container* disk (ephemeral), not the user's Mac.
    """
    apply_runtime_overrides(overrides)
    vault = env_path()
    project = project_env_path()
    existing: dict[str, str] = {}
    for path in (vault, project):
        if path.is_file():
            existing = merge_nonempty(existing, _parse_env_file(path))
    existing = merge_nonempty(existing, overrides)
    existing.setdefault("FX_API_ROOT", str(default_vault_root()))

    written: list[Path] = []
    for path in (vault, project):
        _write_env_file(path, existing)
        if not path.is_file():
            raise OSError(f"env write failed: {path}")
        written.append(path.resolve())
    return written


def timeout_s(cfg: dict[str, str] | None = None) -> int:
    cfg = cfg or load_config()
    try:
        return max(5, int(cfg.get("FX_API_TIMEOUT", "20")))
    except ValueError:
        return 20


def vault_paths(cfg: dict[str, str] | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    root = Path(cfg.get("FX_API_ROOT", str(default_vault_root())))
    return {
        "root": root,
        "env": env_path(),
        "project_env": project_env_path(),
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
    cloud = is_cloud_runtime()
    lines = [
        f"runtime: {'cloud/Railway (server disk)' if cloud else 'local Mac'}",
        f"API vault: {paths['root']}",
        f"vault env: {paths['env']} ({'exists' if paths['env'].is_file() else 'MISSING'})",
        f"project env: {paths['project_env']} ({'exists' if paths['project_env'].is_file() else 'MISSING'})",
        "",
        "Market: ECB/Frankfurter → FRED → Twelve → Alpha",
        "News: Fed/RBA/ECB/BOE RSS → inbox → NewsAPI/Finnhub",
        "AI research: iterative (LLM brain → Tavily/Brave/NewsAPI/GoogleNews hands)",
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
