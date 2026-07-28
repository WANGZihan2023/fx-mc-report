"""API key save/load: no blank wipe, dual-path write, cloud detection."""

from __future__ import annotations

from pathlib import Path

import fx_report.config.api_config as ac


def test_merge_nonempty_never_wipes_with_blank():
    existing = {"FRED_API_KEY": "real-key", "NEWSAPI_KEY": "news"}
    merged = ac.merge_nonempty(
        existing,
        {"FRED_API_KEY": "", "NEWSAPI_KEY": "  ", "TAVILY_API_KEY": "tavily"},
    )
    assert merged["FRED_API_KEY"] == "real-key"
    assert merged["NEWSAPI_KEY"] == "news"
    assert merged["TAVILY_API_KEY"] == "tavily"


def test_save_keys_to_local_dual_write_and_no_blank_overwrite(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    vault_env = vault / ".env"
    vault_env.write_text("FRED_API_KEY=keep-me\nNEWSAPI_KEY=old-news\n", encoding="utf-8")
    project_env = tmp_path / "project.env"

    monkeypatch.setattr(ac, "env_path", lambda: vault_env)
    monkeypatch.setattr(ac, "project_env_path", lambda: project_env)
    monkeypatch.setattr(ac, "default_vault_root", lambda: vault)
    monkeypatch.setattr(ac, "is_cloud_runtime", lambda: False)

    paths = ac.save_keys_to_local(
        {
            "FRED_API_KEY": "",  # must not wipe
            "NEWSAPI_KEY": "new-news",
            "GROQ_API_KEY": "groq-1",
        }
    )
    assert vault_env.resolve() in paths
    assert project_env.resolve() in paths

    for p in (vault_env, project_env):
        raw = ac._parse_env_file(p)
        assert raw["FRED_API_KEY"] == "keep-me"
        assert raw["NEWSAPI_KEY"] == "new-news"
        assert raw["GROQ_API_KEY"] == "groq-1"
        assert all((v or "").strip() for k, v in raw.items() if k.endswith("_KEY") or k.endswith("_TOKEN"))

    verified = ac.verify_env_file(vault_env)
    assert verified["FRED_API_KEY"] != "(empty)"
    assert verified["NEWSAPI_KEY"] != "(empty)"


def test_is_cloud_runtime_railway(monkeypatch):
    monkeypatch.delenv("FX_FORCE_LOCAL_RUNTIME", raising=False)
    monkeypatch.delenv("FX_FORCE_CLOUD_RUNTIME", raising=False)
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    assert ac.is_cloud_runtime() is True
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.setenv("FX_FORCE_LOCAL_RUNTIME", "1")
    assert ac.is_cloud_runtime() is False


def test_env_download_bytes_skips_empty():
    body = ac.env_file_download_bytes(
        {"FRED_API_KEY": "abc", "NEWSAPI_KEY": "", "LLM_MODEL": "llama"}
    ).decode()
    assert "FRED_API_KEY=abc" in body
    assert "NEWSAPI_KEY=" not in body
    assert "LLM_MODEL=llama" in body


def test_write_env_omits_blank_keys(tmp_path):
    path = tmp_path / ".env"
    ac._write_env_file(
        path,
        {
            "FX_API_ROOT": str(tmp_path),
            "FRED_API_KEY": "x",
            "NEWSAPI_KEY": "",
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "FRED_API_KEY=x" in text
    assert "NEWSAPI_KEY=" not in text
