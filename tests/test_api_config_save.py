"""API key save/load: no blank wipe, dual-path write, cloud detection."""

from __future__ import annotations

import subprocess
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


def test_parse_env_bytes_skips_empty_and_comments():
    raw = b"# comment\nFRED_API_KEY=abc123\nNEWSAPI_KEY=\nexport TAVILY_API_KEY='tvly'\n"
    parsed = ac.parse_env_bytes(raw)
    assert parsed["FRED_API_KEY"] == "abc123"
    assert parsed["TAVILY_API_KEY"] == "tvly"
    assert "NEWSAPI_KEY" not in parsed


def test_railway_checklist_names_only_no_values():
    text = ac.railway_variables_checklist(
        only_set_in={"FRED_API_KEY": "secret-should-not-appear", "NEWSAPI_KEY": ""}
    )
    assert "FRED_API_KEY" in text
    assert "secret-should-not-appear" not in text
    assert "SET\tFRED_API_KEY" in text
    assert "—\tNEWSAPI_KEY" in text
    assert "push_env_to_railway.sh" in text


def test_configured_key_sources_from_environ(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "from-railway")
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    cfg = {"FRED_API_KEY": "from-railway", "NEWSAPI_KEY": ""}
    sources = ac.configured_key_sources(cfg)
    assert sources["FRED_API_KEY"] == "已从环境变量加载"
    assert "NEWSAPI_KEY" not in sources


def test_persistence_keys_only_filters_local_only_fields():
    picked = ac.persistence_keys_only(
        {
            "FRED_API_KEY": "fred",
            "FX_API_ROOT": "/tmp/local-only",
            "NEWSAPI_KEY": "news",
            "UNKNOWN_KEY": "nope",
            "LLM_MODEL": "deepseek-chat",
            "EMPTY": "",
        }
    )
    assert picked == {
        "FRED_API_KEY": "fred",
        "NEWSAPI_KEY": "news",
        "LLM_MODEL": "deepseek-chat",
    }


def test_railway_variables_env_block_contains_real_values():
    block = ac.railway_variables_env_block(
        {"NEWSAPI_KEY": "news-123", "FX_API_ROOT": "/tmp/skip", "LLM_MODEL": "llama"}
    )
    assert "NEWSAPI_KEY=news-123" in block
    assert "LLM_MODEL=llama" in block
    assert "FX_API_ROOT" not in block


def test_railway_direct_persist_hint_without_cli(monkeypatch):
    monkeypatch.setattr(ac.shutil, "which", lambda _: None)
    ok, hint = ac.railway_direct_persist_hint()
    assert ok is False
    assert "railway" in hint.lower()


def test_persist_keys_to_railway_variables_success(monkeypatch):
    calls: list[list[str]] = []

    def _fake_run(cmd, check, capture_output, text, timeout):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(ac.shutil, "which", lambda _: "/usr/local/bin/railway")
    monkeypatch.setattr(ac.subprocess, "run", _fake_run)

    result = ac.persist_keys_to_railway_variables(
        {"FRED_API_KEY": "fred", "FX_API_ROOT": "/tmp/skip", "LLM_MODEL": "llama"}
    )
    assert result.ok is True
    assert result.changed_keys == ("FRED_API_KEY", "LLM_MODEL")
    assert calls == [
        ["/usr/local/bin/railway", "variables", "set", "FRED_API_KEY=fred"],
        ["/usr/local/bin/railway", "variables", "set", "LLM_MODEL=llama"],
    ]


def test_persist_keys_to_railway_variables_failure(monkeypatch):
    def _fake_run(cmd, check, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not linked")

    monkeypatch.setattr(ac.shutil, "which", lambda _: "/usr/local/bin/railway")
    monkeypatch.setattr(ac.subprocess, "run", _fake_run)

    result = ac.persist_keys_to_railway_variables({"NEWSAPI_KEY": "news"})
    assert result.ok is False
    assert result.changed_keys == ()
    assert "not linked" in result.message
