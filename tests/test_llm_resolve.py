"""resolve_llm_config: DeepSeek / Groq base URL inference."""

from __future__ import annotations

from fx_report.news.llm import resolve_llm_config


def test_deepseek_env_key_sets_base(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    cfg = resolve_llm_config(allow_ollama_auto=False)
    assert cfg is not None
    assert "deepseek.com" in cfg.base_url
    assert cfg.model == "deepseek-chat"
    assert cfg.api_key == "sk-deepseek-test"


def test_deepseek_key_rewrites_mistaken_openai_base(monkeypatch):
    """UI used to default Base URL to api.openai.com while user pasted DeepSeek key."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = resolve_llm_config(
        api_key="sk-deepseek-test",
        base_url="https://api.openai.com/v1",
        allow_ollama_auto=False,
    )
    assert cfg is not None
    assert "deepseek.com" in cfg.base_url
    assert cfg.model == "deepseek-chat"


def test_deepseek_base_without_v1_is_normalized(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = resolve_llm_config(
        api_key="sk-deepseek-test",
        base_url="https://api.deepseek.com",
        allow_ollama_auto=False,
    )
    assert cfg is not None
    assert cfg.base_url == "https://api.deepseek.com/v1"
    assert cfg.model == "deepseek-chat"


def test_explicit_openai_key_keeps_openai_base(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-other")
    cfg = resolve_llm_config(
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        allow_ollama_auto=False,
    )
    assert cfg is not None
    assert "openai.com" in cfg.base_url


def test_thin_refs_message_for_single_evidence():
    from fx_report.model.label_audit import thin_refs_message

    msg = thin_refs_message(evidence_n=1, fetched=8, news_keys_present=False)
    assert msg is not None
    assert "NEWSAPI" in msg or "Finnhub" in msg
    assert "DeepSeek" in msg or "LLM" in msg
    assert thin_refs_message(evidence_n=5, fetched=20, news_keys_present=True) is None
