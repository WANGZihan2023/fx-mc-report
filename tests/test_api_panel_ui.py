"""API panel UI regressions around editable secret inputs."""

from __future__ import annotations

from textwrap import dedent

from streamlit.testing.v1 import AppTest


APP_SCRIPT = dedent(
    """
    from fx_report.ui.api_panel import render_api_settings_panel

    render_api_settings_panel()
    """
)


def _build_app() -> AppTest:
    return AppTest.from_string(APP_SCRIPT)


def test_secret_inputs_stay_editable_when_env_values_exist(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fred-from-env")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-from-env")
    monkeypatch.setenv("LLM_API_KEY", "llm-from-env")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")

    at = _build_app()
    at.run()

    # Existing secrets stay hidden from the UI.
    assert at.text_input(key="free_keys__FRED_API_KEY").value == ""
    assert at.text_input(key="ai_api_key_input").value == ""

    # User can still type replacement values over env-loaded secrets.
    at.text_input(key="free_keys__FRED_API_KEY").set_value("fred-override")
    at.text_input(key="ai_api_key_input").set_value("deepseek-override")
    at.run()

    assert at.text_input(key="free_keys__FRED_API_KEY").value == "fred-override"
    assert at.text_input(key="ai_api_key_input").value == "deepseek-override"
    assert at.session_state["api_keys_ui"]["FRED_API_KEY"] == "fred-override"
    assert at.session_state["api_keys_ui"]["LLM_API_KEY"] == "deepseek-override"
    assert at.session_state["api_keys_ui"]["DEEPSEEK_API_KEY"] == "deepseek-override"


def test_session_save_keeps_values_but_clears_secret_widgets(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fred-from-env")
    monkeypatch.setenv("LLM_API_KEY", "llm-from-env")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")

    at = _build_app()
    at.run()

    at.text_input(key="free_keys__FRED_API_KEY").set_value("fred-session")
    at.text_input(key="ai_api_key_input").set_value("llm-session")
    at.button[2].click()
    at.run()
    at.run()

    # Saved value remains effective, but the widgets blank back out so secrets
    # are not shown in plain text after the action completes.
    assert at.session_state["api_keys_ui"]["FRED_API_KEY"] == "fred-session"
    assert at.session_state["api_keys_ui"]["LLM_API_KEY"] == "llm-session"
    assert at.text_input(key="free_keys__FRED_API_KEY").value == ""
    assert at.text_input(key="ai_api_key_input").value == ""


def test_persist_requires_admin_unlock(monkeypatch):
    monkeypatch.setenv("ADMIN_SAVE_TOKEN", "save-admin-token")

    at = _build_app()
    at.run()

    assert at.button(key="btn_persist_server").disabled is True

    at.text_input(key="free_keys__FRED_API_KEY").set_value("fred-new")
    at.text_input(key="admin_save_token_input").set_value("save-admin-token")
    at.button(key="btn_unlock_admin_save").click()
    at.run()

    assert at.session_state["admin_save_unlocked"] is True
    assert at.button(key="btn_persist_server").disabled is False
