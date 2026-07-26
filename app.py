"""
Streamlit 入口（Cloud / 本机均用此路径）。

业务 UI 在 fx_report.ui.streamlit_app；此处仅作稳定入口，勿把逻辑写回本文件。
"""

from fx_report.ui import streamlit_app as _app  # noqa: F401
