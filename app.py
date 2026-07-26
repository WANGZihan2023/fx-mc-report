"""
Streamlit 入口（Cloud / 本机 / Railway 均用此路径）。

业务 UI 在 fx_report.ui.streamlit_app；此处必须调用 main()，
仅 import 不会渲染页面（会呈现空白壳）。
"""

from fx_report.ui.streamlit_app import main

main()
