@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
title FX Monte Carlo Report

echo ==========================================
echo   FX 蒙特卡洛情报报告 — 一键启动
echo ==========================================

where py >nul 2>&1 && set PY=py -3
if not defined PY where python >nul 2>&1 && set PY=python
if not defined PY (
  echo.
  echo 未检测到 Python。请先安装：
  echo   https://www.python.org/downloads/
  echo 安装时勾选 "Add python.exe to PATH"
  echo.
  pause
  exit /b 1
)

echo 使用: %PY%
%PY% --version

if not exist ".venv" (
  echo 首次运行：创建虚拟环境…
  %PY% -m venv .venv
)

call .venv\Scripts\activate.bat

echo 检查 / 安装依赖（首次可能要 1–2 分钟）…
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

set PORT=8501
echo.
echo 正在启动… 浏览器打开 http://127.0.0.1:%PORT%
echo 关闭本窗口即停止服务。
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:%PORT%"
streamlit run app.py --server.port %PORT% --server.headless true

pause
