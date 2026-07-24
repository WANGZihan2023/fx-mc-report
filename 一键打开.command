#!/bin/bash
# 双击即可：自动建虚拟环境、装依赖、打开浏览器看报告
set -e
cd "$(dirname "$0")"
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

echo "=========================================="
echo "  FX 蒙特卡洛情报报告 — 一键启动"
echo "=========================================="

pick_python() {
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      echo "$c"
      return
    fi
  done
  return 1
}

PY=$(pick_python) || {
  echo ""
  echo "未检测到 Python3。请先安装："
  echo "  https://www.python.org/downloads/"
  echo "或：brew install python"
  echo ""
  read -n 1 -s -r -p "按任意键退出…"
  exit 1
}

echo "使用解释器: $PY ($($PY --version 2>&1))"

if [ ! -d ".venv" ]; then
  echo "首次运行：创建虚拟环境…"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "检查 / 安装依赖（首次可能要 1–2 分钟）…"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

PORT=8501
# 若端口占用则换一个
if lsof -ti ":$PORT" >/dev/null 2>&1; then
  PORT=8502
fi

URL="http://127.0.0.1:$PORT"
echo ""
echo "正在启动… 浏览器将打开：$URL"
echo "关闭本窗口即停止服务。"
echo ""

# 稍等后打开浏览器
(sleep 2 && open "$URL") &

exec streamlit run app.py --server.port "$PORT" --server.headless true
