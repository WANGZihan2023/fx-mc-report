#!/bin/bash
# 临时公网链接：对方只需浏览器（你本机须已在跑或由本脚本启动 Streamlit）
set -e
cd "$(dirname "$0")"

PORT=8501
if ! lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "8501 未在运行，先启动 Streamlit…"
  if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
  streamlit run app.py --server.port "$PORT" --server.headless true &
  sleep 4
fi

echo ""
echo "正在创建临时公网隧道（对方只需浏览器）…"
echo "关闭本窗口后，链接失效。"
echo ""

# localtunnel via npx — no global install required if node exists
if command -v npx >/dev/null 2>&1; then
  npx --yes localtunnel --port "$PORT"
elif command -v cloudflared >/dev/null 2>&1; then
  cloudflared tunnel --url "http://127.0.0.1:$PORT"
else
  echo "需要 Node.js（npx）或 cloudflared 之一来创建隧道。"
  echo "安装 Node: https://nodejs.org/  然后重新运行本脚本。"
  echo ""
  echo "或安装 cloudflared:"
  echo "  brew install cloudflared"
  exit 1
fi
