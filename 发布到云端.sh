#!/bin/bash
# 推到 GitHub，便于 Streamlit Cloud 部署成长期网址
set -e
cd "$(dirname "$0")"

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub 未登录或 token 失效，请先执行："
  echo "  gh auth refresh -h github.com"
  exit 1
fi

if [ ! -d .git ]; then
  git init
fi

git add -A
if git diff --cached --quiet; then
  echo "没有新的变更需要提交。"
else
  git commit -m "$(cat <<'EOF'
Add multi-pair FX Monte Carlo Streamlit report app.

EOF
)"
fi

NAME="${1:-fx-mc-report}"
if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$NAME" --public --source=. --remote=origin --push
else
  git push -u origin HEAD
fi

URL=$(gh repo view --json url -q .url)
echo ""
echo "仓库已就绪：$URL"
echo ""
echo "下一步（一次性）："
echo "  1. 打开 https://share.streamlit.io/"
echo "  2. New app → 选这个仓库 → Main file: app.py → Deploy"
echo "  3. 把生成的 https://xxxx.streamlit.app 发给对方"
echo ""
