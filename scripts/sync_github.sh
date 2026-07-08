#!/usr/bin/env sh

# 运行指南：
# 1) 确保本地仓库已配置远程（默认 remote 为 origin）
# 2) 在仓库根目录执行：
#    sh scripts/sync_github.sh
# 3) 可选传参：
#    sh scripts/sync_github.sh <remote> <branch>
#
# 说明：
# - 不需要手动输入 commit message
# - 默认自动检测当前分支；如检测失败，则回退到 main
# - 仅执行 git add/commit/push，不会做 reset、rebase 等破坏性操作

set -eu

REMOTE="${1-origin}"
BRANCH="${2-}"

if [ -z "$BRANCH" ]; then
  BRANCH="$(git branch --show-current 2>/dev/null || true)"
fi

if [ -z "$BRANCH" ]; then
  BRANCH="main"
fi

STATUS_SUMMARY="$(git status --short 2>/dev/null || true)"

if [ -z "$STATUS_SUMMARY" ]; then
  echo "No changes to commit."
  exit 0
fi

CHANGE_COUNT="$(printf '%s\n' "$STATUS_SUMMARY" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
COMMIT_MESSAGE="chore: sync repo updates (${CHANGE_COUNT} files, ${TIMESTAMP})"

git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit."
  exit 0
fi

git commit -m "$COMMIT_MESSAGE"
git push "$REMOTE" "$BRANCH"

echo "Pushed to ${REMOTE}/${BRANCH}"
echo "Commit message: ${COMMIT_MESSAGE}"

