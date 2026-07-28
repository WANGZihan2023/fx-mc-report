#!/usr/bin/env bash
# Push non-empty API keys from local .env into Railway Variables.
# Never prints secret values — only variable names and counts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAULT_ENV="${FX_API_ENV_PATH:-/Users/wangzihan/Desktop/工作_汇率/fx_data_apis/.env}"
PROJECT_ENV="${ROOT}/.env"
INPUT_ENV="${1:-}"

if ! command -v railway >/dev/null 2>&1; then
  echo "error: railway CLI not found. Install: https://docs.railway.com/guides/cli" >&2
  exit 1
fi

pick_env=""
if [[ -n "${INPUT_ENV}" ]]; then
  if [[ ! -f "${INPUT_ENV}" ]]; then
    echo "error: env file not found: ${INPUT_ENV}" >&2
    exit 2
  fi
  pick_env="${INPUT_ENV}"
else
  for candidate in "${PROJECT_ENV}" "${VAULT_ENV}"; do
    if [[ -f "${candidate}" ]]; then
      # Count non-empty KEY=value lines (exclude FX_API_* plumbing).
      n="$(
        awk -F= '
          /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
          {
            key=$1
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
            if (key ~ /^FX_API_/) next
            val=substr($0, index($0,"=")+1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
            gsub(/^["'\'']|["'\'']$/, "", val)
            if (length(val)>0) c++
          }
          END { print c+0 }
        ' "${candidate}"
      )"
      if [[ "${n}" -gt 0 ]]; then
        pick_env="${candidate}"
        echo "using env file: ${candidate} (${n} non-empty secret-ish keys)"
        break
      fi
    fi
  done
fi

if [[ -n "${pick_env}" && "${pick_env}" == "${INPUT_ENV}" ]]; then
  n="$(
    awk -F= '
      /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
      {
        key=$1
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
        if (key ~ /^FX_API_/) next
        val=substr($0, index($0,"=")+1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
        gsub(/^["'\'']|["'\'']$/, "", val)
        if (length(val)>0) c++
      }
      END { print c+0 }
    ' "${pick_env}"
  )"
  echo "using env file: ${pick_env} (${n} non-empty secret-ish keys)"
fi

if [[ -z "${pick_env}" ]]; then
  echo "error: no local .env with non-empty API keys." >&2
  echo "  checked: ${PROJECT_ENV}" >&2
  echo "  checked: ${VAULT_ENV}" >&2
  echo "  usage: ./scripts/push_env_to_railway.sh [path/to/file.env]" >&2
  echo "Fill keys locally (Streamlit「保存到本机 .env」 or edit vault), then re-run." >&2
  exit 2
fi

# Keys we push (must match Railway API / LLM persistence whitelist).
ALLOWED='^(FRED_API_KEY|NEWSAPI_KEY|FINNHUB_API_KEY|ALPHA_VANTAGE_API_KEY|TWELVE_DATA_API_KEY|TAVILY_API_KEY|BRAVE_SEARCH_API_KEY|GROQ_API_KEY|DEEPSEEK_API_KEY|LLM_API_KEY|OPENAI_API_KEY|LLM_BASE_URL|LLM_MODEL|FMP_API_KEY|POLYGON_API_KEY|OPENEXCHANGERATES_APP_ID|BROKER_REST_BASE_URL|BROKER_REST_TOKEN)$'

pushed=0
skipped=0

while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
  if [[ "${line}" =~ ^[[:space:]]*export[[:space:]]+ ]]; then
    line="${line#*export }"
  fi
  [[ "${line}" != *"="* ]] && continue
  key="${line%%=*}"
  key="$(echo "${key}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  val="${line#*=}"
  val="$(echo "${val}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^["'\'']//;s/["'\'']$//')"
  [[ -z "${key}" || -z "${val}" ]] && continue
  if ! [[ "${key}" =~ ${ALLOWED} ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  # railway variables set KEY=VALUE — do not echo VALUE
  if railway variables set "${key}=${val}" >/dev/null; then
    echo "set: ${key}"
    pushed=$((pushed + 1))
  else
    echo "failed: ${key}" >&2
    exit 3
  fi
done < "${pick_env}"

echo "done: pushed ${pushed} variable(s), skipped ${skipped} other line(s)."
echo "Redeploy / restart the Railway service if it does not pick up Variables automatically."
