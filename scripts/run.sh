#!/usr/bin/env bash

# 运行指南：
# 1) （可选）复制 .env.example 为 .env 并按需修改
# 2) 运行：
#    bash scripts/run.sh
# 3) 输出位于 outputs/（prediction.zip / A1.csv / A2.csv / experiments.jsonl）

set -eu

if [ -f ".env" ]; then
  set -a
  . "./.env"
  set +a
fi

if [ "${1-}" != "" ]; then
  export AFAC_CONFIG_PATH="$1"
fi

rm -f outputs/agent_outputs.json
python -m afac_agent.run

