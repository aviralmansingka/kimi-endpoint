#!/usr/bin/env bash
# Cheap harness verification: Qwen3-30B-A3B-Instruct-2507-FP8 on one
# A100-80GB ($2.50/hr on Modal).  Full smoke sweep (c=0.0 + c=0.95 at R=50,
# K=32) for ~$0.85 of GPU time (first run ~$1.15 incl. the one-time weight
# pull).  Prices: OpenRouter/StreamLake $0.048/$0.193; cache read assumed
# 10% of input (Qwen convention).  NODE_KW: one A100-80 SXM at 400W TDP.
#
# Usage: MIN_CONTAINERS=1 modal deploy serve_cheap.py   # note the URL
#        ./run_qwen_smoke.sh <endpoint-url>
#        modal app stop qwen3-smoke                     # teardown
set -euo pipefail
URL=${1:?usage: run_qwen_smoke.sh <endpoint-url>}
export TOK=qwen3
export P_IN=0.048 P_CACHE=0.0048 P_OUT=0.193
export NODE_KW=${NODE_KW:-0.4}  # A100-80 SXM 400W TDP; use 0.7 on H100
uv run --with tiktoken --with tokenizers --with aiohttp \
    python bench_grid.py --url "$URL" \
    --ks 32 --cshares 0.0,0.95 --duration-floor 90 --turn-est 10
