# Experiments

## Stage 1 — the cheap, easy answer (this branch)

One question: **what does one H100 serving Qwen3-30B at Qwen OpenRouter
prices earn at u=0.7, at the 50:1 / 95%-cache working mix?**  Cost ≈ $1,
~15 min wall (the weights volume is already warm from the 2026-09-25 run).

```bash
# AIPerf client (needs Python 3.11–3.13)
python3.13 -m venv .venv-aiperf && source .venv-aiperf/bin/activate
pip install 'aiperf==0.13.0' httpx

# Server (~3–5 min boot from the warm volume)
GPU=H100 MIN_CONTAINERS=1 modal deploy serve_cheap.py      # note the URL

# Load: shared-prefix mix ~ R=50, c=0.95 (19k cached prefix + 1k fresh tail,
# ~400-token outputs), 32 concurrent, bounded to 10 minutes
python bench_agentic.py --base-url "$URL" \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --pattern shared-prefix --prefix-tokens 19000 --tail-tokens 1000 \
    --output-tokens 400 --concurrency 32 --duration 600 --num-requests 0 \
    --cache-report --artifact-dir artifacts/bench-h100-shared

# Earnings (Qwen prices; one H100 SXM at 700W TDP)
P_IN=0.048 P_CACHE=0.0048 P_OUT=0.193 NODE_KW=0.7 \
    python earnings.py artifacts/bench-h100-shared

modal app stop qwen3-smoke --yes      # teardown, stops billing
```

The answer line: `earn $X/hr serving -> $Y M/MW-yr @u0.7`.

**Cross-check built in:** the bench_grid smoke measured the same server and
mix at $1.02/hr → $8.9M/MW-yr (2026-09-25, results.json).  If AIPerf +
earnings.py lands within ~10–20% of that, both independent stacks are
validated and you can trust whichever you find easier to reason about.

**Everything you need to understand the number:** token rates × prices =
$/hr; × u × 8760 / node-MW = $/MW-yr.  The mix is three knobs
(prefix / tail / output tokens).  Token counts come from the server's own
usage reporting (`--use-server-token-count`), so Kimi/Qwen reasoning tokens
are billed correctly without any client-side tokenizer.

## Stage 2 — the realistic version (main branch)

8×B300 + Kimi-K3 at K3 prices: the `bench_grid.py` concurrency × fresh-share
grid with corpus-replayed natural text (see README on main).  Same earnings
math; adds real tokenizer counting, DSPARK speculative acceptance on
natural text, and multi-turn session shapes.

## Caveats carried from the AIPerf line

- Synthetic padding text: speculative-decode acceptance and token
  compression differ from natural text — budget ±10–20% on the physics.
- `--cache-report` is required for C; without it the export has no
  cache-read tokens and earnings.py assumes C=0 (and says so).
- Tool-loop and fan-out patterns need the adapters in bench_agentic.py;
  shared-prefix and multi-turn-growth are native AIPerf.
