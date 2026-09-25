# Experiments

## Stage 1 — the cheap, easy answer (this branch)

One question: **what does one H100 serving Qwen3-30B at Qwen OpenRouter
prices earn at u=0.7, at the 50:1 / 95%-cache working mix?**  Cost ≈ $1,
~15 min wall (the weights volume is already warm from the 2026-09-25 run).

```bash
# 0) Client sanity (free). Pin Python 3.12: uv otherwise picks 3.14, where
#    pyzmq has no wheels and its source build fails on libzmq.
uv run --python 3.12 --with 'aiperf==0.13.0' --with httpx aiperf --version

# 1) Free dry-run — prints the dataset row and the exact aiperf command, no network.
uv run --python 3.12 --with 'aiperf==0.13.0' --with httpx python bench_agentic.py \
    --base-url http://dryrun.invalid --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --pattern shared-prefix --prefix-tokens 19000 --tail-tokens 1000 \
    --output-tokens 400 --concurrency 32 --duration 600 --num-requests 0 \
    --cache-report --artifact-dir artifacts/bench-dryrun --dry-run

# 2) Server (~3–5 min from the warm weights volume)
GPU=H100 MIN_CONTAINERS=1 modal deploy serve_cheap.py     # note the URL
URL=<that-url>
until curl -sf --max-time 10 "$URL/health" >/dev/null; do sleep 20; done && echo ready

# 3) The run (~10 min, bounded by --duration; use a FRESH artifact dir each time;
#    --rows 1000 so the sequential dataset doesn't fall short of the window —
#    AIPerf wraps it anyway, but only after exhausting the rows)
uv run --python 3.12 --with 'aiperf==0.13.0' --with httpx python bench_agentic.py \
    --base-url "$URL" --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --pattern shared-prefix --prefix-tokens 19000 --tail-tokens 1000 \
    --output-tokens 400 --concurrency 32 --duration 600 --num-requests 0 \
    --rows 1000 --cache-report --artifact-dir artifacts/bench-h100-shared-v2
tail -f artifacts/bench-h100-shared/runner.log   # (2nd terminal) live progress

# 4) The answer
P_IN=0.048 P_CACHE=0.0048 P_OUT=0.193 NODE_KW=0.7 \
    uv run --python 3.12 --with tiktoken python earnings.py artifacts/bench-h100-shared

# 5) Teardown — stops billing
modal app stop qwen3-smoke --yes
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

- The server must pass `--enable-cache-report` (serve_cheap.py does since
  2026-09-26) or the export contains no cache-read tokens and earnings.py
  assumes C=0 — which **overstates earnings ~4x** at this mix (billing cache
  reads as fresh input).  The 2026-09-25 walkthrough run hit exactly this:
  $10.83/hr as-measured vs ~$2.6/hr cache-corrected.
- Dataset token sizes are chars/4 approximations — measured R drifts from
  the knob (R=37 when the knobs aimed at 50).  The server's usage is truth.
- Synthetic padding text: speculative-decode acceptance and token
  compression differ from natural text — budget ±10–20% on the physics.
- Tool-loop and fan-out patterns need the adapters in bench_agentic.py;
  shared-prefix and multi-turn-growth are native AIPerf.
