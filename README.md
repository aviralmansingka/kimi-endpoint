# kimi-endpoint — $100M/MW-yr validation harness

Serves Kimi-K3-NVFP4 on one 8×B300 Modal replica (11.2 kW at TDP accounting)
and measures **what it earns** at Kimi K3 OpenRouter prices
($3 fresh input / $0.30 cache read / $15 output per Mtok).  The
$100M/MW-yr SemiAnalysis claim is context, not a pass/fail bar.

## The model (one screen)

    earn/hr while serving = 3600/1e6 · (P_IN·P + P_CACHE·C + P_OUT·D)  # tok/s × $/Mtok
    earnings at usage u   = u · earn/hr · 8760 / (NODE_KW/1000)        # $/MW-yr
    B                     = P_OUT + P_IN·R(1−c) + P_CACHE·R·c        # blended $/M out tok

No pass/fail target: the goal is to measure what the node **earns** at a
given utilization (`--u-report`, default 0.7).  B is just the price side —
how much input revenue rides along with each output token sold.

## Commands

One-time prep (tokenizer files already live in `tok/`):

    uv run --with tiktoken python build_corpus.py          # corpus.jsonl from your pi logs
    uv run --with tiktoken --with aiohttp python test_harness.py   # invariants

Serve warm and note the proxy URL from the deploy output:

    MIN_CONTAINERS=1 modal deploy serve.py

Run the grid (full default grid ≈ 4 h):

    uv run --with aiohttp --with tiktoken python bench_grid.py --url <URL>
    # smoke cell: ... --ks 32 --cshares 0.0,0.95 --duration-floor 120

Optional — B measured straight from the raw chat logs:

    uv run --with tiktoken python analyze_chatdist.py

## Reading the output: what we earn

Each cell line prints the measured operating point (`D P C`, measured
`c`/`R`, `B`, TTFT/ITL p95) and the earnings:
`earn $X.XX/hr serving -> $Y.YM/MW-yr @u0.7`.

    jq -r '.cells[] | select(.ok)
           | [.k, .c_target, (.usd_per_hr*100|round/100),
              (.per_mw_yr_u/1e6*10|round/10)] | @tsv' results.json
    # columns: K, c_target, $/hr while serving, $M/MW-yr at --u-report

- Earnings scale linearly with u: `$/MW-yr(u) = u · per_mw_yr_u1`.
- `usd_per_hr` is the rate while the node is serving; `per_mw_yr_u` applies
  the usage factor (demand gaps, restarts, SLO headroom).
- The closing `frontier` table is the summary view: max SLO-feasible `D`
  per c column, with its earnings.
- `SLO fail` means throughput at that concurrency isn't sellable at those
  latencies, whatever the token rate.

## Cheap verification (no B300 time)

Verify the whole harness on one A100-80GB ($2.50/hr) with
Qwen3-30B-A3B-Instruct-2507-FP8 at its OpenRouter prices — full smoke for
~$0.85 (~$1.15 first run, incl. the one-time 31 GB weight pull):

    MIN_CONTAINERS=1 modal deploy serve_cheap.py   # note the URL
    ./run_qwen_smoke.sh <URL>                       # TOK/P_*/NODE_KW envs
    modal app stop qwen3-smoke                      # teardown, stops billing

The envs swap tokenizer, prices, and node watts, so B and the earnings come
out in Qwen terms; machinery under test is identical.

## Caveats

- Keep `--ks` ≤ 64 unless `cuda-graph-max-bs` is raised in `serve.py`.
- Cache hits are counted from what we re-send (exact by construction);
  a server-side cross-check needs SGLang's cache report enabled.
- Your own pi usage keeps growing the session logs — rebuild
  `corpus.jsonl` occasionally if you want the mix to track your habits.
