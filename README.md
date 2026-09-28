# kimi-endpoint — $100M/MW-yr validation harness

Serves Kimi-K3-NVFP4 on one 8×B300 Modal replica (14.5 kW at TDP accounting)
and measures **what it earns** at Kimi K3 OpenRouter prices
($3 fresh input / $0.30 cache read / $15 output per Mtok).  The
$100M/MW-yr SemiAnalysis claim is context, not a pass/fail bar.

## Measured results: run4 (HiCache on 8×B300)

The most recent verified run (`artifacts/k3-agentx-run4/run4-report.md`, 2026-09-26,
3,600 s window, `hicache_size=64` per rank, no speculative decoding) against the
run3 baseline (900 s window, no HiCache). Pricing $3/$0.30/$15 per Mtok, node power
14.5 kW, u=0.7; gross serving revenue, not profit.

| Metric | Run3 baseline | Run4 HiCache |
|---|---:|---:|
| Earnings/hour | $110.91 | $110.41 |
| $/MW-year at u=0.7 (14.5 kW) | $46.9M | $46.7M |
| TTFT p50 / p95 | 2.5 / 40.8 s | 1.4 / 5.1 s |
| ITL p90 | 91.6 ms | 43.6 ms |
| Interactivity (1000/p90 ITL) | 10.9 tok/s/user | 22.9 tok/s/user |
| Cache share c | 0.669 | 0.933 |
| Output throughput D | 174.5 tok/s | 411.3 tok/s |
| Errors | 0.0% | 0.0% |

## How earnings per MW are estimated

`earnings.py` turns measured token usage into estimated gross revenue at chosen
prices, then scales it to one MW-year. It does not calculate profit or measure power.
See [BENCHMARKS.md](BENCHMARKS.md) for running tests and [EXPERIMENT.md](EXPERIMENT.md)
for the reported H100 results.

### 1. Measured tokens → tokens/second

Read `profile_export_aiperf.json`. These counters store run totals in `.avg`
(despite the name):

- `I`: `total_usage_prompt_tokens`, including cached input.
- `H`: `total_usage_prompt_cache_read_tokens`, the reported cache-read subset.
- `O`: `total_usage_completion_tokens`, including reported reasoning output.
- `T`: `benchmark_duration`, the measured duration in seconds.

```text
P = (I − H) / T     fresh input tokens/s
C = H / T           cached input tokens/s
D = O / T           output tokens/s
```

`bench_agentic.py` delegates scheduling and metrics to AIPerf and requests server
token counts. Use achieved throughput, not the offered request rate or token caps.
Concurrency selects the load; it is not a multiplier on these aggregate rates.
Native concurrency caps root sessions; DAG children can exceed that cap.
Live tool-loop exports contain raw events, not the AIPerf summary used here.

### 2. Elapsed time → GPU-hours

For `G` GPUs allocated throughout the measured interval:

```text
GPU-hours = G × T / 3600
```

`G` comes from the deployment (1 for the H100 experiment; 8 for the B300 node),
not concurrency. `earnings.py` does not calculate GPU-hours or need them for revenue;
startup, idle time and extra replicas add to billable GPU-hours.

### 3. Token rates × prices → dollars/hour

```text
$/hr serving = (P_IN × P + P_CACHE × C + P_OUT × D) × 3600 / 1,000,000
```

`P_IN`, `P_CACHE`, `P_OUT` are assumed dollars per million fresh, cached and output
tokens. They are environment settings imported from `analyze_chatdist.py`:
default Kimi prices are `$3 / $0.30 / $15`; the Qwen experiment uses
`$0.048 / $0.0048 / $0.193`. These are pricing inputs, not measured receipts.

The printed mix and blended price describe the same revenue when `O > 0`:

```text
R = I / O;  c = H / I                         input:output ratio; cached share
B = P_OUT + P_IN × R × (1 − c) + P_CACHE × R × c
$/hr serving = B × D × 3600 / 1,000,000
```

`R` and `c` come from usage, not the requested prefix/tail sizes; `B` is dollars
per million output tokens, including the input revenue that accompanies them.

### 4. Dollars/hour → dollars per MW-year

```text
$/MW-year = $/hr serving × U × 8760 / (NODE_KW / 1000)
```

`U` is an assumed usage factor (environment variable, default `0.7`), allowing for
demand gaps, restarts and latency headroom. `8760` is hours/year; `1000` converts
kW to MW. `NODE_KW` is assumed node power: default `14.5` (8 × B300 node, TDP
accounting),
or `0.7` in the H100 experiment. This is not wall-meter power or facility power;
it excludes host/cooling overhead unless you include those in `NODE_KW`.
Scaling assumes the measured operating point can be replicated across that MW.

### Two small checks from EXPERIMENT.md

The 2026-09-25 one-H100 smoke reported `$1.02/hr`. With its assumed power and usage:

```text
1.02 × 0.7 × 8760 / (0.7 / 1000) = $8,935,200/MW-year ≈ $8.9M/MW-year
```

A different walkthrough lacked cache reporting: `$10.83/hr` versus ~$2.6/hr corrected.

```text
10.83 / 2.6 ≈ 4.2× overstatement
```

The correction is an estimate, not a verified cache measurement. Missing cache
usage is unknown: `earnings.py` substitutes `H=0`, billing all input as fresh.
Enable server cache reporting and inspect the export; `--cache-report` alone
cannot guarantee that AIPerf captures the fields.

### What these numbers do not prove

- Synthetic token sizes use chars/4 approximations: the experiment measured
  `R=37` despite targeting `50`. Natural text can change throughput and caching.
- Check errors and latency before treating throughput as sellable. The script
  reports latency but does not impose a pass/fail bar or subtract operating costs.
- BENCHMARKS.md verifies local mocked runs, not Kimi-on-Modal performance.
  The H100 smoke is a separate reported measurement; annual earnings remain
  extrapolations, not a verified year of revenue or proof of the $100M/MW-year claim.

No pass/fail target: the goal is to measure what the node **earns** at a
given utilization (`--u-report`, default 0.7).

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

## Reading the output

Each cell line prints the measured operating point (`D P C`, measured
`c`/`R`, `B`, TTFT/ITL p95) and the earnings:
`earn $X.XX/hr serving -> $Y.YM/MW-yr @u0.7`.

    jq -r '.cells[] | select(.ok)
           | [.k, .c_target, (.usd_per_hr*100|round/100),
              (.per_mw_yr_u/1e6*10|round/10)] | @tsv' results.json
    # columns: K, c_target, $/hr while serving, $M/MW-yr at --u-report

- The closing `frontier` table is the summary view: max SLO-feasible `D`
  per c column, with its earnings.
- `SLO fail` means throughput at that concurrency isn't sellable at those
  latencies, whatever the token rate.

## Caveats

- Keep `--ks` ≤ 64 unless `cuda-graph-max-bs` is raised in `serve.py`.
- Cache hits are counted from what we re-send (exact by construction);
  a server-side cross-check needs SGLang's cache report enabled.
- Your own pi usage keeps growing the session logs — rebuild
  `corpus.jsonl` occasionally if you want the mix to track your habits.
