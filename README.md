# kimi-endpoint — $100M/MW-yr validation harness

Serves Kimi-K3-NVFP4 on one 8×B300 Modal replica (17 kW all-in, ~2.12 kW/GPU
— the SemiAnalysis/InferenceX power basis)
and measures **what it earns** at Kimi K3 OpenRouter prices
($3 fresh input / $0.30 cache read / $15 output per Mtok).  The
$100M/MW-yr SemiAnalysis claim is context, not a pass/fail bar.

## Measured results (8×B300)

Three verified runs: run3 baseline (900 s window, no HiCache), run4 HiCache
(`artifacts/k3-agentx-run4/run4-report.md`, 2026-09-26, 3,600 s window,
`hicache_size=64` per rank, no speculative decoding), and run5 HiCache+DSPARK
(`artifacts/k3-agentx-run5/attempt7-analysis.md`, 2026-09-28, 900 s window,
combined via pinned nightly `dev-cu13-20260928-81f27fb3`). Pricing
$3/$0.30/$15 per Mtok, node power 17 kW all-in (2.12 kW/GPU, matching the
SemiAnalysis basis — host, NICs, cooling share), u=0.7; gross serving revenue,
not profit. Profit rows assume 8 GPUs × $4/GPU-hour billed 8,760 h/yr
(always-on) against the u=0.7 revenue.

| Metric | Run3 baseline (900 s) | Run4 HiCache (3,600 s) | Run5 HiCache+DSPARK (900 s) |
|---|---:|---:|---:|
| Earnings/hour | $110.91 | $110.41 | $80.94 |
| $/MW-year at u=0.7 (17 kW all-in) | $40.0M | $39.8M | $29.2M |
| Profit/hour @ $4/GPU-hour | $45.63 | $45.28 | $24.65 |
| Profit $/MW-year @u0.7, $4/GPU-hour | $23.5M | $23.3M | $12.7M |
| TTFT p50 / p95 | 2.5 / 40.8 s | 1.4 / 5.1 s | 1.4 / 3.5 s |
| ITL p90 | 91.6 ms | 43.6 ms | 48.4 ms |
| Interactivity (1000/p90 ITL) | 10.9 tok/s/user | 22.9 tok/s/user | 20.6 tok/s/user |
| Cache share c | 0.669 | 0.933 | 0.942 |
| Output throughput D | 174.5 tok/s | 411.3 tok/s | 330.1 tok/s |
| DSPARK accept rate / length | — | — | 15.6% / 2.09 |
| Errors | 0.0% | 0.0% | 0.3% |

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
kW to MW. `NODE_KW` is assumed node power: default `17.0` (8 × B300 all-in,
2.12 kW/GPU — the SemiAnalysis/InferenceX basis, including host, NICs, and
cooling share),
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

## Caveats

- Keep `--ks` ≤ 64 unless `cuda-graph-max-bs` is raised in `serve.py`.
- Cache hits are counted from what we re-send (exact by construction);
  a server-side cross-check needs SGLang's cache report enabled.
- Your own pi usage keeps growing the session logs — rebuild
  `corpus.jsonl` occasionally if you want the mix to track your habits.

## Assumptions

Every $/MW-year figure inherits all of these; the prices and the usage
factor do most of the work.

- **Prices are inputs, not receipts.** $3 fresh / $0.30 cached / $15 output
  per Mtok (Kimi K3 OpenRouter); env-overridable. Revenue ≈ B × D, so the
  answer moves linearly with the output price.
- **Usage factor u = 0.7.** Demand gaps, restarts, and SLO headroom; a pure
  linear multiplier on the annual figure.
- **Node power 17 kW all-in** (2.12 kW/GPU, the SemiAnalysis/InferenceX
  basis: GPU TDP + host CPU, NICs, cooling share). Excludes facility PUE;
  not wall-metered. Earlier reports used 14.5 kW TDP-only (gross figures
  ~15% higher on that basis).
- **Gross revenue, not profit.** No operating or GPU cost is subtracted;
  compute cost appears only in the run cost ledgers ($56.80/node-hour,
  a Modal estimate, not an invoice).
- **Achieved throughput is sellable.** Latency is reported but no pass/fail
  bar removes unsellable operating points.
- **Scale replication.** A MW-year assumes the measured operating point —
  and the demand to fill it — replicates for 8,760 hours at constant prices.
- **Token accounting.** Server-reported counts; missing cache data bills all
  input as fresh (H=0), which overstated one walkthrough 4.2×.
- **Synthetic token sizes.** chars/4 approximations; measured R drifted from
  the 50 target to 37, and natural text shifts mix and caching.
