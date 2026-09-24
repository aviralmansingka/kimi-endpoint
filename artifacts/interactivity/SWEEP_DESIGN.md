# Interactivity → supported concurrency → utilization

## Summary
Choose a **joint TTFT + decode-speed SLO**, find its largest sustainable session-tree concurrency with a closed-loop sweep, then validate that point against independent arrivals and real inter-turn gaps. Read goodput, memory/state occupancy and billed time at that operating point—not at unconstrained peak throughput.

**Status:** design ahead of measurement; the 8×B300 endpoint is not running. No capacities, GPU utilization percentages or latency results below are measured on this endpoint. Labels: **Trace** = existing 316-session profile; **Literature** = cited external results; **Illustrative** = invented arithmetic; **Estimate** = analytical hypothesis.

## Findings

### 1. Interactivity, precisely

1. **Use two gates, not one speed number.** For request *i*, with N output tokens and times in milliseconds:
   - `TTFT_i = first generated token arrival − request submission`. Includes transport, admission/queueing, prefill and initial generation. Also record intended arrival → actual submission separately.
   - `TPOT_i = (last token arrival − first token arrival)/(N−1)`, for N>1: the request's average post-first-token generation interval.
   - `decode_speed_i = 1000/TPOT_i` tokens/s/user. It excludes waiting for the first token and tool/think time. A user here is **one decoding stream**, not necessarily a whole agent tree.
   - ITL is the distribution of successive token gaps, not a distribution of request-average TPOT. Request TPOT can pass while a stream freezes periodically; report p99 gaps/stalls as a second diagnostic.
   - DSPARK emits multiple accepted tokens together; SSE may combine or buffer them. Inter-**chunk** latency is observable, but true token-level ITL may not be. Do not assign a chunk's entire delay to each token. Report chunk gaps and request TPOT with explicit counting/timing conventions.
   - Count reasoning + answer + tool-call output exactly once using server usage. Also measure time to first **visible answer/tool call**: fast reasoning tokens do not guarantee fast user-visible progress. Native text history can omit Kimi `reasoning_content` (see `BENCHMARKS.md`).

   SemiAnalysis defines interactivity as post-first-token tok/s/user; its iso-interactivity comparison holds this experience constant across throughput/cost frontiers. [S1]

2. **Percentiles require a population.** Report request-weighted p50/p95/p99 TTFT and TPOT, pooled chunk/ITL p95/p99 separately, and session-completion latency. Split cold first turns, warmed continuations, long-context and fan-out cohorts; aggregate only at the trace's actual mix.
   - Adopt **p99 TTFT and p99 request TPOT** as capacity gates if that is the product promise. Plot p95 too for a less noisy tail view and MLPerf Endpoints comparison; p95 allows 5% violations, p99 only 1%. Neither percentile is inherently the “right” SLO.
   - `1000 / p99(TPOT)` corresponds approximately to **p1 speed**, not p99 speed or mean speed. Never invert a mean TPOT and call it the mean of per-request speeds.
   - Separate p99 gates imply at least ~98% joint compliance by the union bound, not necessarily 99%. Report the actual fraction meeting **both**, and explicitly require ≥99% joint compliance if desired.
   - Hundreds of requests are weak evidence for p99. Extend low-C runs and pool repeated seeds; aim for thousands of requests and bootstrap uncertainty by session, not independently by correlated turns.
   - Your collected v5.1 references (2000/200ms Server, 450/40ms interactive, 2000/80ms R1) are **model/scenario-specific reference targets**, not a universal Kimi standard. For example, MLCommons gives Llama3.1-8B 2000/100ms and 500/30ms. Endpoints instead presents p95 TTFT curves, 7–32 points, ≥600s/point. This design is not an MLPerf submission. [S3, S4]

3. **Closed loop answers the bounded-concurrency question.** Hold C active session trees; each dependent turn waits for its predecessor and prescribed gap; replace finished roots. Sweep C to answer “how many such trees can I sustain at interactivity X?” Offered request rate becomes an outcome. Closed loop can queue inside the server, but completion feedback prevents an unbounded independent arrival queue and masks overload during slowdowns.
   **Open loop answers admission capacity:** launch roots at independent Poisson rate λ, or replay timestamped arrivals; retain dependency barriers within each tree. Measure offered versus achieved rate, scheduler lateness and backlog growth. A concurrency cap that blocks dispatch can turn a nominal Poisson test into a throttled closed-loop test. Latency from actual dispatch alone misses that waiting.
   Report both `C_supported(SLO, workload, gaps)` and `λ_supported(SLO)`; they are not interchangeable. In a stable system, Little's law gives `mean active trees = root arrivals/s × mean tree lifetime`; HTTP concurrency uses HTTP throughput × HTTP latency instead. [S5, S6]

### 2. The sweep design

**Step 0 — Freeze the system and boundary.** One 8×B300 replica, NVFP4, TP8/DCP8, DSPARK block 7 + ReplaySSM; mamba cache 160, graph max batch 32, chunked prefill 16384, static memory fraction .85. Record image digest, model/speculator revisions, generation settings, client location and versions. Freeze temperature/reasoning effort and timeout too.
Modal `target_concurrency=32` is an autoscaling target, not proof of a 32-session engine ceiling. Ensure a one-replica benchmark using supported deployment controls and verify replica IDs/counts; otherwise 64 clients might measure two nodes. If keeping autoscaling, label it an endpoint/fleet sweep and charge for every replica. Do not change engine knobs during the baseline sweep. [Local: `serve.py`; S7]

**Step 1 — Freeze two datasets, not just token averages.**

| Trace-calibrated input | Baseline/production coverage |
|---|---|
| Agentic turns | p50 24; p90 114; p99 338; keep 28.2% single-response sessions separately |
| Initial input | ~10.7k overall p50; independent session-scoped prefixes |
| New input/turn | estimated p50 356, mean 1266, p90 2983, p99 13436; 82.5% growth from tool results |
| Output/turn | p50 243, mean 537, p90 1201, p99 4894; reasoning 44.5% of output |
| Context/cache | re-sent context p50 81k, p90 215k; per-response cache fraction p50 98.9%, mean 91.2%; token-weighted 95.1% |
| Real activity | distinct sessions/active-hour p50 2 is **not simultaneous concurrency**; idle-split overlap p90 3, p99 5, max 7 |

A: use 24 turns / 10,700 prefix / 356 tail as an interpretable smoke baseline. B: bootstrap whole session trajectories from `artifacts/trace-profile/sessions.jsonl`, preserving correlated lengths, tool bursts and parent/child relationships where available; recover missing gaps/topology from original traces. Include warmed 81k/215k contexts and rare long sessions within validated context limits. A 24-turn median trajectory reaches only ~31k tokens, so it is not B's memory-pressure substitute.

The existing generator makes constant lengths with chars/4 padding and **identical system prefixes across sessions**; `--seed` does not sample sizes. Prepare a separate native `--dataset` with unique markers at the **beginning** of each unrelated session prefix (retained on every turn), realistic common schema only, and heterogeneous limits. This preparation is required; the report does not implement a trace converter. Check actual Kimi usage lengths. `--output-tokens 537` is a cap, not a 537-token mean: start at 1280 for smoke or ≥5120 for tail coverage, then verify realized outputs. Natural generation cannot guarantee the original distribution.

**Step 2 — Prime trajectories, not “Hello.”** Exclude startup, kernel/JIT/graph warmup and seed primers from measured steady state. Run representative batches and prefills, then start replay lanes at sampled mid-session positions with their exact prefixes/history primed on the same replica; verify cache reads. Keep warmup seed/state comparable between C points, and namespace unrelated replays to avoid residual accidental hits. Priming the entire dataset would create an unrealistic universally warm workload. Measure natural new-session cold first turns separately and include their real share in the production-mix result. `serve.py`'s three short warmups do not accomplish this. AgentX uses seeded trajectory primers and warmup requests as a methodological cross-check. [S2]

**Step 3 — Sweep `C = 2,4,8,16,32,64` roots**, with C=1 for single-stream calibration and 3/6 for trace-realistic bursts. Refine around the first failure (e.g. 20/24/28/40/48). One root may launch 2–3 simultaneous children: record root-tree concurrency, active HTTP requests, running decode sequences and queued requests independently. Native `--concurrency` limits roots; `--agents` is the total root executions, not their concurrency; live tool-loop concurrency instead caps HTTP calls. Tool rounds make two requests; fan-out children each make their own turns.

Run **900–1800 measured seconds per point**, after warmup; repeat ≥3 seeds near the knee. Use streaming and server token counts, no default 100-request cutoff, sufficient roots to replenish lanes, and fresh artifact directories. Extend for p99 sample size. This command is a smoke/concurrency template; production uses the prepared trace dataset:

```bash
# Run only after readiness and explicit single-replica verification.
# BIG_N must exceed the number of roots that can complete within 1800s.
# Repeated dataset executions need per-execution prefix busting, not just new IDs.
C=8; BIG_N=10000
python bench_agentic.py --base-url "$MODAL_URL" --pattern multi-turn-growth \
  --dataset artifacts/interactivity/trace-growth.jsonl \
  --agents "$BIG_N" --concurrency "$C" --profile concurrent \
  --turns 24 --prefix-tokens 10700 --tail-tokens 356 --output-tokens 5120 \
  --num-requests 0 --duration 1800 --streaming --cache-report \
  --seed 42 --artifact-dir "artifacts/interactivity/run-c${C}-s42"
```

With `--dataset`, authored payloads replace the synthetic length flags above; those flags document the smoke shape only. `--agents C` would end after C sessions even with `--duration 1800`. Check measured duration and realized concurrency. The wrapper has no warmup-window or gap CLI: encode supported per-turn delays in native data/use direct AIPerf, and explicitly separate warmup records. Duration bounds admission; record drain time, incomplete/aborted turns and a consistent throughput window rather than silently dropping slow completions. [S5; local `bench_agentic.py`]

**Step 4 — Collect at every point.**
- Client: p50/p95/p99 TTFT, request TPOT and E2E; token/chunk gap tails; speed distribution; aggregate output tok/s; completed roots/s and requests/s; errors, timeouts, truncation and tool-parse failures; actual output lengths and joint SLO success fraction.
- Cache: token-weighted reads/inputs plus per-turn distribution, initial vs continuation; cache evictions/re-prefill. Missing cache fields mean **unknown**, not zero. Preserve SGLang extension fields from raw exports.
- Engine: running/queued sequences, batch sizes, scheduler delays, admitted-request limit, MLA KV used/capacity, KDA states used/capacity, retractions and natural speculative acceptance length. Scrape SGLang metrics/logs and GPU telemetry over matching time intervals.
- GPU/node: SM active, SM occupancy/tensor activity if available, HBM bandwidth, allocated memory, power, per-GPU variation; replica startup/idle/busy/billed seconds.
- Save raw exports; compute p95 from request records if absent in the summary. Count errors in the denominator rather than allowing failed requests to improve latency percentiles. Report goodput = output tokens from successful requests meeting **both** gates / measured seconds, alongside all successful output throughput.

**Step 5 — Read the frontier.** Plot p1/median decode speed (or p99 TPOT) against C, and overlay the target; plot p99 TTFT, aggregate goodput, queue depth and state occupancy alongside. Select the **largest repeatedly passing tested point**, not a visually interpolated promise. TTFT or errors can fail before the decode-speed crossing. Add operating margin (e.g. one tested rung below the boundary) and validate it.
Expected shape (**hypothesis**): nearly flat per-user speed while batching amortizes weight movement, launches and communication; a knee as compute/bandwidth/verification work fills; then falling speed and possibly throughput collapse from evictions/re-prefill or admission pressure. The node-specific suspects are:
- `max-mamba-cache-size=160` is a **state-slot pool, not 160 clients**. KDA working/cached/overlap state can consume multiple slots/request; cache strategy and version matter. A toy 4–5 slots/request suggests 32–40 running sequences before cached-state reserves, not a promised limit. Read actual boot/runtime caps. Current cookbook also warns of a speculative default running-request cap of 48; verify the pinned image rather than assuming it. [S8]
- CUDA graphs captured through batch 32 can change execution path above that size; this is not a hard concurrency cap, and speculative verification shape is not necessarily root count.
- 16k prefill chunks, including cold starts and p99 tool bursts, compete with decode and can create TTFT/ITL spikes. DCP8 relieves MLA KV pressure, not the KDA slot ceiling. [S8, S9]

**Step 6 — Open-loop validation.** At the chosen C, measure root completion rate; probe independent root arrival rates around 0.5/0.75/0.9/1.0/1.1× that rate with trace-distributed gaps. Use multi-turn `--profile poisson --arrival-rate LAMBDA` and a sufficiently high safety concurrency limit; report any time it binds. The standalone `--pattern poisson` makes independent single-turn requests, not agentic trees. Measure scheduled-arrival lateness and pending roots; reject a “passing” run with growing backlog or falling achieved/offered ratio.
For burst realism, use direct AIPerf Mooncake `timestamp/input_length/output_length/hash_ids` + `--fixed-schedule`; the wrapper does not expose `mooncake_trace`. Namespace hashes by session, preserve known reuse and routing. Fixed timestamps do not enforce live-response dependencies; distinguish recorded replay from a causal live agent. AgentX is a separate cross-check, not a substitute for your different workload or naturally measured DSPARK acceptance. [S2, S6]

### 3. Expected utilization numbers: what can and cannot be predicted

**Keep these quantities separate.**

| Utilization term | Definition / interpretation |
|---|---|
| GPU compute | Time-averaged SM/tensor activity and occupancy; generic “GPU util” is often just kernel-busy time, not percent of peak FLOPs. High busy time can coexist with low arithmetic efficiency. |
| MLA KV occupancy | Occupied KV blocks / usable KV pool; distinguish active/pinned vs evictable cached blocks. Full cache can be healthy, or a precursor to thrashing. |
| KDA/mamba occupancy | Used state slots / actual allocated slots (nominal 160); include working and retained checkpoints. Not C/160. |
| SLO capacity utilization | Achieved goodput / measured sustainable goodput at the same SLO and workload. A business-capacity metric, not SM utilization. |
| Billing utilization | Productive serving seconds / billed replica-seconds; also output goodput / billed second. Include billed startup and idle tails, not only decode time. |

**Trace → compute hypothesis.** ~99% is the median **per-response** cached fraction on the original providers, not guaranteed on SGLang. If history is actually reusable, a median continuation prefills roughly 356 new tokens rather than its ~81k re-sent history; sequential generation of ~537 output tokens can then dominate wall time. But the mean new input is 1266, first turns need ~10.7k, and long tools matter. Cache hits skip prefix recomputation, not MLA attention reads over history, KDA restoration/replay, memory residency or serialization. Therefore “decode-dominated warmed median” is plausible; “prefill is always negligible” is not.

**Estimated low C=2–8:** latency/launch/communication/weight-bandwidth limits and tool gaps likely leave considerable arithmetic capacity unused. Memory allocated to weights/cache remains large; utilization need not be low on a generic GPU-busy gauge. **Estimated C≥32:** batching should improve throughput/compute efficiency until KV/state admission and speculative verification limit it; cache pressure may lower useful goodput even while GPU busy time rises. There is no defensible 30%/80% SM prediction from concurrency alone. Measure the telemetry table and fit separate curves versus C, context and gaps.

**Before the endpoint exists, sanity-check rather than forecast:**
- **Literature:** the model card validates this NVFP4 SGLang 8×B300 shape but gives no matching single-user speed guarantee. SGLang's launch article reports ~113 tok/s non-spec and ~423 tok/s DSPARK batch-1 for K3; cookbook explicitly warns published B300 DSPARK runs used simulated acceptance. Those ~8.85ms/~2.36ms token intervals are optimistic reference scenarios, **not bounds guaranteed for this NVFP4/DCP8 trace load**. [S7–S9]
- Try assumed natural decode speeds `r=5,20,50,100 tok/s` as sensitivity cases. Under the knee, `T_out ≈ mean actively decoding streams × r`, not necessarily `C×r`. For sequential no-gap trees it approaches C×r; at C=8,r=50 that ceiling is 400 output tok/s, or 50 output tok/s/GPU—not 400 per GPU.
- With mean output O, TTFT F and inter-turn gap Z: `R ≈ F+(O−1)/r`, `requests/s ≈ C/(R+Z)`, `T_out ≈ C×O/(R+Z)`. Fan-out needs its measured active-stream multiplier; do not multiply by width on every turn. Stable open-loop output ≈ request arrivals/s × mean O, until capacity fails.
- Approximate memory with `resident unique context tokens × measured KV bytes/token + state slots × measured bytes/slot`, using the **resolved DCP/draft dtypes and capacities**. Do not count cached history as free or infer free HBM from `.85` alone; weights, graphs and draft reservations consume it.
- Model annual/weekly traffic separately. Trace activity occurred in only 122/721 touched hour bins; that ratio is not billing utilization or exact GPU busy time. With `MIN_CONTAINERS=0`, scaled-to-zero time costs no running-node hours, but reloading, warmup and idle tails can dominate paid time and violate cold-start TTFT. Use observed Modal billing intervals; quote a warm-ready SLO separately from cold arrivals. [Local trace profile and `serve.py`]

### 4. Worked example — entirely illustrative, not a forecast

Target: **p99 request TPOT ≤200ms (p1 speed ≥5 tok/s), p99 TTFT ≤2s**, no growing backlog; report joint success separately. Baseline: 24 turns, initial ~10,700 tokens, 356 new tokens/turn, actual mean output assumed 537. A session generates `24×537=12,888` tokens; approximate final context is `10,700 + 23×(356+537) ≈31,239`, ignoring templates. At the target's boundary one 537-token response takes up to roughly `2+536×0.2=109.2s` when both quantities equal their limits; **this is not a p99 E2E bound**. Twenty-four such turns take ~44 minutes before gaps: this is a permissive speed target for an agent, not necessarily a desirable experience.

**Invented sweep outcomes** below demonstrate the decision; speed is reciprocal p99 TPOT, not average user speed. Throughput is independently measured in a real run and cannot be derived from that tail statistic.

| C roots | p99 TTFT (s) | p99 TPOT (ms) | p1 speed (tok/s) | Aggregate output tok/s | Gate |
|---:|---:|---:|---:|---:|---|
| 2 | .45 | 40 | 25.0 | 60 | pass |
| 4 | .50 | 42 | 23.8 | 115 | pass |
| 8 | .65 | 50 | 20.0 | 210 | pass |
| 16 | .95 | 75 | 13.3 | 350 | pass |
| 32 | 1.70 | 160 | 6.25 | 480 | pass |
| 64 | 4.80 | 320 | 3.13 | 420 | fail |

Thus **32 is the largest tested passing point**, not the known true maximum; test 40/48 and repeat 24/32, then use arrival validation and margin. Suppose C=32 yields **480 total successful output tok/s, 470 SLO-compliant output tok/s** (illustrative token-weighted goodput; compute from records, not from 99%×throughput).

- Eight-GPU node: `470/8 = 58.75` compliant output tok/s/GPU.
- Component-only power assumption **10.6–13.5kW = .0106–.0135MW**: `470/.0135` to `470/.0106` = **34,815–44,340 compliant output tok/s/MW**. Using all 480 gives 35,556–45,283, but that is not SLO goodput. No PUE; do not compare this directly with facility/provisioned-MW charts.
- Rental assumption **$56.80/node-hour**: `56.80×1,000,000/(3600×470)` = **$33.57 / 1M compliant output tokens** at continuous steady service. General formula: `actual billed dollars ×1M / actual compliant output tokens`.
- If only 50% of billed time delivers that steady goodput, effective cost doubles to **$67.14/M**; 25% gives **$134.28/M**. This folds startup/idle into cost, not into SM utilization. With replicas, sum node-hours and power across replicas.
- For energy efficiency across variable load, integrate measured node power and divide compliant tokens by joules. The fixed 10.6–13.5kW denominator is a modeling scenario, not measured low-load consumption. Keep total output, compliant output, and visible-answer-only accounting distinct.

### 5. Pitfalls and operational checks

1. **Biggest: closed-loop “capacity” without offered-load validation.** Faster/slower responses change future offered load, so passing C can coexist with failure under production arrivals. Include client-side waiting and backlog, not only server-dispatched TTFT.
2. **False cache reuse:** generated sessions currently share a long prefix. Unique session IDs alone do not bust a content-keyed prefix cache; prepend per-replay content markers as AgentX's `first_turn_prefix` policy does. Preserve within-session matches. Randomizing each turn's whole prefix destroys the real advantage instead. [S2]
3. **Ignoring think/tool time:** replacing observed gaps by zero overstates GPU demand per active tree, understates supportable logged-in/active sessions and changes cache residency. Keep a zero-gap saturation arm and a separate trace-gap service arm; label both. Timestamp intervals that include generation must not be added again as think time.
4. **Speculation fiction:** repeated synthetic padding can change acceptance dramatically. Measure acceptance and compare matched NOSPEC/DSPARK runs on representative natural content before adopting literature speedups. AgentX's forced acceptance is a controlled cross-platform convention, not this endpoint's measured natural performance. [S2, S8]
5. **Cold starts and warmup contamination:** first boot/JIT/graph capture and priming need separate windows; warming everything conceals production cold misses. Report cold-start end-to-end readiness/latency and billed cost separately, then include cold traffic's actual production share.
6. **Censoring and mismatch:** cap-induced short outputs, missing reasoning history, overloaded client CPU/SSE parser, timeouts, context rejection, multi-replica routing and post-window drain can all improve a reported curve artificially. Audit raw counts, timing windows, replicas and failures before interpreting utilization.

## Sources

**Kept (primary sources and local implementation):**
- **Local:** `bench_agentic.py`, `generate_agentic_dataset.py`, `BENCHMARKS.md`, `serve.py`, [`../trace-profile/RECOMMENDATIONS.md`](../trace-profile/RECOMMENDATIONS.md) — actual flags, scheduling semantics, deployment and measured trace distributions.
- **S1:** [SemiAnalysis interactivity](https://inferencex.semianalysis.com/glossary/interactivity) and [throughput](https://inferencex.semianalysis.com/glossary/throughput) — iso-interactivity and token-accounting caveats.
- **S2:** [AgentX methodology/datasets](https://inferencex.semianalysis.com/agentx), [InferenceX methodology](https://inferencex.semianalysis.com/about) — trajectory warmup, closed-loop roots, cache busting, simulated acceptance.
- **S3:** [MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints) — p95 TTFT/interactivity curves, 7–32 points, ≥600 seconds.
- **S4:** [MLPerf v5.1 small LLM](https://mlcommons.org/2025/09/small-llm-inference-5-1) — primary evidence that latency limits differ by model/scenario.
- **S5:** [AIPerf 0.13.0 multi-turn](https://github.com/ai-dynamo/aiperf/blob/v0.13.0/docs/tutorials/multi-turn.md) — sessions, delays and root arrivals; use wrapper-generated flags/installed help where documentation aliases differ.
- **S6:** [AIPerf 0.13.0 trace replay](https://github.com/ai-dynamo/aiperf/blob/v0.13.0/docs/benchmark-modes/trace-replay.md) — timestamp/hash formats, fixed scheduling and recorded-history semantics.
- **S7:** [NVIDIA Kimi-K3-NVFP4 card](https://huggingface.co/nvidia/Kimi-K3-NVFP4) — validated 8×B300 runtime, DSPARK and context constraints.
- **S8:** [SGLang Kimi-K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) — state-pool limits, per-request slots, acceptance-simulation caveat; current docs may differ from pinned image.
- **S9:** [SGLang Kimi-K3 launch engineering report](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support) — batch-1 reference speeds, KDA/MLA behavior and load-dependent speculative cost.

**Dropped:** Morph Kimi-K3 overview and generic AIPerf blog — secondary summaries with potentially incompatible token accounting; GitHub long-context crash report — a single failing run, not a validated speed bound. Missing AIPerf `docs/reference/metrics.md` URL was not used as evidence.

## Gaps / next actions
No matching endpoint performance, natural DSPARK acceptance, actual slot accounting, power or billed-idle data exists yet. Prepare the session-unique trace dataset with gaps/topology and real token calibration; validate single-replica routing and resolved engine limits; run the sweep, then replace the illustrative table with measured goodput/telemetry. Retain uncertainty bands and the cold-start service policy before turning the selected C into a utilization or cost promise.
