# Real conversation workload → agentic benchmark

**Window:** August 25–September 24, 2026, ending 20:10 UTC (before this profiling session). Scanned all **676 Codex, 4 Claude Code, and 1,731 pi transcript files**, not just recent mtimes. Within the window, 125/3/202 files had records; **117 Codex + 1 Claude + 198 pi = 316 sessions** contained model responses. There are **11,335 responses**, 11,334 with real usage. Two recent Claude sessions were authentication failures, not one-shot model requests.

Pi transcripts were found in `~/.pi/agent/sessions/**/*.jsonl`; `~/.pi/agent/run-history.jsonl` is only metadata. A search elsewhere under `~/.pi` found no other transcript store; the Application Support directory search found no pi-specific store. No conversation text, tool arguments, or credentials were exported.

## Observed distributions

Token counts unless indicated. **≈** means chars/4; zeros are retained. Quantiles are nearest-rank. Per-turn statistics weight responses, not sessions.

| Metric | p50 | Mean | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| Initial request, real input¹ | 10,664 | 11,642 | 22,264 | 36,412 | 217,628 |
| Responses/session, all sessions | 12 | 35.9 | 98 | 297 | 624 |
| Responses/session, agentic only | 24 | 49.5 | 114 | 338 | 624 |
| New input/response ≈² | 356 | 1,266 | 2,983 | 13,436 | 106,416 |
| Output/response, real | 243 | 537 | 1,201 | 4,894 | 20,223 |
| Re-sent input/request, real | 81,104 | 104,197 | 215,180 | 417,618 | 464,093 |
| Total input/session, real³ | 331,428 | 3,749,105 | 9,556,977 | 58,145,816 | 164,564,118 |
| Total output/session, real³ | 5,695 | 19,315 | 50,354 | 190,611 | 311,144 |
| Cached input fraction/response | 98.9% | 91.2% | 99.8% | 99.9% | ≈100% |
| Distinct sessions/active hour | 2 | 4.29 | 11 | 25 | 25 |

¹ Full first-request usage includes hidden instructions, schemas, and initial user context: better prefix calibration than incomplete transcript text. System text is present in only 120/316 sessions. The 217,628-token maximum is a warm fork with inherited history; normal non-fork first requests max out at 41,153. System + first-user text estimates and per-source distributions are in `distributions.md` / `summary.json`.

² Subsequent users + previous tool arguments + tool results + recorded injected context; initial input is excluded. Component counts are estimates because providers do not report tokens per message. Without arguments already retained in assistant history: **p50 150 / p90 2,698 / p99 13,238**.

³ 315 sessions with complete usage; input totals include repeated/cached history, not unique text or final context length. Codex cumulative checkpoints are differenced/deduplicated; Claude split blocks share one response; 84 inherited pi responses are counted only once. Pi input = input + cacheRead + cacheWrite; reasoning is already included in output and is not added again.

## Recommended starting run

```bash
python3 bench_agentic.py --base-url "$MODAL_URL" \
  --pattern multi-turn-growth \
  --agents 2 --concurrency 2 --turns 24 \
  --prefix-tokens 10700 --tail-tokens 350 --output-tokens 1280 \
  --num-requests 0 --cache-report
```

This is a **median-sized agentic smoke/load baseline**, not a complete production replay. Two conversations reflect active-hour p50=2 (also pi in-flight p90=2); 24 responses is agentic-session p50. Prefix 10,700 + tail 350 + generator boilerplate gives ≈11,147 initial tokens, matching agentic first-input p50=11,111; tail 350 matches p50=356. Output 1,280 is deliberately a **p90 cap** (agentic p90=1,212), not a p50 target: a 256 cap would truncate roughly half the observed responses. `--num-requests 0` avoids silently cutting longer finite conversation sweeps at the default 100 requests.

## Interpretation and variability (9 points)

- **Mix:** 227/316 sessions (**71.8%**) are multi-response tool-using runs; 89 (**28.2%**) are single-response. Pi is 97.0% multi-response, Codex 29.1%; one valid Claude run is too little to generalize. Of the 227 agentic sessions, 103 have only one user prompt—one prompt is not one model request.
- **Use sampled session lengths, not 24 everywhere:** agentic turns **24 / 114 / 338** at p50/p90/p99. Preserve a separate 28% single-response cohort. Pi supplies 89.2% of responses, so a source-balanced average would misrepresent actual request load.
- **Tail bursts dominate input:** tool results account for **82.5%** of estimated additions. Sample tails around **356 / 2,983 / 13,436**, with occasional 100k-token events; never inject p90 on every turn and call that typical. With live tool history, avoid double-counting arguments—use the excluding-args distribution above.
- **Outputs are heavy-tailed and reasoning-heavy:** **243 / 1,201 / 4,894** tokens at p50/p90/p99; recorded reasoning contributes **44.5% of all output tokens**. Text-only estimates have median 8 tokens and badly understate generation. Use **5,120 output cap** for p99 coverage; 1,280 deliberately misses the longest ≈10%. Caps and early EOS cannot force an observed output-length distribution.
- **Prefix/context must vary:** agentic initial-input p50/p90/p99 = **11,111 / 30,375 / 38,895**; include rare warm inherited contexts separately. Actual request context p50 is **81k**, far above what a 24-turn all-median run develops. For throughput/KV tests, sample full session trajectories or include warmed 81k/215k contexts; do not mistake the smoke baseline for memory pressure coverage.
- **Concurrency is bursty, not 11 simultaneous requests:** active-hour counts overstate concurrency. Five-minute-idle-split overlap has p50/p90/p99/max **1/3/5/7**; pi response timestamp overlap has **1/2/4/6**. Use agents/concurrency **2** initially, **3** for p90 session activity, **6** for a burst; timestamp overlap includes client latency and is not exact server GPU occupancy. Most hours are idle (122 active of 721 touched hour bins).
- **Fan-out is real but not universal:** **68 subagent launches across 27 sessions**, including 7 responses launching multiple children (max 3 together). Add a separate `fan-out --fan-out-width 2` or `3` case. Parallel *tool* batches have p50=3/p90=4/max=10; those are not ten independent subagents.
- **Cache target:** 95.1% of all input tokens are cache hits, but mean per-response hit rate is 91.2% because cold starts matter. Test cold and warmed runs separately with `--cache-report`; cache hit fraction is an observed outcome, not a padding knob. Repeating identical generated prefixes across unrelated agents may overstate real cross-session reuse; provider cache policy/TTL and sticky routing also differ.
- **Harness gap/minimal solution:** both files currently use constant prefix/tail/output/turn counts; `--seed` does not randomize generated sizes. Existing **`--dataset` supports heterogeneous native conversation rows**, so a seeded empirical bootstrap of saved per-session turn vectors can vary lengths without changing the harness. If adding CLI convenience later, change only dataset generation to sample per-session prefix/turn counts and per-turn tail/output limits, retaining correlations and occasional bursts. Sweep **tail=3,000**, **turns=114**, and **initial context≈30k** separately before combining marginal p90s; combining them creates an extreme workload and can exceed the endpoint context limit. No harness files were modified.

## Reproduction / verification

```bash
python3 analysis_profile_traces.py --self-test
python3 analysis_profile_traces.py --end 2026-09-24T20:10:00Z
```

Analyzer self-tests passed; full-data checks confirm 11,335 responses / 11,019 non-initial tails, bounded cache/reasoning counts, and exact agreement between Codex cumulative deltas and summed per-response usage. The recommended benchmark command passed `--dry-run`; no endpoint requests were sent. Results: `summary.json`, `sessions.jsonl` (numeric turn/session summaries), `distributions.md`, `run.log`, and `recommended-dry-run.jsonl`.
