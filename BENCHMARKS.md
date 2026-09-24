# Agentic serving benchmarks: NVIDIA AIPerf

**Recommendation:** Use **NVIDIA AIPerf 0.13.0** for shared-prefix, constant/Poisson arrivals, growing conversations, and coordinator-driven fan-out. NVIDIA is phasing out GenAI-Perf in favor of AIPerf, and its own Kimi-K3 performance recipe uses AIPerf. This release's native `dag_jsonl` forks inherit the coordinator's actual response, so there is **no custom fan-out load generator**. Only the live tool loop needs an HTTP adapter: AIPerf does not execute tools or remap recorded result IDs to generated calls, and its generic history handling does not preserve Kimi's `reasoning_content` field. These are synthetic serving tests, **not official MLPerf submissions or agent-quality evaluations**.

## Install

AIPerf needs **Python 3.11–3.13**; the glue scripts need Python 3.10+. The client runs on macOS ARM without a GPU, CUDA, NGC image, or Modal deployment.

```bash
cd /Users/aviral/kimi-endpoint
python3.13 -m venv .venv-aiperf
source .venv-aiperf/bin/activate
python -m pip install 'aiperf==0.13.0' httpx
aiperf --version
aiperf profile --help
```

The verified local installation is also available at `/tmp/kimi-nvidia-bench-venv`; activate it or pass `--aiperf /tmp/kimi-nvidia-bench-venv/bin/aiperf`. Nothing was installed into the server image.

`serve.py` listens on port **8000** inside the server and exposes OpenAI `/v1/chat/completions`, using `kimi_k3` tool/reasoning parsers. The public proxy uses HTTPS **without `:8000`** and currently requires no authentication:

```bash
export MODAL_URL='https://aviralmansingka--kimi-k3-nvfp4-server-dev.us-west.modal.direct'
export MODEL='nvidia/Kimi-K3-NVFP4'
```

The wrapper defaults to **localhost**, not Modal, to avoid accidental cloud traffic. Root URLs, `/v1`, and full chat-completions URLs are accepted. Optional `OPENAI_API_KEY` / `--api-key` supports authenticated endpoints; neither is needed here. Payloads stream by default; use `--no-streaming` to disable SSE.

## Ready-to-run patterns

These examples are for use **after the server's startup issue is resolved**. They were not run against Modal.

```bash
# Long identical beginning with varying tails; fixed rate, bounded concurrency.
python bench_agentic.py --base-url "$MODAL_URL" --pattern shared-prefix \
  --prefix-tokens 8192 --tail-tokens 128 --profile constant --arrival-rate 2 \
  --concurrency 8 --num-requests 100 --artifact-dir artifacts/shared

# Four independent sessions; each re-sends prior live assistant replies.
# 0 disables the wire-request cap; the four eight-turn sessions still bound the run.
python bench_agentic.py --base-url "$MODAL_URL" --pattern multi-turn-growth \
  --agents 4 --turns 8 --concurrency 4 --num-requests 0 \
  --artifact-dir artifacts/growth

# Genuine live calls -> matching mock tool results -> assistant continuation.
# Four rounds produce eight HTTP requests per agent. No generated code is executed.
python bench_agentic.py --base-url "$MODAL_URL" --pattern tool-loop \
  --agents 2 --turns 4 --output-tokens 1024 --num-requests 0 \
  --save-results artifacts/live-tools.json

# Native DAG: one coordinator reply releases eight parallel child requests.
# --turns is the number of sequential turns IN EACH CHILD, not coordinator rounds.
python bench_agentic.py --base-url "$MODAL_URL" --pattern fan-out \
  --agents 1 --fan-out-width 8 --turns 1 --concurrency 1 --num-requests 0 \
  --artifact-dir artifacts/fanout

# Exponentially distributed background arrivals, mean 2 requests/second.
python bench_agentic.py --base-url "$MODAL_URL" --pattern poisson \
  --arrival-rate 2 --concurrency 8 --duration 60 --num-requests 0 \
  --artifact-dir artifacts/poisson
```

To overlay background traffic, run the Poisson command in one terminal and another pattern in a second terminal. Or append `&` to the Poisson command and `wait` after the foreground run. Their request limits, concurrency, artifacts and statistics are **independent**; there is no duplicate scheduler or combined-metrics engine in the glue.

For longer reasoning-output stress, raise `--output-tokens` (for example, `4096`). This is a **maximum**, not a promised output length. No `ignore_eos` override is injected; reasoning consumes the output budget too.

### Dataset generator / direct native commands

```bash
python generate_agentic_dataset.py --pattern shared-prefix --rows 100 \
  --prefix-tokens 8192 --output shared.jsonl
python generate_agentic_dataset.py --pattern multi-turn-growth --rows 4 \
  --turns 8 --output growth.jsonl
python generate_agentic_dataset.py --pattern fan-out --rows 1 \
  --fan-out-width 8 --turns 1 --output fanout.jsonl

# Run without the wrapper: all scheduling and metrics are AIPerf's.
aiperf profile --model "$MODEL" --url "$MODAL_URL" \
  --endpoint-type chat --streaming --tokenizer builtin --use-server-token-count \
  --input-file shared.jsonl --custom-dataset-type single_turn \
  --concurrency 8 --request-rate 2 --arrival-pattern constant --request-count 100 \
  --header Modal-Session-ID:shared-example --export-level raw \
  --artifact-dir artifacts/direct
```

Native formats:

- **Shared-prefix / Poisson:** `{"text":"shared prefix + varying tail","output_length":256}`. The identical schema/reference text is part of the user prompt; it does not trigger tool execution.
- **Multi-turn:** `{"session_id":"agent-0","turns":[{"role":"system","text":"shared prefix"},{"text":"first task","output_length":256},{"text":"follow-up","output_length":256}]}`. The leading system entry is not a separate request.
- **Fan-out:** `dag_jsonl` coordinator and child rows with `forks` references. Children inherit live parent history and start after its response completes.
- **Recorded tools:** `raw_payload` **directory**, one JSONL file per sequential conversation. Each line is a complete OpenAI request body, including model and streaming settings. Authored histories/IDs are not replaced by live responses.

```bash
# Metrics on recorded tool traffic, explicitly NOT a live tool loop.
# A new/empty directory is required to avoid stale extra conversation files.
python generate_agentic_dataset.py --pattern tool-loop --rows 2 --turns 4 \
  --output tool-replay/
python bench_agentic.py --base-url "$MODAL_URL" --pattern tool-loop --replay-tools \
  --dataset tool-replay/ --agents 2 --num-requests 0 \
  --artifact-dir artifacts/tool-replay
```

With `--dataset`, its authored lengths/messages replace synthetic payload knobs. Raw replay also owns its model/stream flags: regenerate the raw files with `--no-streaming` if required. `--cache-report` is rejected for raw replay because global extras do not alter raw bodies; put `return_cached_tokens_details` into those bodies yourself.

## Metrics, limits and caveats

- Use a fresh/empty `--artifact-dir` for each native run; the wrapper refuses stale artifact directories instead of mixing results. Each normal run prints a **3–4 sentence report**. AIPerf's full output is in `runner.log`, aggregate JSON/CSV in `profile_export_aiperf.*`, per-request metrics in `profile_export.jsonl`, and exact request/response records in `profile_export_raw.jsonl`. `--save-results PATH.json` copies the native summary; it does not recompute statistics.
- Native results include input/output tok/s, req/s, errors, request latency, TTFT, inter-token/inter-chunk latency, and p50/p90/p99. Use **AIPerf's metric definitions**, not an assumed equivalence to SGLang TPOT/E2E conventions. DSPARK and proxy buffering can put several tokens in one chunk; chunk latency is not token latency.
- `--tokenizer builtin --use-server-token-count` avoids downloading Kimi tokenizer files and uses **server usage** for real token counts. The builtin tokenizer is not Kimi's; when usage is absent, metrics can be unavailable rather than accurate estimates. Padding lengths are only chars/4 estimates.
- AIPerf captures standard `usage.prompt_tokens_details.cached_tokens`, including native cache-read totals/percentages. SGLang-only `sglext.cached_tokens_details`, flat `cached_tokens`, or `meta_info` may not be aggregated, but remain in the raw exports. `--cache-report` requests SGLang's version-dependent cache-details extension; absent reporting is **unknown**, not zero or inferred reuse.
- `--num-requests` counts **wire requests/turns**, including DAG children. Multi-turn/DAG/replay runs also set `--num-sessions` to `--agents`, avoiding accidental repeated first turns consuming the whole request budget. Dataset sessions are sampled executions; an external dataset can be cycled. Request/duration caps may truncate a conversation or a burst; use `--num-requests 0` for complete finite sessions.
- **Native `--concurrency` limits root sessions, not all DAG child HTTP requests.** One root with width 8 can send eight children even with concurrency 1. `--agents`, `--fan-out-width` and the request cap bound the fan-out workload. On native rate profiles, saturation can prevent achieving the offered rate; report achieved throughput.
- Native multi-turn/DAG history preserves generated text but does **not faithfully retain Kimi `reasoning_content` as its own field**. This can affect prefix-cache fidelity. The live tool adapter preserves both reasoning and native tool metadata, validates argument JSON/IDs/schema, returns harmless deterministic mock observations, and reports `tool_call_parse_failures` separately. Refusal/truncation can also cause that category; it is not proof of a server-parser bug.
- The live adapter saves raw timestamps, usage/cache fields, success/errors and complete response events; it deliberately has **no independent percentile/throughput engine**. Streaming is required for meaningful token-arrival timings. API traffic/response artifacts may contain sensitive data; keep them private.
- Modal routing uses `Modal-Session-ID`: one shared key for independent prefix requests, native per-session IDs for AIPerf conversations, and per-agent keys for live tools. Replicas, cold starts, context limits, client CPU/network and existing cache state affect results. AIPerf fork placement does not guarantee Modal places every child on the parent's GPU replica.

## Why not GenAI-Perf or classic MLPerf LoadGen?

| Tool | Installation / compatibility | Fit here |
|---|---|---|
| **AIPerf 0.13.0** | `pip install aiperf==0.13.0`; Python 3.11–3.13; macOS ARM/Linux; OpenAI chat + SSE; native JSONL/raw/DAG; rate/concurrency controls | Selected: NVIDIA's current replacement and Kimi-K3 recipe client. |
| **GenAI-Perf 0.0.16** | `pip install genai-perf`; full profiling also needs native `perf_analyzer`. Current native wheels target Linux, not macOS; use supported Linux pip or NVIDIA's Triton SDK NGC image. OpenAI chat, custom inputs, rate/concurrency supported | Its `--help` works locally, but pip selected **placeholder `perf-analyzer==0.1.0`**, leaving no analyzer binary. NVIDIA is phasing it out; adding a Linux/container layer buys little here. |
| **MLCommons tooling** | `mlc` is the CLI from `mlcflow` / `mlc-scripts`, not a `pip install mlc` package. Classic `mlcommons-loadgen` requires QSL plus SUT/QDL integration | Appropriate for a rules-compliant MLPerf workflow, not a zero-glue OpenAI endpoint benchmark. It does not natively accept this URL plus custom JSONL and handle SSE. |

Sources:
- [NVIDIA GenAI-Perf installation and phase-out notice](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)
- [AIPerf](https://github.com/ai-dynamo/aiperf), [native custom data](https://github.com/ai-dynamo/aiperf/blob/v0.13.0/docs/tutorials/custom-dataset.md), [DAG schema](https://github.com/ai-dynamo/aiperf/blob/v0.13.0/src/aiperf/dataset/loader/dag_jsonl_models.py), [raw replay](https://github.com/ai-dynamo/aiperf/blob/v0.13.0/docs/tutorials/raw-payload-replay.md)
- [NVIDIA Kimi-K3 recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3/perf/README.md): existing AIPerf/Mooncake 64K-input, 400-output, 90%-reuse load. It is synthetic cache-reuse replay, not a live tool benchmark. Do not copy its hardware concurrency or synthetic speculative-acceptance settings into a natural DSPARK comparison.
- [MLCFlow](https://github.com/mlcommons/mlcflow), [MLPerf automation](https://github.com/mlcommons/mlperf-automations), [LoadGen adapter requirements](https://github.com/mlcommons/inference/blob/master/loadgen/README.md)

## Local verification

```bash
python -m py_compile bench_agentic.py generate_agentic_dataset.py
python bench_agentic.py --help
python generate_agentic_dataset.py --help
python bench_agentic.py --self-test
python generate_agentic_dataset.py --self-test
python bench_agentic.py --pattern fan-out --agents 1 --turns 1 --dry-run
python bench_agentic.py --pattern tool-loop --agents 1 --turns 2 --dry-run
```

Verified installed AIPerf 0.13.0 and GenAI-Perf 0.0.16 help/version commands, native schema/DAG loading, and mocked localhost HTTP/SSE runs for every pattern (including recorded tool replay). The mock tests check live conversation history, the parent-response barrier and overlapping children, fragmented tool arguments, real call IDs, retained Kimi reasoning, usage/cache capture, nonstreaming replies and distinct missing-tool errors. No real Modal endpoint was probed, no deployment was made, and `serve.py` was not modified by this work.
