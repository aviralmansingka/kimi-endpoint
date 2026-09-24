# Fireworks agentic harness validation

## Run report
The harness generated three eight-turn conversations, sent 24 streaming requests, parsed 24 SSE `[DONE]` markers, and produced native AIPerf metrics/raw exports with zero errors in 32.423 seconds. Aggregate throughput was 0.740 requests/s and 183.45 output tokens/s; median TTFT was 359.80 ms and median decode throughput was 316.07 tokens/s/user, plausible for a fast 3B-active MoE. Inputs grew from 1,046 to 4,203–4,399 tokens while median TTFT fell from 3,521.65 ms on turn 1 to 335.81 ms on turn 8; the server explicitly reported 44,480 cached input tokens (68.49%). One 18.17-second TTFT outlier inflated tail latency, and all responses contained reasoning rather than final-answer text, so this validates conversation growth and transport/metrics, not coding-task quality or a representative production benchmark.

## Environment and credential
- Existing uv: `/opt/homebrew/bin/uv`.
- Venv: `/Users/aviral/kimi-endpoint/.venv-bench`, Python 3.12.13.
- Verified AIPerf **0.13.0** and successful `import httpx`.
- Credential source: `~/.pi/agent/auth.json`, provider `fireworks`, field `key` (**redacted**). The same key also appeared in `~/.claude/settings.json`; no credential was printed or written to a temporary file. Native exports redact authorization.
- Model: `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` (NVIDIA Nemotron 3.5 Lightning 30B A3B).
- Published standard pricing: **$0.05 input / $0.01 cached input / $0.20 output per million tokens**, from https://docs.fireworks.ai/serverless/pricing (checked during run).

## Commands
```sh
uv venv .venv-bench
uv pip install --python .venv-bench/bin/python 'aiperf==0.13.0' httpx
.venv-bench/bin/aiperf --version
.venv-bench/bin/python -c "import httpx; print('httpx import OK')"
```
The live run was launched using this Python wrapper, keeping the key out of shell expansion/output:
```python
import os, pathlib, json, subprocess
key = json.loads((pathlib.Path.home()/'.pi/agent/auth.json').read_text())['fireworks']['key']
env = dict(os.environ, OPENAI_API_KEY=key)
cmd = ['.venv-bench/bin/python', 'bench_agentic.py',
       '--pattern', 'multi-turn-growth',
       '--base-url', 'https://api.fireworks.ai/inference/v1',
       '--model', 'accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b',
       '--agents', '3', '--turns', '8', '--prefix-tokens', '1024',
       '--tail-tokens', '256', '--output-tokens', '256',
       '--num-requests', '0', '--concurrency', '3',
       '--aiperf', '.venv-bench/bin/aiperf', '--timeout', '30',
       '--artifact-dir', 'artifacts/fireworks-validation']
result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
```
`server_root()` correctly removed the final `/v1`; native requests went to `https://api.fireworks.ai/inference/v1/chat/completions`.

Also ran `.venv-bench/bin/python bench_agentic.py --self-test`: dataset and driver self-tests passed.

## Native metrics
| Metric | p50 | p95 | p99 |
|---|---:|---:|---:|
| TTFT (ms) | 359.80 | 3,521.68 | 14,802.72 |
| ITL / per-request TPOT (ms) | 3.164 | 4.137 | 22.172 |
| E2E request latency (ms) | 1,202.23 | 4,469.26 | 16,295.30 |

- Requests: **24/24 HTTP 200**, error rate **0%**, exit code **0**.
- Duration: **32.423 s**; throughput **0.740 req/s**.
- Tokens: **64,948 input**, including **44,480 cached** and **20,468 uncached**; **5,949 output**; **70,897 total**.
- Output length: mean **247.875**, range **61–256**; 23 requests hit the length cap, one stopped naturally.
- Throughput: input **2,002.82 tok/s**; output **183.45 tok/s**; total **2,186.27 tok/s**.
- Decode throughput/user: mean **302.16**, median **316.07 tok/s**.
- Mean TTFT **1,550.73 ms**, ITL **4.254 ms**, E2E **2,411.47 ms**.

## Multi-turn checks
Raw payloads contained 2, 4, 6, …, 16 messages per thread, retaining the complete preceding message list and appending the live assistant response plus next user message. Assertions confirmed all three threads had eight turns and preserved each prior history prefix exactly. Typical input growth was **479 tokens/turn** (256 generated tokens plus approximately 223 new user/template tokens); the nominal prefix/tail knobs use character-based estimates, not exact model tokenization.

| Turn (1-based) | Median TTFT, ms |
|---|---:|
| 1 | 3,521.65 |
| 2 | 493.88 |
| 3 | 342.64 |
| 4 | 340.47 |
| 5 | 375.29 |
| 6 | 331.34 |
| 7 | 336.70 |
| 8 | 335.81 |

This confirms growing-history behavior and real prefix-cache hits. Later TTFT was broadly flat, not proportional to input size; cache hits were imperfect, and network/routing/load also affect latency, so this is not an isolated cache-speedup experiment. The largest outlier occurred at agent-0 turn 6 despite 2,961 cached tokens and returned only 61 tokens; its cause cannot be established from these exports.

## Surprises and limitations
- Authenticated `GET /inference/v1/models` returned HTTP 500 with `Error listing serverless models`, both through httpx and curl. The management model-list endpoint returned HTTP 403 requiring the employee Tailscale network. These discovery failures did **not** prevent successful inference.
- Two tiny non-streaming probes (`llama-v3p1-8b-instruct` and `gpt-oss-20b`, max_tokens=2) returned HTTP 404, with no usage/completion. The actual run used the currently published cheap Nemotron model instead.
- Every response used `reasoning_content`, with no final `content`. AIPerf accumulated that reasoning into assistant `content` for subsequent turns. This is acceptable evidence of synthetic history growth but is not faithful preservation of separate reasoning fields or proof of tool execution; no tools were exercised by this pattern.
- Initial TTFT was about 3.52 seconds and one later request took 18.17 seconds to first token. Fast median decode is plausible, but these tails are not uniformly low-latency serving. With only 24 requests, p99 is descriptive and unstable.
- No source files were changed. Existing modifications to `serve.py` were left untouched. Venv and benchmark artifacts are retained; no temporary credential/script files remain.

## Cost
Estimated list-price cost: `(20,468 × $0.05 + 44,480 × $0.01 + 5,949 × $0.20) / 1,000,000` = **$0.002658** (about **0.27 US cents**). Even ignoring cache discounts it is **$0.0044372**. Failed discovery/probes returned no billable token usage; actual account billing may differ.
