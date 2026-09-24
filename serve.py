# Kimi-K3 on Modal with SGLang — low-latency `@app.server` structure.

# Serves nvidia/Kimi-K3-NVFP4 with the DSPARK speculative decoder, using the
# validated 8xB300 recipe from the model card:
#   https://huggingface.co/nvidia/Kimi-K3-NVFP4
# Weights are pre-fetched into a dedicated Modal Volume (kimi-k3-nvfp4-cache),
# so server boots load from the Volume instead of the Hub.
# No autoinference packages — SGLang is launched directly as a subprocess.

import asyncio
from collections import deque
import json
import os
import subprocess
import tempfile
import time

import aiohttp
import modal

MINUTES = 60  # seconds

# ## Container image

# The `lmsysorg/sglang:dev-dev-kimi-k3-nvfp4` image (CUDA 13 build from
# SGLang PR #35077) is the only published build that can load the NVFP4
# checkpoint; released SGLang versions cannot. Do not substitute a stock tag.

SGLANG_IMAGE_TAG = "lmsysorg/sglang:dev-dev-kimi-k3-nvfp4"

sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG)
    .entrypoint(
        []  # silence chatty logs on container start
    )
    .run_commands(
        # Raise uvicorn keep-alive so long streaming responses aren't cut.
        "sed -i 's/timeout_keep_alive=5/timeout_keep_alive=300/g'"
        " /sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py"
        " || true",
        # Modal requires an empty mountpoint. Build time has no mounted volume.
        "ls -la /root/.cache/huggingface; rm -rf /root/.cache/huggingface",
    )
)

# ## GPU choice

# B300: 288 GB per GPU; 8x is the validated TP8 config for Kimi-K3.

GPU_TYPE, N_GPUS = "B300", 8
GPU = f"{GPU_TYPE}:{N_GPUS}"

# ## Model and weights cache

MODEL_NAME = "nvidia/Kimi-K3-NVFP4"
SPECULATOR_NAME = "RadixArk/Kimi-K3-DSpark"

# Dedicated volume, pre-populated (1.6 TB) via the download-models function
# from the previous revision — mounted at HF_HOME, with model repos under hub/.

HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("kimi-k3-nvfp4-cache", create_if_missing=True)

# ## Kernel/compilation artifact cache

# The flashinfer_trtllm MoE kernels JIT-compile at first boot; cache the
# artifacts on a Volume so later boots skip that work.

DG_CACHE_PATH = "/root/dg-cache"
DG_CACHE_VOL = modal.Volume.from_name("dg-cache", create_if_missing=True)

sglang_image = sglang_image.env(
    {
        "HF_HOME": HF_CACHE_PATH,
        "HF_HUB_CACHE": f"{HF_CACHE_PATH}/hub",
        "HF_HUB_OFFLINE": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "SGLANG_DG_CACHE_DIR": DG_CACHE_PATH,
        # Kimi-K3 specifics
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_DISABLE_CUDNN_CHECK": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",  # required for DSPARK
    }
)

# ## SGLang server configuration

# DSPARK speculative decoding (cookbook default operating point: block size 7).
# Drop the speculative config below to serve without speculation.

speculative_config = {
    "speculative-algorithm": "DSPARK",
    "speculative-draft-model-path": SPECULATOR_NAME,
    "speculative-dspark-block-size": 7,
    # Required for spec dec with Kimi-K3's hybrid SSM arch.
    "enable-linear-replayssm-spec": None,
}

# None means a valueless switch: emit only the flag, never an empty argv element.
SERVER_ARGS = {
    "--trust-remote-code": None,
    "--tool-call-parser": "kimi_k3",
    "--reasoning-parser": "kimi_k3",
    "--dcp-size": str(N_GPUS),
    # RadixArk recipe values for Mamba cache capacity and chunked prefill.
    "--max-mamba-cache-size": "160",
    "--chunked-prefill-size": "16384",
    "--mem-fraction-static": "0.85",
    # Required, not optional: flashinfer_cutlass has no SiTU kernel for the
    # routed experts and auto-resolution never picks the TRT-LLM path.
    "--moe-runner-backend": "flashinfer_trtllm",
}

# ## Infrastructure

# Compute region is intentionally unset: Modal schedules the 8xB300 worker
# in ANY region with capacity (us-west B300s were persistently unavailable).
# Proxies stay pinned near the users; that is a routing choice, not a
# scheduling constraint.
ROUTING_REGION = "us-west"
MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "0"))  # 1 = always warm
TARGET_INPUTS = 32
CUDA_GRAPH_MAX_BS = 32  # Align graph capture with the Modal concurrency target.
STARTUP_TIMEOUT = 90 * MINUTES  # first boot loads ~1.6 TB from the Volume


def build_server_cmd(port):
    """The exact launch argv shared by GPU startup and CPU verification."""
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp",  # use all GPUs to split up tensor-parallel operations
        str(N_GPUS),
        "--cuda-graph-max-bs",  # only capture CUDA graphs for likely batch sizes
        str(CUDA_GRAPH_MAX_BS),
        "--enable-metrics",  # expose metrics endpoints for telemetry
        "--decode-log-interval",  # how often to log during decoding, in tokens
        "10",
    ]
    for flags in (SERVER_ARGS, speculative_config):
        for key, value in flags.items():
            cmd.append("--" + key.removeprefix("--"))
            if value is not None:
                cmd.append(str(value))
    return cmd


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        # Full output streams to container logs; surface the exit code loudly here.
        print(f"SGLang exited with return code {rc}; see container logs for the tail", flush=True)
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def wait_ready(process: subprocess.Popen, timeout: int = 90 * MINUTES):
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        # A dead child is not a transient connection failure: propagate immediately.
        check_running(process)
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(5)
    raise TimeoutError(f"SGLang server not ready within {timeout} seconds")


def warmup():
    import requests

    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=10
        ).raise_for_status()


# ## The server


app = modal.App(name="kimi-k3")
PORT = 8000


@app.server(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    min_containers=MIN_CONTAINERS,
    startup_timeout=STARTUP_TIMEOUT,
    port=PORT,  # wrapped code must listen on this port
    routing_region=ROUTING_REGION,  # location of proxies, should be close to region
    exit_grace_period=25,  # seconds, time to finish up requests when closing down
    target_concurrency=TARGET_INPUTS,
    unauthenticated=True,
)
class SGLang:
    @modal.enter()
    def startup(self):
        """Start the SGLang server, block until healthy, then warm it up."""
        cmd = build_server_cmd(PORT)
        # Merge SGLang's stderr (Python logging) into stdout so ALL output
        # streams to container logs — Modal drains container stdout, so there
        # is no pipe backpressure. This keeps `modal container logs` and the
        # app page live during the ~15-min weight load.
        self.process = subprocess.Popen(cmd, env=os.environ, stderr=subprocess.STDOUT)
        wait_ready(self.process)
        warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## CPU-only argv check: modal run serve.py::verify


@app.function(
    image=sglang_image,
    gpu=None,
    cpu=1,
    memory=2048,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    timeout=10 * MINUTES,
)
def check_argv():
    cmd = build_server_cmd(PORT)
    print("Launch argv:", json.dumps(cmd), flush=True)
    started = time.monotonic()
    timed_out = False
    returncode = None
    # Merge stdout/stderr so config dumps on either stream count as evidence.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as output:
        try:
            result = subprocess.run(
                cmd, env=os.environ, stdout=output, stderr=subprocess.STDOUT,
                timeout=3 * MINUTES,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True  # subprocess.run kills and reaps the child.
        output.seek(0)
        logs = output.read()

    lines = logs.splitlines()
    # Be conservative: imports can fail BEFORE argparse, especially without CUDA.
    # SGLang's resolved ServerArgs dump is positive evidence parsing completed.
    evidence = [line for line in lines if "server_args=ServerArgs(" in line]
    if returncode == 2 or any(
        marker in logs.lower()
        for marker in ("usage:", "error: unrecognized arguments", "error: argument")
    ):
        verdict = "FAIL"
        print(logs, flush=True)
    elif timed_out:
        verdict = "INCONCLUSIVE"
    elif evidence:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"

    if verdict != "FAIL":
        print("First 50 output lines:\n" + "\n".join(lines[:50]), flush=True)
        for line in evidence:
            print("Resolved args evidence:", line, flush=True)
    summary = {
        "verdict": verdict,
        "returncode": returncode,
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - started, 2),
        "scope": "argv only; not weight loading, kernels, or serving",
    }
    print(json.dumps(summary), flush=True)
    return summary


@app.local_entrypoint()
def verify():
    result = check_argv.remote()
    print("CPU argv verification:", json.dumps(result))
    if result["verdict"] == "FAIL":
        raise RuntimeError("SGLang rejected the launch argv; see output above")


# ## Test the server

# Run with `modal run serve.py::test` — spins up a fresh replica and streams a
# couple of test completions from your local machine.


@app.local_entrypoint()
async def test(test_timeout: int = 10 * MINUTES, prompt: str | None = None, twice: bool = True):
    url = await SGLang.get_url.aio()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if prompt is None:
        prompt = "Explain the Singular Value Decomposition."

    content = [{"type": "text", "text": prompt}]

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    await probe(url, messages, timeout=test_timeout)
    if twice:
        messages[0]["content"] = "You are Jar Jar Binks."
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=test_timeout)


# Send `Modal-Session-ID` with each request to get sticky routing across
# replicas, which improves KV cache hit rates for multi-turn conversations.


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    client_id = str(0)  # set this to some string per multi-turn interaction
    headers = {"Modal-Session-ID": client_id}
    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        while time.time() < deadline:
            try:
                await _send_request_streaming(session, messages)
                return
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
            except aiohttp.client_exceptions.ClientResponseError as e:
                if e.status == 503:
                    await asyncio.sleep(1)
                    continue
                raise e
    raise TimeoutError(f"No response from server within {timeout} seconds")


async def _send_request_streaming(
    session: aiohttp.ClientSession, messages: list, timeout: int | None = None
) -> None:
    payload = {"messages": messages, "stream": True}
    headers = {"Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        full_text = ""

        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Server-Sent Events format: "data: ...."
            if not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                # ignore any non-JSON keepalive
                continue

            delta = (evt.get("choices") or [{}])[0].get("delta") or {}
            chunk = delta.get("content")

            if chunk:
                print(chunk, end="", flush="\n" in chunk or "." in chunk)
                full_text += chunk
        print()  # newline after stream completes
