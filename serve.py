# Kimi-K3 on Modal with SGLang — low-latency `@app.server` structure.

# Serves nvidia/Kimi-K3-NVFP4 with the DSPARK speculative decoder, using the
# validated 8xB300 recipe from the model card:
#   https://huggingface.co/nvidia/Kimi-K3-NVFP4
# Weights are pre-fetched into a dedicated Modal Volume (kimi-k3-nvfp4-cache),
# so server boots load from the Volume instead of the Hub.
# No autoinference packages — SGLang is launched directly as a subprocess.

import asyncio
import json
import os
import subprocess
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
# from the previous revision — still valid, mounted at the canonical HF path.

HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name(
    "kimi-k3-nvfp4-cache", create_if_missing=True
)

# ## Kernel/compilation artifact cache

# The flashinfer_trtllm MoE kernels JIT-compile at first boot; cache the
# artifacts on a Volume so later boots skip that work.

DG_CACHE_PATH = "/root/dg-cache"
DG_CACHE_VOL = modal.Volume.from_name("dg-cache", create_if_missing=True)

sglang_image = sglang_image.env(
    {
        "HF_HOME": HF_CACHE_PATH,
        "HF_HUB_CACHE": HF_CACHE_PATH,
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
# Drop the three speculative args below to serve without speculation.

speculative_config = {
    "speculative-algorithm": "DSPARK",
    "speculative-draft-model-path": SPECULATOR_NAME,
    "speculative-dspark-block-size": 7,
    # Required for spec dec with Kimi-K3's hybrid SSM arch.
    "enable-linear-replayssm-spec": "",
}

SERVER_ARGS = {
    "--trust-remote-code": "",
    "--tool-call-parser": "kimi_k3",
    "--reasoning-parser": "kimi_k3",
    "--dcp-size": str(N_GPUS),
    "--mem-fraction-static": "0.85",
    # Required, not optional: flashinfer_cutlass has no SiTU kernel for the
    # routed experts and auto-resolution never picks the TRT-LLM path.
    "--moe-runner-backend": "flashinfer_trtllm",
}

# ## Infrastructure

REGION = "us-west"
ROUTING_REGION = "us-west"
MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "0"))  # 1 = always warm
TARGET_INPUTS = 32
STARTUP_TIMEOUT = 90 * MINUTES  # first boot loads ~1.6 TB from the Volume


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def wait_ready(process: subprocess.Popen, timeout: int = 90 * MINUTES):
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
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
    compute_region=REGION,
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
            f"{PORT}",
            "--tp",  # use all GPUs to split up tensor-parallel operations
            f"{N_GPUS}",
            "--cuda-graph-max-bs",  # only capture CUDA graphs for likely batch sizes
            f"{TARGET_INPUTS * 2}",
            "--enable-metrics",  # expose metrics endpoints for telemetry
            "--decode-log-interval",  # how often to log during decoding, in tokens
            "10",
        ]

        cmd += [
            item for k, v in SERVER_ARGS.items() for item in (k, str(v))
        ]
        cmd += [  # add speculative config
            item for k, v in speculative_config.items() for item in (f"--{k}", str(v))
        ]

        self.process = subprocess.Popen(cmd, env=os.environ)
        wait_ready(self.process)
        warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Test the server

# Run with `modal run serve.py` — spins up a fresh replica and streams a
# couple of test completions from your local machine.


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, prompt=None, twice=True):
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
