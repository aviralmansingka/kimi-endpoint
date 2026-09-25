# Qwen3-30B-A3B-Instruct-2507-FP8 on one A100-80GB — cheap harness-verification
# server.  Same Modal pattern as serve.py (kimi-k3) minus the K3 specifics:
# reuses the already-pulled SGLang dev image, no DSPARK, no NVFP4, fp8 KV so
# the R=50 c=0.95 cell's ~550k resident prefix tokens fit next to 31 GB of
# weights.

import os
import subprocess
import time

import modal

MINUTES = 60  # seconds

# Reuse the image already pulled for serve.py — it serves Qwen3 MoE fine.
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-dev-kimi-k3-nvfp4")
    .entrypoint([])  # silence chatty logs on container start
)

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
GPU = os.getenv("GPU", "A100-80GB")  # H100 swap when A100 capacity stalls

HF_CACHE_PATH = "/root/hf-cache"  # image ships files at the HF default path
HF_CACHE_VOL = modal.Volume.from_name(
    "qwen3-30b-fp8-cache", create_if_missing=True
)

sglang_image = sglang_image.env(
    {
        "HF_HOME": HF_CACHE_PATH,
        "HF_HUB_CACHE": HF_CACHE_PATH + "/hub",  # volume layout: hub/ at root
        "HF_XET_HIGH_PERFORMANCE": "1",
    }
)

MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "0"))  # 1 = always warm
STARTUP_TIMEOUT = 30 * MINUTES  # first boot pulls ~31 GB from the Hub
TARGET_CONCURRENCY = 40         # keep one replica at K=32
PORT = 8000


def wait_ready(process, timeout):
    """Block until the SGLang HTTP server answers /health."""
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        if (rc := process.poll()) is not None:
            raise subprocess.CalledProcessError(rc, cmd=process.args)
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except requests.exceptions.RequestException:
            time.sleep(5)
    raise TimeoutError(f"SGLang not ready within {timeout}s")


app = modal.App(name="qwen3-smoke")


@app.server(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    min_containers=MIN_CONTAINERS,
    startup_timeout=STARTUP_TIMEOUT,
    port=PORT,
    target_concurrency=TARGET_CONCURRENCY,
    unauthenticated=True,
)
class SGLang:
    @modal.enter()
    def startup(self):
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
            str(PORT),
            "--cuda-graph-max-bs", "64",
            "--kv-cache-dtype", "fp8_e4m3",   # ~26 GB KV for the c=0.95 cell
            "--mem-fraction-static", "0.9",
            "--enable-metrics",
            "--enable-cache-report",  # usage carries cached_tokens for AIPerf
        ]
        self.process = subprocess.Popen(cmd, env=os.environ)
        wait_ready(self.process, STARTUP_TIMEOUT)

    @modal.exit()
    def stop(self):
        self.process.terminate()
