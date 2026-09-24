"""Kimi-K2.6 inference endpoint on Modal (single-file).

Adapted from modal-projects/autoinference deployments/kimi_k2_6_8xb200, with
versions bumped to latest: SGLang v0.5.20, autoinference-utils 0.2.6, and
Modal's current client (installed via inline script metadata).

Deploy (scale-to-zero):
    uv run modal deploy serve.py

Keep one container always warm:
    MIN_CONTAINERS=1 uv run modal deploy serve.py
"""

import os
import re

# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5.5"]
# ///
import modal
import modal.experimental


def app_name_from_model(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.split("/")[-1].lower()).strip("-")


MINUTES = 60
AUTOINFERENCE_UTILS_VERSION = "0.2.6"
DEFAULT_PORT = 8000
HF_CACHE_PATH = "/root/hf-cache"
HF_CACHE_VOLUME_NAME = "huggingface-cache"
HF_IMAGE_ENV = {
    "HF_HOME": HF_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}

MODEL_NAME = "nvidia/Kimi-K2.6-NVFP4"
SPECULATOR_PATH = "lightseekorg/kimi-k2.6-eagle3"
app = modal.App(name=app_name_from_model(MODEL_NAME))

GPU_TYPE = "B200"
N_GPUS = 8
GPU = f"{GPU_TYPE}:{N_GPUS}"

SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.20"

DG_CACHE_DIR = "/root/dg-cache"
DG_CACHE_VOLUME_NAME = "dg-cache"

MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "0"))
SCALEDOWN_WINDOW = 10 * MINUTES
PROXY_REGIONS = os.getenv("PROXY_REGIONS", "us-west").split(",")
TARGET_INPUTS = 32
STARTUP_TIMEOUT = 1800

HF_CACHE_VOL = modal.Volume.from_name(HF_CACHE_VOLUME_NAME)
DG_CACHE_VOL = modal.Volume.from_name(DG_CACHE_VOLUME_NAME, create_if_missing=True)

serving_image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG)
    .pip_install("distro")
    .apt_install("git")
    .run_commands(
        # Raise uvicorn keep-alive so long streaming responses aren't cut.
        "sed -i 's/timeout_keep_alive=5/timeout_keep_alive=300/g'"
        " /sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py"
        " || true",
    )
    .env(
        HF_IMAGE_ENV
        | {
            "SGLANG_DG_CACHE_DIR": DG_CACHE_DIR,
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
    )
    .pip_install(f"autoinference-utils=={AUTOINFERENCE_UTILS_VERSION}")
)

with serving_image.imports():
    from autoinference_utils.endpoint import (
        SGLangEndpoint,
        start_heartbeat_thread,
        warmup_chat_completions,
    )

SERVER_ARGS = {
    "--trust-remote-code": "",
    "--tool-call-parser": "kimi_k2",
    "--reasoning-parser": "kimi_k2",
    "--dist-timeout": "3600",
    "--dtype": "bfloat16",
    "--kv-cache-dtype": "fp8_e4m3",
    "--quantization": "modelopt_fp4",
    "--attention-backend": "tokenspeed_mla",
    "--moe-runner-backend": "flashinfer_trtllm",
    "--mem-fraction-static": "0.7",
    "--chunked-prefill-size": "65536",
    "--max-running-requests": str(TARGET_INPUTS),
    "--cuda-graph-max-bs": str(TARGET_INPUTS),
    "--collect-tokens-histogram": "",
    "--speculative-algorithm": "EAGLE3",
    "--speculative-num-steps": "5",
    "--speculative-eagle-topk": "1",
    "--speculative-num-draft-tokens": "6",
    "--speculative-draft-model-quantization": "unquant",
    "--speculative-draft-attention-backend": "trtllm_mha",
    "--skip-server-warmup": "",
}

WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "A " * 32000}],
    "max_tokens": 1,
    "temperature": 0.0,
}


@app.cls(
    image=serving_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: HF_CACHE_VOL,
        DG_CACHE_DIR: DG_CACHE_VOL,
    },
    min_containers=MIN_CONTAINERS,
    timeout=30 * MINUTES,
    scaledown_window=SCALEDOWN_WINDOW,
    experimental_options={"override_eof_timeout": 30 * 60},
)
@modal.experimental.http_server(
    port=DEFAULT_PORT,
    proxy_regions=PROXY_REGIONS,
    exit_grace_period=25,
    startup_timeout=STARTUP_TIMEOUT,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class Server:
    @modal.enter()
    def startup(self):
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_NAME,
            speculative_model_path=SPECULATOR_PATH,
            worker_port=DEFAULT_PORT,
            tp=N_GPUS,
            extra_server_args=SERVER_ARGS,
            health_timeout=30 * MINUTES,
            health_poll_interval=10.0,
        )
        self.endpoint.start()
        warmup_chat_completions(
            port=DEFAULT_PORT,
            payload=WARMUP_PAYLOAD,
            successful_requests=3,
            request_timeout=120.0,
        )
        start_heartbeat_thread(
            self.endpoint.health_check,
            on_failure=lambda: modal.experimental.stop_fetching_inputs(),
        )
        print("Kimi-K2.6 (8xB200) is ready to serve.")

    @modal.exit()
    def stop(self):
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
