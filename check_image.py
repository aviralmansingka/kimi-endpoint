"""One-off: verify the dev image's SGLang revision contains the HiCache+DCP+DSPARK fixes.

modal run check_image.py
"""

import subprocess

import modal

from serve import sglang_image

check = modal.App("kimi-image-check")

# Modal mounts only the entry file: `from serve import ...` fails remotely
# unless the dependency rides along explicitly.
_check_image = sglang_image.add_local_python_source("serve")


@check.function(image=_check_image, timeout=20 * 60)
def revision():
    SRT = "/sgl-workspace/sglang/python/sglang/srt"
    SERVER_ARGS = f"{SRT}/server_args.py"  # legacy layout; hicache args MOVED
    HICACHE_HOOK = f"{SRT}/arg_groups/hicache_hook.py"  # new arg-group home
    KV_HOOK = f"{SRT}/arg_groups/kv_cache_hook.py"
    ASSEMBLER = f"{SRT}/mem_cache/hybrid_cache/hybrid_pool_assembler.py"
    LOADER = f"{SRT}/model_loader/loader.py"

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        print(f"$ {' '.join(cmd)}\n{out or '(no output)'}\n", flush=True)

    def found(path, pattern):
        return subprocess.run(
            ["grep", "-qE", pattern, path]
        ).returncode == 0

    run(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"])
    run(["git", "-C", "/sgl-workspace/sglang", "log", "--oneline", "-5"])
    run(["python", "-c",
         "import importlib.metadata as m; print(m.version('sglang'))"])
    # The nightly moved hicache args from server_args.py to arg_groups/.
    run(["grep", "-n", "-iE", "hicache|hierarchical|dcp|dspark", HICACHE_HOOK])
    run(["grep", "-n", "-iE", "dcp|draft", ASSEMBLER])

    gates = [
        ("hicache arg-group exists (new layout)",
         HICACHE_HOOK, "resolve_hicache_dcp_compatibility"),
        ("DSPARK allowed with HiCache+DCP (#35221)",
         HICACHE_HOOK, "only supports DSPARK"),
        ("DCP index translation in assembler", ASSEMBLER, "dcp_rank"),
        ("enable-hierarchical-cache flag still exists",
         KV_HOOK, "enable-hierarchical-cache"),
        ("NVFP4 checkpoint loading (#3507)", LOADER, "nvfp4"),
    ]
    missing = [name for name, path, pattern in gates if not found(path, pattern)]
    if missing:
        raise SystemExit(f"IMAGE MISSING FIX MARKERS: {missing}")
    print("IMAGE VERIFIED: all fix markers present", flush=True)
