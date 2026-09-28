"""One-off: verify the dev image's SGLang revision contains the HiCache+DCP+DSPARK fixes.

modal run check_image.py
"""

import subprocess

import modal

from serve import sglang_image

check = modal.App("kimi-image-check")


@check.function(image=sglang_image, timeout=20 * 60)
def revision():
    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        print(f"$ {' '.join(cmd)}\n{out or '(no output)'}\n", flush=True)

    run(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"])
    run(["git", "-C", "/sgl-workspace/sglang", "log", "--oneline", "-5"])
    run(["python", "-c",
         "import importlib.metadata as m; print(m.version('sglang'))"])
    # PR #35221 territory: DCP+HiCache+DSPARK draft host-pool sizing.
    run(["grep", "-rn", "-A30", "hicache_dcp",
         "/sgl-workspace/sglang/python/sglang/srt/server_args.py"])
    run(["grep", "-rn", "draft",
         "/sgl-workspace/sglang/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py"])
    # Required HiCache flags must exist in this build.
    run(["grep", "-n", "hicache-size\\|hicache-write-policy\\|"
         "hicache-io-backend\\|hicache-mem-layout",
         "/sgl-workspace/sglang/python/sglang/srt/server_args.py"])
