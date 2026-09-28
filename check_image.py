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
    SERVER_ARGS = "/sgl-workspace/sglang/python/sglang/srt/server_args.py"
    ASSEMBLER = (
        "/sgl-workspace/sglang/python/sglang/srt/mem_cache/"
        "hybrid_cache/hybrid_pool_assembler.py"
    )

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
    # PR #35221 territory: DCP+HiCache+DSPARK draft host-pool sizing.
    run(["grep", "-rn", "-A30", "hicache_dcp", SERVER_ARGS])
    run(["grep", "-rn", "draft", ASSEMBLER])
    # Required HiCache flags must exist in this build.
    run(["grep", "-n", "hicache-size\\|hicache-write-policy\\|"
         "hicache-io-backend\\|hicache-mem-layout", SERVER_ARGS])

    gates = [
        ("hicache_dcp (#35221 draft host-pool DCP sizing)",
         SERVER_ARGS, "hicache_dcp"),
        ("draft pool handling in hybrid_pool_assembler", ASSEMBLER, "draft"),
        ("hicache-size flag", SERVER_ARGS, "hicache-size"),
        ("hicache-mem-layout flag", SERVER_ARGS, "hicache-mem-layout"),
    ]
    missing = [name for name, path, pattern in gates if not found(path, pattern)]
    if missing:
        raise SystemExit(f"IMAGE MISSING FIX MARKERS: {missing}")
    print("IMAGE VERIFIED: all fix markers present", flush=True)
