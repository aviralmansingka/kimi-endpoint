import modal
from serve import sglang_image, HF_CACHE_PATH, HF_CACHE_VOL, DG_CACHE_PATH, DG_CACHE_VOL

app = modal.App("kimi-cache-verification")

# Clear only the image's baked-in cache, before any volume is mounted.
verify_image = sglang_image.run_commands("find /root/.cache/huggingface -maxdepth 3 -type f -print; rm -rf /root/.cache/huggingface").add_local_python_source("serve")

@app.function(image=verify_image, volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL}, timeout=600, cpu=1, memory=2048)
def verify():
    import os, json, pathlib, struct, subprocess, sys
    root = pathlib.Path(HF_CACHE_PATH)
    print("ENV", {k: os.getenv(k) for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")}, flush=True)
    print("ROOT", sorted(p.name for p in root.iterdir()), flush=True)
    for repo in ("nvidia/Kimi-K3-NVFP4", "RadixArk/Kimi-K3-DSpark"):
        cache = root / "hub" / ("models--" + repo.replace("/", "--"))
        revision = (cache / "refs/main").read_text().strip()
        snapshot = cache / "snapshots" / revision
        files = [p for p in snapshot.rglob("*") if not p.is_dir()]
        broken = [str(p.relative_to(snapshot)) for p in files if not p.exists()]
        sizes = {str(p.relative_to(snapshot)): p.stat().st_size for p in files if p.exists()}
        shards = sorted(snapshot.glob("*.safetensors"))
        errors = []
        tensor_count = 0
        for p in shards:
            with p.open("rb") as f:
                length = struct.unpack("<Q", f.read(8))[0]
                if length > 100_000_000:
                    errors.append((p.name, "invalid header length", length))
                    continue
                header = json.loads(f.read(length))
            tensors = [v for k, v in header.items() if k != "__metadata__"]
            tensor_count += len(tensors)
            data_end = max(v["data_offsets"][1] for v in tensors)
            if 8 + length + data_end != p.stat().st_size:
                errors.append((p.name, "size mismatch", 8 + length + data_end, p.stat().st_size))
        index_file = snapshot / "model.safetensors.index.json"
        index_summary = None
        if index_file.exists():
            index = json.loads(index_file.read_text())
            expected = set(index["weight_map"].values())
            present = {p.name for p in shards}
            index_summary = {"indexed_shards": len(expected), "missing": sorted(expected - present), "extra": sorted(present - expected), "indexed_tensors": len(index["weight_map"])}
        config = json.loads((snapshot / "config.json").read_text())
        print("SNAPSHOT", json.dumps({"repo": repo, "path": str(snapshot), "recursive_files": len(files), "broken_links": broken, "safetensors": len(shards), "total_bytes": sum(sizes.values()), "header_errors": errors, "tensor_count": tensor_count, "index": index_summary, "other_files": {k:v for k,v in sizes.items() if not k.endswith(".safetensors")}, "shard_first": shards[0].name, "shard_last": shards[-1].name, "config": {k:config[k] for k in ("architectures", "model_type", "auto_map", "quantization_config") if k in config}}), flush=True)
        print("SYMLINK", shards[0], "->", os.readlink(shards[0]), "resolved", shards[0].resolve(), flush=True)
        card = snapshot / "README.md"
        if card.exists():
            lines = card.read_text().splitlines()
            for i, line in enumerate(lines):
                if any(x in line.lower() for x in ("lmsysorg/sglang", "runtime_patch", "8xb300", "8x b300")):
                    print("MODEL_CARD", repo, i+1, "\n".join(lines[max(0,i-2):i+4]), flush=True)
    code = '''
import os, json
import huggingface_hub
from huggingface_hub import snapshot_download, try_to_load_from_cache, constants
print("HF_CHECK", huggingface_hub.__version__, constants.HF_HUB_CACHE, "offline", constants.HF_HUB_OFFLINE, flush=True)
for repo in ("nvidia/Kimi-K3-NVFP4", "RadixArk/Kimi-K3-DSpark"):
    print("CONFIG_LOOKUP", repo, repr(try_to_load_from_cache(repo, "config.json")), flush=True)
    try:
        print("SNAPSHOT_LOOKUP", repo, snapshot_download(repo, local_files_only=True), flush=True)
    except Exception as e:
        print("SNAPSHOT_ERROR", repo, type(e).__name__, str(e), flush=True)
'''
    for cache_dir in (str(root), str(root / "hub")):
        env = dict(os.environ, HF_HUB_CACHE=cache_dir, HF_HUB_OFFLINE="1")
        result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=120)
        print("CACHE_TEST", cache_dir, "exit", result.returncode, result.stdout, result.stderr, flush=True)
    import importlib.metadata
    print("VERSIONS", {x: importlib.metadata.version(x) for x in ("sglang", "transformers", "huggingface_hub")}, flush=True)
    source_root = pathlib.Path("/sgl-workspace/sglang/python/sglang")
    for relative, needles in {
        "srt/server_args.py": ("--trust-remote-code", "--enable-linear-replayssm-spec"),
        "srt/model_loader/weight_utils.py": ("snapshot_download(", "cache_dir=", "HF_HUB_OFFLINE"),
        "srt/utils/hf_transformers_utils.py": ("cache_dir=", "HF_HUB_OFFLINE"),
    }.items():
        p = source_root / relative
        if p.exists():
            lines = p.read_text().splitlines()
            for i, line in enumerate(lines):
                if any(n in line for n in needles):
                    print("SGLANG_SOURCE", relative, i+1, "\n".join(lines[max(0,i-2):i+5]), flush=True)

@app.local_entrypoint()
def main():
    verify.remote()
