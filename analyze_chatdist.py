"""Blended price B for Kimi K3 pricing, measured from real pi chat logs.

A request is one assistant completion in a session (agent auto-continues
included).  With prefix caching, request t re-sends request t-1's prompt and
completion as a cached prefix:

    cached_t = prompt_{t-1} + out_{t-1}       (no eviction assumed)
    fresh_t  = prompt_t - cached_t            (new user text / tool results)
    out_t    = thinking + text + toolCall args (all billed as output)

Compaction resets the prefix: context := tokens(summary), cached := 0.

Run: uv run --with tiktoken python analyze_chatdist.py [sessions-glob]
TOK=qwen3 (plus --with tokenizers) swaps the tokenizer; P_IN / P_CACHE /
P_OUT / NODE_KW env vars swap the pricing and node (see run_qwen_smoke.sh).
"""

import glob
import json
import os
import sys
from collections import Counter

import tiktoken
from tiktoken.load import load_tiktoken_bpe

# ---- shared revenue model (bench_grid.py imports these) --------------------
# Defaults = Kimi K3 on OpenRouter + one 8xB300 node; override via env for
# other models/hardware.
P_IN = float(os.getenv("P_IN", "3.00"))        # $/Mtok fresh input
P_CACHE = float(os.getenv("P_CACHE", "0.30"))  # $/Mtok cache read
P_OUT = float(os.getenv("P_OUT", "15.00"))     # $/Mtok output
NODE_KW = float(os.getenv("NODE_KW", str(8 * 1400 / 1000)))  # TDP accounting
TARGET_PER_MW_YR = 100e6                  # the SemiAnalysis claim under test
NODE_USD_PER_HR = TARGET_PER_MW_YR * NODE_KW / 1000.0 / 8760.0


def blended_price(R, c):
    """$/M output tokens at in:out ratio R and cache-hit share c."""
    return P_IN * R * (1 - c) + P_CACHE * R * c + P_OUT


def required_out_tps(B, u):
    """Sustained output tok/s needed to earn the node's $/hr share at usage u."""
    return NODE_USD_PER_HR * 1e6 / (3600.0 * B * u)


# ---- exact Kimi K3 tokenizer (tiktoken.model from nvidia/Kimi-K3-NVFP4) ----
PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

_TOKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tok")


def _k3_encoding():
    """The exact K3 tiktoken BPE from the NVFP4 checkpoint."""
    ranks = load_tiktoken_bpe(os.path.join(_TOKDIR, "tiktoken.model"))
    return tiktoken.Encoding(
        name="kimi-k3", pat_str=PAT_STR, mergeable_ranks=ranks,
        special_tokens={f"<|reserved_token_{i}|>": i
                        for i in range(len(ranks), len(ranks) + 256)},
    )


class _QwenTokenizer:
    """tiktoken-compatible shim over the HF tokenizers Qwen3 tokenizer."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)

    def encode_ordinary(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_ordinary_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids):
        return self.tok.decode(ids)


def _load_tokenizer():
    if os.getenv("TOK") == "qwen3":
        return _QwenTokenizer(os.path.join(_TOKDIR, "qwen3-tokenizer.json"))
    return _k3_encoding()


ENC = _load_tokenizer()

CAP = int(os.getenv("CAP", "0"))  # ponytail: crude prompt cap for sensitivity runs


def ntok_texts(texts):
    """Total token count across a list of strings."""
    return sum(len(ids) for ids in ENC.encode_ordinary_batch(texts)) if texts else 0


def block_text(block):
    """The billable text of one content block ('' for non-text kinds)."""
    kind = block.get("type")
    if kind == "toolCall":
        args = block.get("arguments", block.get("input", ""))
        if not isinstance(args, str):
            args = json.dumps(args, separators=(",", ":"))
        return (block.get("name") or "") + args
    if kind in ("text", "thinking"):
        return block.get(kind) or ""
    return ""


def message_tokens(content):
    """Tokens in all text-bearing blocks of a message (str or block list)."""
    if isinstance(content, str):
        return ntok_texts([content])
    texts = [t for t in (block_text(b) for b in content or []) if t]
    return ntok_texts(texts)


def out_split(content):
    """Completion tokens by block kind: thinking / text / toolCall."""
    split = Counter()
    for block in content or []:
        text = block_text(block)
        if text:
            split[block["type"]] += len(ENC.encode_ordinary(text))
    return split


def process_file(path):
    """Yield (prompt, cached, fresh, out, split) per assistant completion."""
    ctx = 0     # tokens of context sent with the next request
    prefix = 0  # tokens reusable from the previous request (its prompt + out)
    for line in open(path, errors="replace"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = record.get("type")
        if kind == "compaction":
            ctx, prefix = ntok_texts([record.get("summary", "")]), 0
        elif kind == "custom_message":
            content = record.get("content")
            if isinstance(content, str):
                ctx += ntok_texts([content])
        elif kind == "message":
            message = record.get("message", {})
            role, content = message.get("role"), message.get("content")
            if role == "assistant":
                out = message_tokens(content)
                if out:
                    cached = min(prefix, ctx)
                    yield ctx, cached, ctx - cached, out, out_split(content)
                ctx, prefix = ctx + out, ctx + out
            elif role in ("user", "toolResult"):
                ctx += message_tokens(content)
                if CAP:
                    ctx = min(ctx, CAP)


def scan_sessions(files):
    """Every request across all sessions, plus session counts."""
    records, n_sessions, n_tool_sessions = [], 0, 0
    for path in files:
        n_sessions += 1
        session_records = list(process_file(path))
        records += session_records
        if any(r[4].get("toolCall") for r in session_records):
            n_tool_sessions += 1
    return records, n_sessions, n_tool_sessions


def token_totals(records):
    """Sum prompt/cached/fresh/out tokens, plus the out split by kind."""
    tot = Counter()
    for prompt, cached, fresh, out, split in records:
        tot["prompt"] += prompt
        tot["cached"] += cached
        tot["fresh"] += fresh
        tot["out"] += out
        for kind, n in split.items():
            tot["out_" + kind] += n
    return tot


def percentile(values, p):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p / 100), len(ordered) - 1)]


def print_counts(tot, records, n_sessions, n_tool_sessions):
    print(f"sessions={n_sessions} requests={len(records)} "
          f"sessions_with_tools={n_tool_sessions}")
    print(f"tokens: prompt={tot['prompt']:,} cached={tot['cached']:,} "
          f"fresh={tot['fresh']:,} out={tot['out']:,}")
    print(f"out split: thinking={tot['out_thinking']:,} "
          f"text={tot['out_text']:,} toolCall={tot['out_toolCall']:,}")


def print_mix(R, c, B, B_no_cache):
    print(f"in:out ratio R={R:.2f}  cache-hit c={c:.3f}")
    print(f"B = ${B:.2f}/M output tok   (no-cache contrast: ${B_no_cache:.2f})")


def print_request_stats(records, tot):
    prompts = sorted(r[0] for r in records)
    outs = sorted(r[3] for r in records)
    cached_frac = sorted(r[1] / r[0] for r in records if r[0])
    freshs = sorted(r[2] for r in records)
    big_cached = sum(r[1] for r in records if r[0] > 200_000) / tot["cached"]
    print()
    print("per-request stats: prompt p50/p90/p99 = "
          f"{percentile(prompts, 50):,}/{percentile(prompts, 90):,}/"
          f"{percentile(prompts, 99):,}")
    print("                 out p50/p90/p99 = "
          f"{percentile(outs, 50):,}/{percentile(outs, 90):,}/"
          f"{percentile(outs, 99):,}")
    print(f"                 cache-hit frac p50/p10 = "
          f"{percentile(cached_frac, 50):.3f}/{percentile(cached_frac, 10):.3f}")
    print(f"                 fresh/request p50/p90 = "
          f"{percentile(freshs, 50):,}/{percentile(freshs, 90):,} tok; "
          f"cached tokens in >200k-prompt reqs: {big_cached:.1%}")


def print_throughput_table(B):
    print()
    print(f"target: ${NODE_USD_PER_HR:,.0f}/hr per {NODE_KW:.1f} kW node "
          f"($100M/MW-yr)")
    for u in (1.0, 0.8, 0.7, 0.5):
        tps = required_out_tps(B, u)
        print(f"  u={u:.1f}: need {tps:,.0f} out tok/s sustained per node "
              f"({tps / 8:,.0f}/GPU) at B=${B:.2f}")


def print_projection_examples(B):
    print()
    for tps in (1000, 2000, 3000, 4000):
        per_node_yr = tps * 3600 * 24 * 365 * B / 1e6
        print(f"  measured {tps:,} tok/s/node, u=0.7 -> "
              f"${0.7 * per_node_yr / (NODE_KW / 1000) / 1e6:,.0f}M/MW-yr")


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/avirus/.pi/agent/sessions/*/*.jsonl"
    records, n_sessions, n_tool_sessions = scan_sessions(sorted(glob.glob(pattern)))
    tot = token_totals(records)
    R = tot["prompt"] / tot["out"]
    c = tot["cached"] / tot["prompt"]
    B = blended_price(R, c)

    print_counts(tot, records, n_sessions, n_tool_sessions)
    print_mix(R, c, B, P_IN * R + P_OUT)
    print_request_stats(records, tot)
    print_throughput_table(B)
    print_projection_examples(B)
    if CAP:
        print(f"[CAP={CAP:,} sensitivity run]")


if __name__ == "__main__":
    main()
