"""Build corpus.jsonl: natural text chunks (with K3 token counts) from pi
chat logs. Sampled by bench_grid.py for load generation, so speculative-
decode acceptance sees real text distributions."""

import glob
import json
import os
import random
import sys

from analyze_chatdist import ENC, block_text

MIN_TOK, MAX_TOK = 10, 30_000

def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/avirus/.pi/agent/sessions/*/*.jsonl"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "corpus.jsonl")
    docs = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path, errors="replace"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "message":
                continue
            m = r.get("message", {})
            ct = m.get("content")
            if not isinstance(ct, list):
                continue
            for b in ct:
                txt = block_text(b)
                if len(txt) < 40:
                    continue
                n = len(ENC.encode_ordinary(txt))
                if MIN_TOK <= n <= MAX_TOK:
                    docs.append({"ntok": n, "text": txt})
    random.Random(0).shuffle(docs)
    with open(out, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    sizes = sorted(d["ntok"] for d in docs)

    def pct(p):
        return sizes[min(int(len(sizes) * p / 100), len(sizes) - 1)]

    print(f"docs={len(docs):,} total_tokens={sum(sizes):,} "
          f"p50={pct(50):,} p90={pct(90):,} max={sizes[-1]:,}")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
