"""Self-check for the harness's load-bearing logic.

Run: uv run --with tiktoken --with aiohttp python test_harness.py
"""

from analyze_chatdist import (
    NODE_USD_PER_HR, blended_price, required_out_tps,
)
from bench_grid import Corpus, build_body, cell_shape, turn_suffix

# revenue model
assert round(blended_price(50, 0.95), 6) == 36.75
assert round(NODE_USD_PER_HR, 2) == 127.85
assert round(required_out_tps(36.75, 0.7)) == 1381

# session shape hits the 50:1 mix at the target cache share
assert cell_shape(0.0, 50, 336, 300) == (1, 16500)
assert cell_shape(0.95, 50, 336, 300) == (28, 829)

# prefix invariance: turn t's body extends turn t-1's byte-for-byte
corpus = Corpus("corpus.jsonl")
seed = corpus.sample(300)
note = "Reply in roughly 300 words."
suffix1 = turn_suffix(corpus.sample(829), note)
suffix2 = turn_suffix(corpus.sample(829), note)
body1 = build_body(seed, [], suffix1)
body2 = build_body(seed, [suffix1 + "\n\nASSISTANT:\nfake output"], suffix2)
assert body2.startswith(body1)

print("all harness invariants hold")
