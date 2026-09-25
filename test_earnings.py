"""Offline check of earnings.py against the real fireworks AIPerf export."""

import earnings
from analyze_chatdist import P_IN, P_CACHE, P_OUT, NODE_KW, blended_price

export = earnings.load_export("fixtures/fireworks-aiperf.json")
pt = earnings.operating_point(export)
s = earnings.summarize(export, u=0.7)

# totals as reported by the server (AIPerf keeps run totals in .avg)
assert pt["prompt"] == 64948 and pt["cached"] == 44480 and pt["out"] == 5949
assert abs(pt["fresh"] - 20468) < 1

# cache share matches AIPerf's own overall_usage_prompt_cache_read_pct
assert abs(s["c"] - export["overall_usage_prompt_cache_read_pct"]["avg"] / 100) < 1e-3

# usage-derived D agrees with AIPerf's native output_token_throughput
assert abs(s["D"] - export["output_token_throughput"]["avg"]) / s["D"] < 0.01

# earnings arithmetic
expected_hr = (P_IN * s["P"] + P_CACHE * s["C"] + P_OUT * s["D"]) * 3600 / 1e6
assert abs(s["usd_per_hr"] - expected_hr) < 1e-9
expected_mw = 0.7 * expected_hr * 8760 / (NODE_KW / 1000)
assert abs(s["per_mw_yr_u"] - expected_mw) < 1e-6
assert abs(s["B"] - blended_price(s["R"], s["c"])) < 1e-9

print("earnings.py checks out against the fireworks fixture")
