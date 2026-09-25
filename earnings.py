#!/usr/bin/env python3
"""Earnings from an AIPerf export (profile_export_aiperf.json).

    P_IN=... P_CACHE=... P_OUT=... NODE_KW=... python earnings.py <artifact-dir|json>

Reads the server-reported usage totals — real token counts, including cache
reads when the run used --cache-report — and prints the measured operating
point (P fresh prefill / C cache-read / D output, tok/s), the mix (R, c),
blended price B, and earnings: $/hr while serving, $/MW-yr at usage factor
U (env, default 0.7).  Prices and node watts come from analyze_chatdist
(same envs, same defaults).  No pass/fail bar — this reports what was earned.
"""

import json
import os
import sys

from analyze_chatdist import P_IN, P_CACHE, P_OUT, NODE_KW, blended_price

EXPORT_NAME = "profile_export_aiperf.json"


def load_export(path):
    """The export JSON from a file path or an AIPerf artifact dir."""
    if os.path.isdir(path):
        path = os.path.join(path, EXPORT_NAME)
    with open(path) as f:
        return json.load(f)


def total(export, name):
    """Run total of a counter metric (AIPerf keeps totals in .avg)."""
    return export[name]["avg"]


def operating_point(export):
    """Server-reported token totals for the run, split fresh/cached/out."""
    prompt = total(export, "total_usage_prompt_tokens")
    cached = export.get("total_usage_prompt_cache_read_tokens", {}).get("avg", 0.0)
    return {
        "prompt": prompt,
        "cached": cached,
        "fresh": prompt - cached,
        "out": total(export, "total_usage_completion_tokens"),
        "duration": total(export, "benchmark_duration"),
        "reqs": total(export, "completed_request_count"),
    }


def summarize(export, u):
    """Measured operating point + earnings at usage factor u."""
    pt = operating_point(export)
    seconds = pt["duration"]
    P, C, D = pt["fresh"] / seconds, pt["cached"] / seconds, pt["out"] / seconds
    R = pt["prompt"] / pt["out"] if pt["out"] else 0.0
    c = pt["cached"] / pt["prompt"] if pt["prompt"] else 0.0
    B = blended_price(R, c)
    usd_per_hr = (P_IN * P + P_CACHE * C + P_OUT * D) * 3600 / 1e6
    return {
        **pt, "P": P, "C": C, "D": D, "R": R, "c": c, "B": B,
        "usd_per_hr": usd_per_hr,
        "per_mw_yr_u": u * usd_per_hr * 8760 / (NODE_KW / 1000),
        "native_out_tps": export.get("output_token_throughput", {}).get("avg"),
        "ttft_p95_s": export.get("time_to_first_token", {}).get("p95", 0) / 1e3,
        "itl_p95_ms": export.get("inter_token_latency", {}).get("p95", 0),
        "error_rate": export.get("request_error_rate", {}).get("avg", 0.0),
    }


def report(s, u):
    """One-screen human summary of a run."""
    cache_pct = 100 * s["c"]
    lines = [
        f"requests={s['reqs']:.0f} duration={s['duration']:.1f}s | "
        f"prompt={s['prompt']:,.0f} ({cache_pct:.1f}% cached) "
        f"fresh={s['fresh']:,.0f} out={s['out']:,.0f}",
        f"P={s['P']:,.0f} fresh tok/s  C={s['C']:,.0f} cache tok/s  "
        f"D={s['D']:,.0f} out tok/s"
        + (f"  [native out tps {s['native_out_tps']:.1f}]"
           if s["native_out_tps"] else ""),
        f"R={s['R']:.1f} c={s['c']:.3f} | B=${s['B']:.2f}/M out tok",
        f"earn ${s['usd_per_hr']:.2f}/hr serving -> "
        f"${s['per_mw_yr_u'] / 1e6:,.1f}M/MW-yr @u{u}  [{NODE_KW:g} kW node]",
        f"TTFT p95={s['ttft_p95_s']:.1f}s  ITL p95={s['itl_p95_ms']:.1f}ms  "
        f"errors={s['error_rate']:.1f}%",
    ]
    if s["native_out_tps"] and abs(s["D"] - s["native_out_tps"]) > 0.05 * s["D"]:
        lines.append("WARNING: usage-derived D disagrees with native "
                     "output_token_throughput by >5% — inspect the export.")
    if not s["cached"]:
        lines.append("NOTE: no cache-read tokens in the export — run with "
                     "--cache-report, or C=0 is being assumed.")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    u = float(os.getenv("U", "0.7"))
    print(report(summarize(load_export(sys.argv[1]), u), u))


if __name__ == "__main__":
    main()
