"""Concurrency x fresh-share grid harness for the kimi-k3 Modal endpoint.

Maps the prefill-decode frontier.  Each cell holds K sticky sessions whose
shape targets the 50:1 mix at cache-hit share c:

    fresh tokens/turn  F = R(1-c)O
    turns/session      n = 1 + 2cR / (R(1-c) + 1)

Each turn appends only a small suffix of new bytes; everything before it is
re-sent byte-identical, so the KV prefix cache hits by construction:

    body_t = seed + (suffix_1 + out_1) + ... + suffix_t

Per cell it reports the measured operating point (P fresh prefill tok/s,
C cache-read tok/s, D output tok/s), measured R and c, blended B, TTFT/ITL
p95, and earnings: $/hr while serving and $/MW-yr at a usage factor u
(--u-report, default 0.7).  No pass/fail target — the goal is to measure
what the node earns.

Server must be warm (MIN_CONTAINERS=1 modal deploy serve.py).  Run:

    uv run --with aiohttp --with tiktoken python bench_grid.py \
        --url https://<modal-proxy-url> [--ks 8,16,32,64] \
        [--cshares 0.0,0.5,0.9,0.95,0.99]
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp

from analyze_chatdist import ENC, P_IN, P_CACHE, P_OUT, NODE_KW, blended_price

RETRY_DELAYS = (1, 2, 4, 8, 16)


# ---- session shape ---------------------------------------------------------

def cell_shape(c, R, out_per_turn, seed_tok):
    """(turns, fresh tokens/turn) so a session's cumulative traffic hits (R, c)."""
    if c >= 0.999:
        return 80, 0  # pure-replay corner
    turns = max(1, round(1 + 2 * c * R / (R * (1 - c) + 1)))
    fresh = max(0, round(R * (1 - c) * out_per_turn - seed_tok / turns))
    return turns, fresh


# ---- natural-text corpus ---------------------------------------------------

class Corpus:
    """Endless shuffled stream of text chunks from corpus.jsonl."""

    def __init__(self, path):
        self.docs = [json.loads(line) for line in open(path)]
        random.Random(0).shuffle(self.docs)
        self.i = 0

    def sample(self, ntok):
        """~ntok tokens of natural text."""
        if ntok <= 0:
            return ""
        parts, total = [], 0
        while total < ntok:
            doc = self.docs[self.i % len(self.docs)]
            self.i += 1
            parts.append(doc["text"])
            total += doc["ntok"]
        if total > 1.5 * ntok:  # last chunk overshot: keep cell shape honest
            parts[-1] = fit_text(parts[-1], ntok - (total - doc["ntok"]))
        return "\n\n".join(parts)


def fit_text(text, ntok):
    """First ~ntok tokens of text (chars ~= 4x tokens, then an exact trim)."""
    ids = ENC.encode_ordinary(text[: ntok * 4])
    return ENC.decode(ids[:ntok])


# ---- one streaming request -------------------------------------------------

async def sse_data_bodies(resp):
    """Yield each SSE 'data:' payload from a streaming response."""
    buffer = b""
    async for chunk in resp.content.iter_any():
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.decode("utf-8", "ignore").strip()
            if line.startswith("data:"):
                body = line[5:].strip()
                if body and body != "[DONE]":
                    yield body


def delta_of(event):
    """The delta dict of a completion-stream event ({} if absent)."""
    return (event.get("choices") or [{}])[0].get("delta") or {}


async def collect_stream(resp, started):
    """Assemble one streamed completion: text, TTFT, mean ITL, usage."""
    text, stamps, ttft, usage = [], [], None, None
    async for body in sse_data_bodies(resp):
        try:
            event = json.loads(body)
        except json.JSONDecodeError:
            continue  # keepalive noise
        usage = event.get("usage") or usage
        piece = delta_of(event).get("content") or \
            delta_of(event).get("reasoning_content")
        if piece:
            stamps.append(time.monotonic())
            ttft = ttft if ttft is not None else stamps[0] - started
            text.append(piece)
    itl = (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else 0.0
    return {"text": "".join(text), "ttft": ttft or 0.0, "itl": itl,
            "usage": usage}


async def stream_completion(http, ctx, session_id, body):
    """POST one streaming chat completion against the endpoint."""
    payload = {
        "messages": [{"role": "user", "content": body}],
        "max_tokens": ctx["max_tokens"],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Modal-Session-ID": session_id, "Accept": "text/event-stream"}
    started = time.monotonic()
    async with http.post(f"{ctx['url']}/v1/chat/completions", json=payload,
                         headers=headers,
                         timeout=aiohttp.ClientTimeout(total=ctx["timeout"])) as resp:
        resp.raise_for_status()
        return await collect_stream(resp, started)


async def completion_with_retries(http, ctx, session_id, body):
    """stream_completion with backoff; None if the endpoint stays dead."""
    for delay in RETRY_DELAYS:
        try:
            return await stream_completion(http, ctx, session_id, body)
        except Exception:
            await asyncio.sleep(delay)
    return None


# ---- one sticky session ----------------------------------------------------

def turn_suffix(fresh_text, note):
    """The only new bytes a turn appends: the user chunk plus the note."""
    user = f"\n\nUSER:\n{fresh_text}" if fresh_text else ""
    return f"{user}\n\n{note}"


def build_body(seed, history, suffix):
    """seed + prior turns + this suffix: a byte-extension of the last body."""
    return seed + "".join(history) + suffix


def request_record(cell, turn, body, new_bytes, result):
    """One measured request, in the units the revenue model needs."""
    prompt_tok = len(ENC.encode_ordinary(body))
    fresh_tok = len(ENC.encode_ordinary(new_bytes))
    return {
        "k": cell["k"], "c_target": cell["c"], "turn": turn,
        "prompt_tok": prompt_tok, "fresh_tok": fresh_tok,
        "cached_tok": prompt_tok - fresh_tok,
        "out_tok": len(ENC.encode_ordinary(result["text"])),
        "ttft": result["ttft"], "itl": result["itl"],
        "usage": result["usage"], "done_at": time.monotonic(),
    }


async def run_session(worker_id, cell, ctx, records, http, deadline):
    """Run one sticky session for cell['turns'] turns, recording each request."""
    corpus = ctx["corpus"]
    session_id = f"k{cell['k']}-c{cell['c']}-w{worker_id}"
    seed = corpus.sample(ctx["seed_tok"])
    history = []
    for turn in range(1, cell["turns"] + 1):
        if time.monotonic() >= deadline:
            return
        fresh = corpus.sample(cell["fresh"]) if cell["fresh"] else ""
        suffix = turn_suffix(fresh, ctx["note"])
        new_bytes = (seed + suffix) if turn == 1 else suffix
        body = build_body(seed, history, suffix)
        result = await completion_with_retries(http, ctx, session_id, body)
        if result is None:
            return  # dead session; the worker starts a new one
        history.append(suffix + f"\n\nASSISTANT:\n{result['text']}")
        records.append(request_record(cell, turn, body, new_bytes, result))


async def run_worker(worker_id, cell, ctx, records, http):
    """Back-to-back sessions until the cell deadline."""
    deadline = cell["start"] + cell["warmup"] + cell["duration"]
    while time.monotonic() < deadline:
        await run_session(worker_id, cell, ctx, records, http, deadline)


# ---- per-cell summary ------------------------------------------------------

def steady_records(records, cell):
    """Requests that finished inside the measurement window."""
    lo = cell["start"] + cell["warmup"]
    hi = lo + cell["duration"]
    return [r for r in records if lo <= r["done_at"] <= hi]


def token_sums(records):
    """Sum the four token counters across records."""
    sums = {"prompt": 0, "fresh": 0, "cached": 0, "out": 0}
    for r in records:
        for key in sums:
            sums[key] += r[key + "_tok"]
    return sums


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p / 100), len(ordered) - 1)]


def summarize(cell, records, ctx):
    """The measured operating point of one cell, plus its earnings."""
    steady = steady_records(records, cell)
    sums = token_sums(steady)
    summary = {"k": cell["k"], "c_target": cell["c"], "n": len(steady),
               "ok": bool(steady and sums["out"])}
    if not summary["ok"]:
        return summary

    seconds = cell["duration"]
    R = sums["prompt"] / sums["out"]
    c = sums["cached"] / sums["prompt"]
    D = sums["out"] / seconds
    P = sums["fresh"] / seconds
    C = sums["cached"] / seconds
    B = blended_price(R, c)
    ttft_p95 = percentile([r["ttft"] for r in steady], 95)
    itl_p95 = percentile([r["itl"] for r in steady if r["itl"] > 0], 95)
    usd_per_hr = (P_IN * P + P_CACHE * C + P_OUT * D) * 3600 / 1e6

    summary.update(
        R=R, c_meas=c, D=D, P=P, C=C, B=B,
        ttft_p95=ttft_p95, itl_p95=itl_p95,
        slo_ok=ttft_p95 <= ctx["slo_ttft"] and itl_p95 <= ctx["slo_itl"],
        usd_per_hr=usd_per_hr,  # earn rate while the node is serving
        per_mw_yr_u1=usd_per_hr * 8760 / (NODE_KW / 1000),
        per_mw_yr_u=ctx["u_report"] * usd_per_hr * 8760 / (NODE_KW / 1000),
    )
    return summary


async def run_cell(cell, ctx):
    """Drive one grid cell; returns (summary, records)."""
    records = []
    async with aiohttp.ClientSession() as http:
        cell["start"] = time.monotonic()
        await asyncio.gather(*[
            run_worker(w, cell, ctx, records, http)
            for w in range(cell["k"])])
    return summarize(cell, records, ctx), records


# ---- CLI -------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--ks", default="8,16,32,64")
    ap.add_argument("--cshares", default="0.0,0.5,0.9,0.95,0.99")
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--O", type=int, default=336, help="target out tokens/turn")
    ap.add_argument("--seed-tok", type=int, default=300)
    ap.add_argument("--note", default="Reply in roughly 300 words.")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--duration-floor", type=int, default=180)
    ap.add_argument("--turn-est", type=float, default=12.0,
                    help="est. seconds/turn; sizes cell durations")
    ap.add_argument("--slo-ttft", type=float, default=3.0)
    ap.add_argument("--slo-itl", type=float, default=0.08)
    ap.add_argument("--u-report", type=float, default=0.7)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="results.json")
    return ap.parse_args()


def make_context(args):
    return {
        "url": args.url.rstrip("/"),
        "corpus": Corpus(args.corpus),
        "seed_tok": args.seed_tok, "note": args.note,
        "max_tokens": args.max_tokens, "timeout": args.timeout,
        "slo_ttft": args.slo_ttft, "slo_itl": args.slo_itl,
        "u_report": args.u_report,
    }


def plan_cells(args):
    """One cell per c column; duration scales with session depth."""
    cells = []
    for c in (float(x) for x in args.cshares.split(",")):
        turns, fresh = cell_shape(c, args.R, args.O, args.seed_tok)
        duration = max(args.duration_floor, int(2.2 * turns * args.turn_est))
        cells.append({"c": c, "turns": turns, "fresh": fresh,
                      "warmup": args.warmup, "duration": duration})
    return cells


def print_plan(ks, cells, ctx):
    total = sum(c["warmup"] + c["duration"] for c in cells) * len(ks)
    print(f"grid: {len(ks)}x{len(cells)} cells, ~{total / 3600:.1f}h total; "
          f"node {NODE_KW:g} kW, prices ${P_IN}/${P_CACHE}/${P_OUT} per Mtok, "
          f"earnings reported at u={ctx['u_report']}")
    for c in cells:
        print(f"  c={c['c']:<4} turns={c['turns']:>3} "
              f"fresh/turn={c['fresh']:>6,} duration={c['duration']}s")


def print_summary(summary, u):
    if not summary["ok"]:
        print(f"K={summary['k']:>3} c_t={summary['c_target']:<4} | "
              f"no steady requests (n={summary['n']})")
        return
    print(f"K={summary['k']:>3} c_t={summary['c_target']:<4} | "
          f"c_m={summary['c_meas']:.3f} R={summary['R']:.1f} | "
          f"D={summary['D']:>6,.0f} P={summary['P']:>6,.0f} "
          f"C={summary['C']:>7,.0f} | B=${summary['B']:.2f} | "
          f"TTFT95={summary['ttft_p95']:.1f}s "
          f"ITL95={summary['itl_p95'] * 1e3:.0f}ms | "
          f"earn ${summary['usd_per_hr']:.2f}/hr serving -> "
          f"${summary['per_mw_yr_u'] / 1e6:,.1f}M/MW-yr @u{u} | "
          f"{'SLO ok' if summary['slo_ok'] else 'SLO fail'}")


def print_frontier(summaries):
    print("\nfrontier (max SLO-feasible D per c):")
    for c in sorted({s["c_target"] for s in summaries}):
        feasible = [s for s in summaries
                    if s["c_target"] == c and s.get("slo_ok")]
        if feasible:
            best = max(feasible, key=lambda s: s["D"])
            print(f"  c={c:<4} -> D={best['D']:,.0f} out tok/s at K={best['k']} "
                  f"(P={best['P']:,.0f}, B=${best['B']:.2f}, "
                  f"earns ${best['per_mw_yr_u'] / 1e6:,.1f}M/MW-yr)")
        else:
            print(f"  c={c:<4} -> no SLO-feasible cell")


def save_results(path, args, summaries, records):
    with open(path, "w") as f:
        json.dump({"config": vars(args),
                   "prices": {"p_in": P_IN, "p_cache": P_CACHE,
                              "p_out": P_OUT, "node_kw": NODE_KW},
                   "cells": summaries, "per_request": records}, f, indent=1)


async def main():
    args = parse_args()
    ctx = make_context(args)
    ks = [int(x) for x in args.ks.split(",")]
    cells = plan_cells(args)
    print_plan(ks, cells, ctx)

    summaries, records = [], []
    for k in ks:
        for base in cells:
            cell = {**base, "k": k}
            print(f"\n== K={k} c_target={cell['c']} "
                  f"({cell['duration']}s + {cell['warmup']}s warmup) ==",
                  flush=True)
            summary, cell_records = await run_cell(cell, ctx)
            summaries.append(summary)
            records += cell_records
            print_summary(summary, ctx["u_report"])
            save_results(args.out, args, summaries, records)
    print_frontier(summaries)


if __name__ == "__main__":
    asyncio.run(main())
