#!/usr/bin/env python3
"""Thin NVIDIA AIPerf 0.13.0 glue for the Kimi-K3 SGLang endpoint.

  python bench_agentic.py --list-patterns
  python bench_agentic.py --pattern shared-prefix --dry-run
  python bench_agentic.py --base-url "$MODAL_URL" --pattern fan-out --turns 1
  python bench_agentic.py --base-url "$MODAL_URL" --pattern tool-loop --agents 2

Install in Python 3.11-3.13: pip install 'aiperf==0.13.0' httpx
The scripts themselves require only Python 3.10+ and stdlib (httpx for live tools).
AIPerf owns scheduling, SSE metrics, percentiles and native DAG fan-out. Only
live tool-loop orchestration is custom: preserve Kimi reasoning/tool calls,
append deterministic mock results using real call IDs, and send the next turn.
--replay-tools uses AIPerf metrics on RECORDED tool history instead of a live loop.
No deployment or implicit readiness probe. --dry-run never accesses the network.
See BENCHMARKS.md for examples, metric/cache gaps and the NVIDIA tool comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid

from generate_agentic_dataset import (
    MODEL, PATTERNS, TOOL, padding, records, self_test as dataset_self_test,
    system_prompt, tool_transcript, user_prompt, write_dataset,
)

# NVIDIA phases GenAI-Perf out in favor of AIPerf; Kimi-K3's official recipe uses it:
# https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
# https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3/perf/README.md
# Native v0.13.0 DAG forks implement the dependency barrier: no custom fan-out client.
# AIPerf does not execute tools or remap authored tool-result IDs to live call IDs:
# https://github.com/ai-dynamo/aiperf/blob/v0.13.0/docs/benchmark-modes/trace-replay.md
DATASET_TYPES = {"shared-prefix": "single_turn", "poisson": "single_turn",
                 "multi-turn-growth": "multi_turn", "fan-out": "dag_jsonl", "tool-loop": "raw_payload"}


def server_root(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1/chat/completions").removesuffix("/v1")


def aiperf_command(args: argparse.Namespace, dataset: str, redact: bool = False) -> list[str]:
    command = [args.aiperf, "profile", "--model", args.model, "--url", server_root(args.base_url),
               "--endpoint-type", "chat", "--endpoint", "/v1/chat/completions",
               "--input-file", dataset, "--custom-dataset-type", DATASET_TYPES[args.pattern],
               "--tokenizer", args.tokenizer, "--use-server-token-count", "--use-legacy-max-tokens",
               "--concurrency", str(args.concurrency), "--random-seed", str(args.seed),
               "--request-timeout-seconds", str(args.timeout), "--dataset-sampling-strategy", "sequential",
               "--artifact-dir", args.artifact_dir, "--export-level", "raw", "--ui", "none", "--no-auto-plot"]
    # Modal routes by this header, not by TCP connection. Group independent shared
    # prefixes on one key; native conversations/DAGs use AIPerf's per-session key.
    if args.pattern in ("shared-prefix", "poisson"):
        command += ["--header", f"Modal-Session-ID:{args.session_id}"]
    else:
        command += ["--session-header", "Modal-Session-ID", "--num-sessions", str(args.agents)]
    if args.streaming:
        command += ["--streaming"]
    if args.api_key:
        command += ["--api-key", "[REDACTED]" if redact else args.api_key]
    if args.cache_report:
        command += ["--extra-inputs", '{"return_cached_tokens_details":true}']
    profile = args.profile or ("poisson" if args.pattern == "poisson" else "concurrent")
    if profile != "concurrent":
        command += ["--request-rate", str(args.arrival_rate), "--arrival-pattern", profile]
    if args.num_requests:
        command += ["--request-count", str(args.num_requests)]
    if args.duration:
        command += ["--benchmark-duration", str(args.duration)]
    return command


def run_aiperf(args: argparse.Namespace) -> int:
    directory = Path(args.artifact_dir)
    dataset = args.dataset or str(directory / ("tool-replay" if args.pattern == "tool-loop" else "input.jsonl"))
    if args.dry_run:
        if not args.dataset:
            iterator = (tool_transcript(0, args.turns, args.prefix_tokens, args.tail_tokens, args.output_tokens,
                                       args.model, args.streaming) if args.pattern == "tool-loop" else
                        records(args.pattern, 1, args.turns, args.prefix_tokens, args.tail_tokens,
                                args.output_tokens, args.fan_out_width))
            # For DAG preview include all children, so references are resolvable.
            for row in iterator:
                print(json.dumps({"dataset_record": row}, ensure_ascii=False))
        print(shlex.join(aiperf_command(args, dataset, redact=True)))
        return 0
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a new/empty --artifact-dir to avoid mixing stale benchmark results")
    directory.mkdir(parents=True, exist_ok=True)
    if not args.dataset:
        count = args.rows if args.pattern in ("shared-prefix", "poisson") else args.agents
        write_dataset(dataset, args.pattern, count, args.turns, args.prefix_tokens, args.tail_tokens,
                      args.output_tokens, args.model, args.streaming, args.fan_out_width)
    log = directory / "runner.log"
    try:
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(aiperf_command(args, dataset), stdout=output, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError:
        print("AIPerf was not found. Install aiperf==0.13.0 in Python 3.11-3.13. Use --aiperf PATH if it is in another venv.", file=sys.stderr)
        return 2
    summary = directory / "profile_export_aiperf.json"
    if args.save_results and summary.exists():
        destination = Path(args.save_results)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if summary.resolve() != destination.resolve():
            shutil.copyfile(summary, destination)
    print(f"AIPerf finished {args.pattern} with process exit code {result.returncode}.")
    if summary.exists():
        metrics = json.loads(summary.read_text())
        def stat(name, field="avg"):
            value = (metrics.get(name) or {}).get(field)
            return f"{value:.2f}" if isinstance(value, (int, float)) else "unavailable"
        print(f"Native results: {stat('request_throughput')} req/s, {stat('output_token_throughput')} output tok/s, "
              f"TTFT p50/p99 {stat('time_to_first_token', 'p50')}/{stat('time_to_first_token', 'p99')} ms, "
              f"request error rate {stat('request_error_rate')}%.")
    print(f"Native metrics, raw records, input and runner.log are in {directory}.")
    print("This is recorded tool-history replay, not live tool execution." if args.pattern == "tool-loop" else
          "Inspect server token usage and cache fields before comparing this synthetic run with production.")
    return result.returncode


async def read_events(response):
    """Small SSE framing helper; no scheduling or metric engine."""
    data = []
    finished = False
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            raw = "\n".join(data)
            data.clear()
            if raw.strip() == "[DONE]":
                return
            event = json.loads(raw)
            finished |= any(c.get("finish_reason") for c in event.get("choices") or [])
            yield event
    if data:
        raw = "\n".join(data)
        if raw.strip() == "[DONE]":
            return
        event = json.loads(raw)
        finished |= any(c.get("finish_reason") for c in event.get("choices") or [])
        yield event
    if not finished:
        raise ValueError("Truncated SSE: no [DONE] or finish_reason")


class ToolParseError(ValueError):
    pass


class ToolReply:
    """Only glue AIPerf lacks: retain Kimi reasoning and real fragmented call IDs."""
    def __init__(self):
        self.message = {"role": "assistant", "content": "", "reasoning_content": ""}
        self.calls = {}

    def feed(self, event: dict, streaming: bool) -> None:
        for choice in event.get("choices") or []:
            delta = choice.get("delta" if streaming else "message") or {}
            self.message["content"] += delta.get("content") or ""
            self.message["reasoning_content"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
            try:
                parts = delta.get("tool_calls") or []
                if not isinstance(parts, list):
                    raise ValueError("tool_calls must be an array")
                for position, part in enumerate(parts):
                    index = part.get("index", position)
                    if type(index) is not int or index < 0:
                        raise ValueError("Invalid call index")
                    call = self.calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if part.get("id"):
                        call["id"] = part["id"]
                    if part.get("type"):
                        call["type"] = part["type"]
                    for field in ("name", "arguments"):
                        call["function"][field] += (part.get("function") or {}).get(field) or ""
            except (ValueError, TypeError, AttributeError) as exc:
                raise ToolParseError(str(exc)) from exc

    def finish(self, expect_tools: bool) -> dict:
        calls = [self.calls[index] for index in sorted(self.calls)]
        if bool(calls) != expect_tools:
            raise ToolParseError("Missing expected tool_calls (possibly refusal/truncation), or unexpected tool call")
        ids = set()
        for call in calls:
            try:
                arguments = json.loads(call["function"]["arguments"])
                valid = (isinstance(call["id"], str) and bool(call["id"]) and call["id"] not in ids
                         and call["type"] == "function" and call["function"]["name"] == "lookup_record"
                         and isinstance(arguments, dict) and set(arguments) == {"key"}
                         and isinstance(arguments["key"], str))
                if not valid:
                    raise ValueError("Invalid call ID, tool name, or argument schema")
                ids.add(call["id"])
            except (ValueError, TypeError) as exc:
                raise ToolParseError(str(exc)) from exc
        if calls:
            self.message["tool_calls"] = calls
        elif not self.message["content"] and not self.message["reasoning_content"]:
            raise ValueError("Empty assistant reply")
        self.message["content"] = self.message["content"] or None
        if not self.message["reasoning_content"]:
            del self.message["reasoning_content"]
        return self.message


async def run_live_tools(args: argparse.Namespace) -> int:
    semaphore = asyncio.Semaphore(args.concurrency)
    started, admitted, results = time.perf_counter(), 0, []
    client = None
    limit = min(args.num_requests or args.dry_run_count, args.dry_run_count) if args.dry_run else args.num_requests

    async def send(agent: int, history: list[dict], expect_tools: bool) -> dict | None:
        nonlocal admitted
        async with semaphore:
            if ((limit and admitted >= limit) or
                    (args.duration and time.perf_counter() - started >= args.duration)):
                return None
            admitted += 1
            payload = {"model": args.model, "stream": args.streaming, "max_tokens": args.output_tokens,
                       "messages": history, "tools": [TOOL], "tool_choice": "auto" if expect_tools else "none"}
            if args.streaming:
                payload["stream_options"] = {"include_usage": True}
            if args.cache_report:
                payload["return_cached_tokens_details"] = True
            headers = {"Modal-Session-ID": f"{args.session_id}-agent-{agent}"}
            if args.dry_run:
                print(json.dumps({"url": server_root(args.base_url) + "/v1/chat/completions", "headers": headers, "payload": payload}))
                return ({"role": "assistant", "content": None, "reasoning_content": "Mock reasoning retained.",
                         "tool_calls": [{"id": f"dry-{admitted}", "type": "function", "function": {
                             "name": "lookup_record", "arguments": '{"key":"test"}'}}]} if expect_tools else
                        {"role": "assistant", "content": "Mock summary."})
            request_started = time.perf_counter()
            record = {"request_id": admitted, "agent": agent, "success": False, "error_category": None,
                      "start_offset_seconds": request_started - started, "events": []}
            reply = ToolReply()
            try:
                async def consume() -> None:
                    async with client.stream("POST", server_root(args.base_url) + "/v1/chat/completions",
                                             json=payload, headers=headers) as response:
                        record["http_status"] = response.status_code
                        if response.is_error:
                            await response.aread()
                            raise ValueError(f"HTTP {response.status_code}: {response.text[:1000]}")
                        if args.streaming:
                            events = read_events(response)
                        else:
                            await response.aread()
                            async def one_event():
                                yield response.json()
                            events = one_event()
                        async for event in events:
                            # Keep raw usage/sglext/extra/meta_info, including cached_tokens.
                            record["events"].append({"elapsed_seconds": time.perf_counter() - request_started, "data": event})
                            if event.get("error"):
                                raise ValueError(json.dumps(event["error"]))
                            reply.feed(event, args.streaming)
                await asyncio.wait_for(consume(), timeout=args.timeout)
                if not record["events"]:
                    raise ValueError("Empty completion")
                message = reply.finish(expect_tools)
                record["success"] = True
            except (httpx.HTTPError, asyncio.TimeoutError, ValueError, TypeError, AttributeError) as exc:
                detail = f"{type(exc).__name__}: {exc}"[:2000]
                parser_error = isinstance(exc, ToolParseError) or ("tool" in detail.lower() and "pars" in detail.lower())
                record.update(error=detail, error_category="tool_call_parse" if parser_error else "request")
                message = None
            record["elapsed_seconds"] = time.perf_counter() - request_started
            results.append(record)
            return message

    async def agent_loop(agent: int) -> None:
        history = [{"role": "system", "content": system_prompt(args.prefix_tokens)}]
        for turn in range(args.turns):
            history.append({"role": "user", "content": user_prompt(args.tail_tokens, agent, turn, True)})
            message = await send(agent, history, True)
            if message is None:
                break
            history.append(message)
            for call in message["tool_calls"]:
                # Never execute model-generated code or external tools.
                history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({
                    "key": json.loads(call["function"]["arguments"])["key"],
                    "observation": "Boundary test failed. " + padding(args.tail_tokens)})})
            message = await send(agent, history, False)
            if message is None:
                break
            history.append(message)

    if args.dry_run:
        await asyncio.gather(*(agent_loop(i) for i in range(args.agents)))
        return 0
    try:
        import httpx
    except ImportError:
        print("Install httpx: python -m pip install httpx", file=sys.stderr)
        return 2
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with httpx.AsyncClient(headers=headers, timeout=args.timeout,
                                limits=httpx.Limits(max_connections=args.concurrency)) as client:
        await asyncio.gather(*(agent_loop(i) for i in range(args.agents)))
    errors = sum(not record["success"] for record in results)
    parse_errors = sum(record["error_category"] == "tool_call_parse" for record in results)
    destination = Path(args.save_results or str(Path(args.artifact_dir) / "live-tool-results.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"pattern": "tool-loop", "requests": len(results), "errors": errors,
                                      "tool_call_parse_failures": parse_errors, "responses": results}, indent=2) + "\n")
    print(f"Live tool-loop sent {len(results)} requests with {errors} errors, including {parse_errors} tool-call parse failures.")
    print(f"Raw event timestamps and server usage are saved in {destination}.")
    print("Use --replay-tools for AIPerf statistics on recorded history; live workflow metrics are not reimplemented here.")
    return 1 if errors else 0


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--pattern", choices=PATTERNS, default="shared-prefix")
    cli.add_argument("--list-patterns", action="store_true")
    cli.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000"))
    cli.add_argument("--model", default=MODEL)
    cli.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="Optional Bearer key; Modal endpoint needs none")
    cli.add_argument("--concurrency", type=int, default=8, help="AIPerf root sessions (DAG children can exceed this); live tools: HTTP cap")
    cli.add_argument("--num-requests", "--request-count", type=int, default=100, help="Wire request/turn cap; 0 disables")
    cli.add_argument("--duration", type=float, default=0, help="Admission window in seconds; 0 disables")
    cli.add_argument("--prefix-tokens", type=int, default=2048)
    cli.add_argument("--tail-tokens", type=int, default=128)
    cli.add_argument("--output-tokens", type=int, default=256, help="Natural cap including reasoning, not forced length")
    cli.add_argument("--agents", "--threads", type=int, default=4, help="Conversation rows, DAG roots, or live tool threads")
    cli.add_argument("--turns", type=int, default=8, help="Conversation turns, fan-out child turns, or two-request tool rounds")
    cli.add_argument("--fan-out-width", type=int, default=4)
    cli.add_argument("--rows", type=int, default=100, help="Independent shared-prefix/Poisson dataset rows")
    cli.add_argument("--arrival-rate", type=float, default=2)
    cli.add_argument("--profile", choices=("constant", "concurrent", "poisson"), help="Native AIPerf arrival profile override")
    cli.add_argument("--tokenizer", default="builtin", help="AIPerf tokenizer; server usage supplies real Kimi token counts")
    cli.add_argument("--dataset", help="Existing native file, or raw replay directory; overrides synthetic payload knobs")
    cli.add_argument("--aiperf", default="aiperf", help="Path to AIPerf 0.13.0 executable")
    cli.add_argument("--timeout", type=float, default=600)
    cli.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--cache-report", action="store_true", help="Request version-dependent SGLang cache details")
    cli.add_argument("--replay-tools", action="store_true", help="Tool-loop: native recorded replay, NOT live tool results")
    cli.add_argument("--session-id", default=f"bench-{uuid.uuid4().hex[:12]}", help="Shared-prefix/live-tool sticky key; native sessions use AIPerf IDs")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--artifact-dir", default=None, help="Default: artifacts/bench-<unique-id>")
    cli.add_argument("--save-results", metavar="PATH.json", help="Copy native summary JSON, or save raw live-tool results")
    cli.add_argument("--dry-run", action="store_true", help="Print native records+command or live tool payloads, without network")
    cli.add_argument("--dry-run-count", type=int, default=4, help="Maximum live-tool preview requests")
    cli.add_argument("--self-test", action="store_true")
    return cli


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    args.artifact_dir = args.artifact_dir or f"artifacts/bench-{uuid.uuid4().hex[:12]}"
    if args.list_patterns:
        for pattern in PATTERNS:
            print(f"{pattern}: {'live HTTP adapter; --replay-tools for AIPerf' if pattern == 'tool-loop' else 'AIPerf ' + DATASET_TYPES[pattern]}")
        return 0
    for name in ("concurrency", "output_tokens", "agents", "turns", "fan_out_width", "rows", "arrival_rate", "timeout", "dry_run_count"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            cli.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("prefix_tokens", "tail_tokens", "num_requests", "duration"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            cli.error(f"--{name.replace('_', '-')} must be nonnegative and finite")
    if not args.base_url.startswith(("http://", "https://")):
        cli.error("--base-url must start with http:// or https://")
    if (not args.num_requests and not args.duration and args.pattern in ("shared-prefix", "poisson")
            and not (args.dry_run or args.self_test)):
        cli.error("Set --num-requests or --duration to bound independent traffic")
    if args.replay_tools and args.pattern != "tool-loop":
        cli.error("--replay-tools only applies to tool-loop")
    if args.pattern == "tool-loop" and not args.replay_tools and (args.dataset or args.profile):
        cli.error("Live tools do not implement rate profiles or dataset replay; use --replay-tools")
    if args.replay_tools and args.cache_report:
        cli.error("Raw replay bypasses extra-inputs; put return_cached_tokens_details in the raw bodies instead")
    if args.self_test:
        dataset_self_test()
        reply = ToolReply()
        reply.feed({"choices": [{"delta": {"reasoning_content": "plan", "tool_calls": [{"index": 0, "id": "x", "function": {"name": "lookup_record", "arguments": '{"ke'}}]}}]}, True)
        reply.feed({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'y":"test"}'}}]}}]}, True)
        assert reply.finish(True)["reasoning_content"] == "plan"
        try:
            ToolReply().finish(True)
            raise AssertionError("Missing call not detected")
        except ToolParseError:
            pass
        assert server_root("http://localhost:8000/v1/chat/completions") == "http://localhost:8000"
        print("Driver self-test passed.")
        return 0
    try:
        return asyncio.run(run_live_tools(args)) if args.pattern == "tool-loop" and not args.replay_tools else run_aiperf(args)
    except (FileExistsError, ValueError) as exc:
        cli.error(str(exc))
    except KeyboardInterrupt:
        print("Interrupted; partial live-tool results were not saved.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
