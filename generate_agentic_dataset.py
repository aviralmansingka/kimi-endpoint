#!/usr/bin/env python3
"""Create NVIDIA AIPerf 0.13.0 native input; stdlib only, no network.

  python generate_agentic_dataset.py --output shared.jsonl
  python generate_agentic_dataset.py --pattern multi-turn-growth --rows 4 \
      --turns 8 --output conversations.jsonl
  python generate_agentic_dataset.py --pattern tool-loop --output tool-replay/

shared-prefix/poisson: single_turn JSONL (text, output_length).
multi-turn-growth: multi_turn JSONL (session_id, turns); AIPerf appends live replies.
fan-out: dag_jsonl JSONL; coordinator forks children inheriting its actual reply.
tool-loop: raw_payload DIRECTORY, one sequential JSONL file per conversation.
The last format is RECORDED replay, not a live tool loop: IDs/results are authored.
bench_agentic.py uses a small live adapter instead unless --replay-tools is set.
Token lengths are chars/4 padding estimates, excluding task/schema instructions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterator

# Native formats, not GuideLLM columns or arbitrary token-count-only records:
# https://github.com/ai-dynamo/aiperf/blob/v0.13.0/docs/tutorials/custom-dataset.md
# https://github.com/ai-dynamo/aiperf/blob/v0.13.0/docs/tutorials/raw-payload-replay.md
# Prefix reuse follows NVIDIA's existing Kimi-K3 recipe (not an MLPerf submission):
# https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3/perf/README.md
# Repair/observation prompts are synthetic, inspired by SWE-agent and Aider:
# https://swe-agent.com/latest/usage/trajectories/
# https://aider.chat/docs/benchmarks-1106.html
PATTERNS = ("shared-prefix", "poisson", "multi-turn-growth", "tool-loop", "fan-out")
MODEL = "nvidia/Kimi-K3-NVFP4"
TOOL = {"type": "function", "function": {
    "name": "lookup_record", "description": "Read a mock repository record by key.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                   "required": ["key"], "additionalProperties": False},
}}


def padding(tokens: int) -> str:
    phrase = "Repository context: inspect code, compare tests, report findings. "
    return (phrase * (tokens * 4 // len(phrase) + 1))[:tokens * 4]


def system_prompt(prefix_tokens: int) -> str:
    return ("You are a repository analysis agent. Shared tool reference: "
            + json.dumps(TOOL, sort_keys=True) + "\n" + padding(prefix_tokens))


def user_prompt(tail_tokens: int, agent: int, turn: int, tool: bool = False) -> str:
    task = (f"Call lookup_record with key='test-{agent}-{turn}' using the native tool API."
            if tool else "A boundary test failed. Extend your analysis and propose a repair.")
    return f"{padding(tail_tokens)}\nAgent {agent}, step {turn}: {task}"


def records(pattern: str, rows: int, turns: int, prefix_tokens: int,
            tail_tokens: int, output_tokens: int = 256, fan_out_width: int = 4) -> Iterator[dict]:
    """Native single_turn, multi_turn or dag_jsonl rows (never fake live history)."""
    prefix = system_prompt(prefix_tokens)
    for agent in range(rows):
        if pattern == "fan-out":
            # v0.13.0 DAG forks inherit the coordinator's LIVE assistant reply.
            # https://github.com/ai-dynamo/aiperf/blob/v0.13.0/src/aiperf/dataset/loader/dag_jsonl_models.py
            children = [f"agent-{agent}-child-{child}" for child in range(fan_out_width)]
            yield {"session_id": f"agent-{agent}", "turns": [{
                "messages": [{"role": "system", "content": prefix},
                             {"role": "user", "content": padding(tail_tokens) +
                              f"\nPropose {fan_out_width} independent code-review subtasks."}],
                "max_tokens": output_tokens, "forks": children}]}
            for child, session_id in enumerate(children):
                yield {"session_id": session_id, "turns": [{
                    "messages": [{"role": "user", "content": user_prompt(tail_tokens, agent, turn) +
                                  f" Carry out subtask {child} from the coordinator plan above."}],
                    "max_tokens": output_tokens} for turn in range(turns)]}
            continue
        prompts = [{"text": user_prompt(tail_tokens, agent, turn), "output_length": output_tokens}
                   for turn in range(turns if pattern == "multi-turn-growth" else 1)]
        if pattern == "multi-turn-growth":
            # A leading system turn is hoisted by AIPerf, not sent as a request.
            yield {"session_id": f"agent-{agent}", "turns": [{"role": "system", "text": prefix}, *prompts]}
        else:
            prompts[0]["text"] = prefix + "\n" + prompts[0]["text"]
            yield prompts[0]


def tool_transcript(agent: int, turns: int, prefix_tokens: int, tail_tokens: int,
                    output_tokens: int, model: str, streaming: bool) -> Iterator[dict]:
    """Recorded raw OpenAI requests; intentionally do NOT claim live ID remapping."""
    history = [{"role": "system", "content": system_prompt(prefix_tokens)}]
    for turn in range(turns):
        history.append({"role": "user", "content": user_prompt(tail_tokens, agent, turn, True)})
        base = {"model": model, "stream": streaming, "max_tokens": output_tokens}
        if streaming:
            base["stream_options"] = {"include_usage": True}
        yield {**base, "messages": list(history), "tools": [TOOL], "tool_choice": "auto"}
        call_id = f"recorded-{agent}-{turn}"
        history.extend([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": "lookup_record", "arguments": json.dumps({"key": f"test-{agent}-{turn}"})}}]},
            {"role": "tool", "tool_call_id": call_id,
             "content": json.dumps({"observation": "Boundary test failed. " + padding(tail_tokens)})},
        ])
        yield {**base, "messages": list(history)}
        history.append({"role": "assistant", "content": "Recorded summary: inspect the boundary condition."})


def write_dataset(path: str, pattern: str, rows: int, turns: int, prefix_tokens: int,
                  tail_tokens: int, output_tokens: int = 256, model: str = MODEL,
                  streaming: bool = True, fan_out_width: int = 4) -> None:
    target = Path(path)
    if pattern == "tool-loop":
        target.mkdir(parents=True, exist_ok=True)
        # Do not leave old agent files in a replay directory when shrinking rows.
        if any(target.iterdir()):
            raise ValueError("Tool replay output directory must be empty")
        for agent in range(rows):
            with (target / f"agent-{agent:05d}.jsonl").open("w", encoding="utf-8") as output:
                for row in tool_transcript(agent, turns, prefix_tokens, tail_tokens, output_tokens, model, streaming):
                    print(json.dumps(row, ensure_ascii=False), file=output)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as output:
            for row in records(pattern, rows, turns, prefix_tokens, tail_tokens, output_tokens, fan_out_width):
                print(json.dumps(row, ensure_ascii=False), file=output)


def self_test() -> None:
    rows = list(records("shared-prefix", 2, 1, 8, 4))
    assert rows[0]["text"].startswith(system_prompt(8))
    assert rows[0]["text"] != rows[1]["text"] and len(padding(8)) == 32
    row = next(records("multi-turn-growth", 1, 3, 0, 0))
    assert len(row["turns"]) == 4 and row["turns"][0]["role"] == "system"
    dag = list(records("fan-out", 1, 1, 0, 0, fan_out_width=2))
    assert len(dag) == 3 and dag[0]["turns"][0]["forks"] == [r["session_id"] for r in dag[1:]]
    transcript = list(tool_transcript(0, 2, 0, 0, 128, MODEL, True))
    assert len(transcript) == 4 and len(transcript[0]["messages"]) == 2
    assert transcript[1]["messages"][-1]["tool_call_id"] == "recorded-0-0"
    print("Dataset self-test passed.", file=sys.stderr)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--pattern", choices=PATTERNS, default="shared-prefix")
    cli.add_argument("--rows", type=int, default=100, help="Requests or independent conversations")
    cli.add_argument("--turns", type=int, default=8, help="User turns/child turns; tool replay creates two requests per turn")
    cli.add_argument("--fan-out-width", type=int, default=4)
    cli.add_argument("--prefix-tokens", type=int, default=2048)
    cli.add_argument("--tail-tokens", type=int, default=128)
    cli.add_argument("--output-tokens", type=int, default=256, help="Natural output cap, not ignore_eos")
    cli.add_argument("--model", default=MODEL, help="Model embedded in raw tool replay only")
    cli.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--output", default="-", help="JSONL file, or empty directory for tool-loop; '-' previews stdout")
    cli.add_argument("--dry-run", action="store_true", help="Print two records (one complete graph for fan-out), no files")
    cli.add_argument("--self-test", action="store_true")
    args = cli.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.rows, args.turns, args.output_tokens, args.fan_out_width) < 1 or min(args.prefix_tokens, args.tail_tokens) < 0:
        cli.error("rows/turns/output-tokens must be positive; padding must be nonnegative")
    if args.output == "-" or args.dry_run:
        iterator = (tool_transcript(0, args.turns, args.prefix_tokens, args.tail_tokens,
                                  args.output_tokens, args.model, args.streaming) if args.pattern == "tool-loop"
                    else records(args.pattern, args.rows, args.turns, args.prefix_tokens, args.tail_tokens, args.output_tokens, args.fan_out_width))
        preview_limit = args.fan_out_width + 1 if args.pattern == "fan-out" else 2
        for index, row in enumerate(iterator):
            if args.dry_run and index >= preview_limit:
                break
            print(json.dumps(row, ensure_ascii=False))
    else:
        try:
            write_dataset(args.output, args.pattern, args.rows, args.turns, args.prefix_tokens,
                          args.tail_tokens, args.output_tokens, args.model, args.streaming, args.fan_out_width)
        except ValueError as exc:
            cli.error(str(exc))


if __name__ == "__main__":
    main()
