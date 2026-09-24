#!/usr/bin/env python3
"""Local-only, stdlib trace profiler. Never exports message text or tool arguments.

A turn is one model response (including tool-only responses), not one user prompt.
Pi/Claude use message usage; Codex uses changing token_count cumulative checkpoints.
Text component sizes use chars/4; full input/output/cache counts use provider usage.
Run: python3 analysis_profile_traces.py --end 2026-09-24T20:10:00Z
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

UTC = timezone.utc


def stamp(value):
    if isinstance(value, (int, float)):
        return value / 1000 if value > 1e11 else value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError:
            pass
    return None


def text(value):
    """Exclude image payloads, signatures and hidden/encrypted reasoning blobs."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(text(v) for v in value)
    if isinstance(value, dict):
        if value.get('type') in ('image', 'input_image', 'image_url'):
            return ''
        for key in ('text', 'thinking', 'content'):
            if key in value:
                return text(value[key])
    return ''


def estimate(value):
    return len(text(value)) / 4


def args_size(value):
    return len(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) / 4


def dist(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {'n': 0}
    # Nearest-rank quantiles, retaining zero-size additions and cold-cache turns.
    return {'n': len(values), 'p50': values[math.ceil(.5 * len(values)) - 1],
            'mean': statistics.mean(values), 'p90': values[math.ceil(.9 * len(values)) - 1],
            'p99': values[math.ceil(.99 * len(values)) - 1], 'max': values[-1]}


def normalize_usage(u, src):
    if not u:
        return None
    if src == 'pi':
        inp = sum(u.get(k, 0) or 0 for k in ('input', 'cacheRead', 'cacheWrite'))
        out, cache, reasoning = u.get('output', 0), u.get('cacheRead', 0), u.get('reasoning')
    elif src == 'claude':
        inp = sum(u.get(k, 0) or 0 for k in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'))
        out, cache = u.get('output_tokens', 0), u.get('cache_read_input_tokens', 0)
        reasoning = (u.get('output_tokens_details') or {}).get('thinking_tokens')
    else:
        # Codex input_tokens ALREADY includes cached_input_tokens.
        inp, out, cache = u.get('input_tokens', 0), u.get('output_tokens', 0), u.get('cached_input_tokens', 0)
        reasoning = u.get('reasoning_output_tokens')
    if not inp and not out:
        return None  # Synthetic authentication failures / pre-request aborts.
    return dict(input=inp or 0, output=out or 0, cached=cache or 0, reasoning=reasoning)


def parse_file(path, src, begin, end, audit):
    session = dict(source=src, id=hashlib.sha256(str(path).encode()).hexdigest()[:16],
                   file=str(path), turns=[], system_tokens=0, first_user_tokens=0,
                   system_recorded=False, compactions=0, user_messages=0, user_prompts=0, custom_messages=0,
                   tool_calls=0, subagent_calls=0, parallel_batches=[], assistant_text_items=0,
                   cumulative_input=0, cumulative_output=0, checkpoint_gaps=0)
    pending = Counter(user=0., tool_args=0., tool_outputs=0., other=0.)
    current = None
    seen_usage = None
    seen_ids = {}
    first_user_seen = False
    prior_response = False
    started = False
    in_window = False

    def new_turn(ts, identifier=None):
        nonlocal pending, started
        turn = dict(timestamp=ts, start=None, response_id=identifier, new_input=dict(pending),
                    visible_output=0., text_output=0., thinking_text=0., usage=None,
                    tool_names=[], subagents=0)
        if not started:
            session['initial_text_tokens'] = session['system_tokens'] + sum(pending.values())
            turn['new_input'] = None  # Initial input is the prefix, not a subsequent tail.
        pending = Counter(user=0., tool_args=0., tool_outputs=0., other=0.)
        started = True
        session['turns'].append(turn)
        return turn

    def tool_call(name, arguments, turn):
        size = args_size(arguments)
        pending['tool_args'] += size
        turn['visible_output'] += size
        turn['tool_names'].append(name)
        session['tool_calls'] += 1
        # Count native fan-out evidence, not mere mentions in prompts/tool output.
        if name in ('subagent', 'spawn_agent', 'Agent', 'Task') or name.endswith('.subagent'):
            turn['subagents'] += 1
            session['subagent_calls'] += 1
        if name == 'multi_tool_use.parallel':
            try:
                obj = json.loads(arguments) if isinstance(arguments, str) else arguments
                calls = obj.get('tool_uses', [])
                session['parallel_batches'].append(len(calls))
                count = sum(c.get('recipient_name', '').split('.')[-1] in ('subagent', 'spawn_agent') for c in calls)
                turn['subagents'] += count
                session['subagent_calls'] += count
            except (ValueError, AttributeError):
                audit['unparsed_parallel_calls'] += 1

    with path.open(errors='replace') as handle:
        for line in handle:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                audit['malformed_lines'] += 1
                continue
            ts = stamp(r.get('timestamp'))
            if ts is not None and ts > end:
                continue
            recent = ts is not None and begin <= ts <= end
            in_window |= recent
            typ, p = r.get('type'), r.get('payload') or {}
            if typ == 'session_meta':
                base = p.get('base_instructions') or ''
                if isinstance(base, dict):
                    base = base.get('text', '')
                session['system_tokens'] = estimate(base)
                session['system_recorded'] = bool(base)
            if typ in ('session', 'session_meta'):
                session['created'] = ts
            if typ in ('compaction', 'compacted') and ts and ts >= begin:
                session['compactions'] += 1
            if typ == 'custom_message':
                if ts and ts >= begin:
                    session['custom_messages'] += 1
                pending['other'] += estimate(r.get('content'))
                continue
            if src == 'codex':
                if typ == 'event_msg' and p.get('type') == 'task_started' and ts and ts >= begin:
                    session['user_prompts'] += 1
                if typ == 'event_msg' and p.get('type') == 'token_count':
                    info = p.get('info') or {}
                    total = info.get('total_token_usage')
                    if not total or total == seen_usage:
                        if recent:
                            audit['codex_repeated_or_empty_token_count'] += 1
                        continue
                    u = normalize_usage(info.get('last_token_usage'), src)
                    if not u:
                        continue
                    if ts and begin <= ts <= end:
                        for key in ('input', 'output'):
                            delta = total.get(key + '_tokens', 0) - (seen_usage or {}).get(key + '_tokens', 0)
                            # A resumed/forked trace may begin with historical counters.
                            if seen_usage is None or delta < 0:
                                delta = u[key]
                            session['cumulative_' + key] += delta
                            if delta != u[key]:
                                session['checkpoint_gaps'] += 1
                    seen_usage = total
                    if current is None:
                        current = new_turn(ts)
                    current['usage'] = u
                    current['timestamp'] = ts
                    current = None
                    continue
                if typ != 'response_item':
                    continue  # token_usage_record duplicates token_count; do not bill twice.
                kind, role = p.get('type'), p.get('role')
                if kind == 'message' and role in ('system', 'developer', 'user'):
                    size = estimate(p.get('content'))
                    if role in ('system', 'developer'):
                        if not started:
                            session['system_tokens'] += size
                            session['system_recorded'] = True
                        else:
                            pending['other'] += size
                    else:
                        pending['user'] += size
                        if not first_user_seen:
                            session['first_user_tokens'] = size
                            first_user_seen = True
                        if ts and ts >= begin:
                            session['user_messages'] += 1
                    prior_response = False
                elif kind in ('function_call_output', 'custom_tool_call_output'):
                    value = p.get('output', '')
                    pending['tool_outputs'] += estimate(value) if isinstance(value, (str, list)) else args_size(value)
                    prior_response = False
                elif (kind == 'message' and role == 'assistant') or kind in ('function_call', 'custom_tool_call', 'reasoning'):
                    # token_count normally closes each response. For absent usage, a
                    # new assistant after a tool result/input begins another response.
                    if current is None or (not prior_response and current.get('has_output')):
                        current = new_turn(ts)
                    current['has_output'] = True
                    prior_response = True
                    if kind in ('function_call', 'custom_tool_call'):
                        tool_call(p.get('name', ''), p.get('arguments', p.get('input', '')), current)
                    else:
                        size = estimate(p.get('content') or p.get('summary'))
                        current['visible_output'] += size
                        current['thinking_text' if kind == 'reasoning' else 'text_output'] += size
                        if kind == 'message':
                            session['assistant_text_items'] += 1
                continue

            m = r.get('message') or {}
            role = m.get('role')
            if role in ('system', 'developer'):
                value = m.get('content') or '\n'.join(str(v) for v in (m.get('sections') or {}).values())
                if not started:
                    session['system_tokens'] += estimate(value)
                    session['system_recorded'] = True
                else:
                    pending['other'] += estimate(value)
            elif role in ('user', 'toolResult', 'tool'):
                content = m.get('content', '')
                tool_result = role != 'user'
                if isinstance(content, list):
                    for block in content:
                        if block.get('type') == 'tool_result':
                            pending['tool_outputs'] += estimate(block.get('content'))
                            tool_result = True
                        else:
                            pending['tool_outputs' if role != 'user' else 'user'] += estimate(block)
                else:
                    pending['tool_outputs' if tool_result else 'user'] += estimate(content)
                if not tool_result:
                    if not first_user_seen:
                        session['first_user_tokens'] = estimate(content)
                        first_user_seen = True
                    if ts and ts >= begin:
                        session['user_messages'] += 1
                        session['user_prompts'] += 1
            elif role == 'assistant':
                if m.get('model') == '<synthetic>' or (m.get('stopReason') in ('error', 'aborted') and not normalize_usage(m.get('usage'), src)):
                    if recent:
                        audit[src + '_non_model_errors'] += 1
                    continue
                identifier = m.get('responseId') or m.get('id')
                # Claude emits separate thinking/text/tool blocks with the SAME id/usage.
                if identifier and identifier in seen_ids:
                    current = seen_ids[identifier]
                    if recent:
                        audit[src + '_merged_response_records'] += 1
                    if src == 'pi':
                        continue
                else:
                    current = new_turn(ts, identifier)
                    if identifier:
                        seen_ids[identifier] = current
                    if src == 'pi':
                        current['start'] = stamp(m.get('timestamp'))
                current['usage'] = normalize_usage(m.get('usage'), src)
                current['timestamp'] = ts
                content = m.get('content') or []
                if isinstance(content, str):
                    content = [{'type': 'text', 'text': content}]
                for block in content:
                    if block.get('type') in ('toolCall', 'tool_use'):
                        tool_call(block.get('name', ''), block.get('arguments', block.get('input', {})), current)
                    else:
                        size = estimate(block)
                        current['visible_output'] += size
                        current['thinking_text' if block.get('type') == 'thinking' else 'text_output'] += size
                session['assistant_text_items'] += bool(current['text_output'])

    if in_window:
        audit[src + '_files_with_window_records'] += 1
    # Record-level time filtering; pre-window history informs prefix and checkpoint deltas.
    turns = [t for t in session['turns'] if t['timestamp'] is not None and begin <= t['timestamp'] <= end]
    session['left_censored'] = len(turns) != len(session['turns'])
    session['turns'] = turns
    session['prefix_text_lower_bound'] = session['system_tokens'] + session['first_user_tokens']
    if not turns:
        return None
    session['first_request_input'] = (turns[0]['usage'] or {}).get('input')
    if session['left_censored']:
        session['first_request_input'] = None
    return session


def summarize(sessions):
    turns = [t for s in sessions for t in s['turns']]
    real = [t['usage'] for t in turns if t['usage']]
    complete = [s for s in sessions if all(t['usage'] for t in s['turns'])]
    total = lambda s, k: (s['cumulative_' + k] if s['source'] == 'codex' else sum(t['usage'][k] for t in s['turns']))
    tails = [t['new_input'] for t in turns if t['new_input'] is not None]
    return dict(sessions=len(sessions), responses=len(turns), responses_with_usage=len(real),
                one_response_sessions=sum(len(s['turns']) == 1 for s in sessions),
                multi_response_tool_sessions=sum(len(s['turns']) > 1 and s['tool_calls'] > 0 for s in sessions),
                multi_response_no_tool_sessions=sum(len(s['turns']) > 1 and s['tool_calls'] == 0 for s in sessions),
                single_user_prompt_sessions=sum(s['user_prompts'] == 1 for s in sessions),
                system_recorded_sessions=sum(s['system_recorded'] for s in sessions),
                compactions=sum(s['compactions'] for s in sessions),
                subagent_calls=sum(s['subagent_calls'] for s in sessions),
                sessions_with_subagent_calls=sum(s['subagent_calls'] > 0 for s in sessions),
                multi_subagent_responses=sum(t['subagents'] > 1 for t in turns),
                cached_token_weighted_fraction=sum(u['cached'] for u in real) / max(1, sum(u['input'] for u in real)),
                reasoning_share_of_output=sum(u['reasoning'] or 0 for u in real) / max(1, sum(u['output'] for u in real)),
                new_input_component_totals={k: sum(t[k] for t in tails) for k in ('user', 'tool_args', 'tool_outputs', 'other')},
                distributions={
                    'prefix_text_system_plus_first_user': dist(s['prefix_text_lower_bound'] for s in sessions),
                    'initial_recorded_text_context': dist(s.get('initial_text_tokens') for s in sessions),
                    'first_request_input_real': dist(s['first_request_input'] for s in sessions),
                    'turns_per_session': dist(len(s['turns']) for s in sessions),
                    'new_input_estimated': dist(sum(t.values()) for t in tails),
                    'new_input_excluding_args_estimated': dist(sum(v for k, v in t.items() if k != 'tool_args') for t in tails),
                    'tool_output_per_turn_estimated': dist(t['tool_outputs'] for t in tails),
                    'output_real': dist(u['output'] for u in real),
                    'output_real_or_estimated': dist(t['usage']['output'] if t['usage'] else t['visible_output'] for t in turns),
                    'assistant_text_only_estimated': dist(t['text_output'] for t in turns),
                    'input_per_request_real': dist(u['input'] for u in real),
                    'total_session_input_real': dist(total(s, 'input') for s in complete),
                    'total_session_output_real': dist(total(s, 'output') for s in complete),
                    'cached_fraction_per_turn': dist(u['cached'] / u['input'] for u in real if u['input']),
                    'reasoning_fraction_per_turn': dist(u['reasoning'] / u['output'] for u in real if u['reasoning'] is not None and u['output']),
                    'parallel_tool_batch_width': dist(len(t['tool_names']) for t in turns if len(t['tool_names']) > 1),
                    'subagent_calls_per_session_positive': dist(s['subagent_calls'] for s in sessions if s['subagent_calls']),
                    'subagent_calls_per_response_positive': dist(t['subagents'] for t in turns if t['subagents']),
                })


def concurrency(sessions, begin, end):
    hours, minutes = defaultdict(set), defaultdict(set)
    inference = defaultdict(list)
    intervals = []
    for s in sessions:
        times = sorted(t['timestamp'] for t in s['turns'])
        for ts in times:
            hours[int(ts // 3600)].add(s['id'])
            minutes[int(ts // 60)].add(s['id'])
        # Active-session proxy: split at five-minute idle gaps; never bridge days.
        start = last = times[0]
        for ts in times[1:]:
            if ts - last > 300:
                intervals.append((start, last + 1, s['id']))
                start = ts
            last = ts
        intervals.append((start, last + 1, s['id']))
        for t in s['turns']:
            if s['source'] == 'pi' and t['start'] and begin <= t['start'] < t['timestamp']:
                inference[s['id']].append((t['start'], t['timestamp']))

    def overlaps(rows):
        events = defaultdict(int)
        for start, stop, _ in rows:
            events[start] += 1
            events[stop] -= 1
        weighted = Counter()
        count, prev = 0, None
        for ts, change in sorted(events.items()):
            if count and prev is not None:
                weighted[count] += ts - prev
            count += change
            prev = ts
        if not weighted:
            return {'n': 0}
        total = sum(weighted.values())
        def q(p):
            acc = 0
            for n, seconds in sorted(weighted.items()):
                acc += seconds
                if acc >= p * total:
                    return n
        return dict(active_seconds=total, p50=q(.5), mean=sum(k*v for k,v in weighted.items())/total,
                    p90=q(.9), p99=q(.99), max=max(weighted))

    merged = []
    for sid, rows in inference.items():
        for start, stop in sorted(rows):
            if merged and merged[-1][2] == sid and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]), sid)
            else:
                merged.append((start, stop, sid))
    return dict(active_hours=len(hours),
                distinct_sessions_per_active_hour=dist(len(v) for v in hours.values()),
                distinct_sessions_per_all_hours=dist(len(hours[h]) for h in range(int(begin//3600), int(end//3600)+1)),
                distinct_sessions_per_active_minute=dist(len(v) for v in minutes.values()),
                five_minute_idle_split_overlap=overlaps(intervals),
                pi_timestamp_inflight_overlap=overlaps(merged),
                pi_response_seconds=dist(stop-start for rows in inference.values() for start,stop in rows))


def self_test():
    import tempfile
    assert dist([0, 0, 4])['p50'] == 0
    assert normalize_usage(dict(input=2, output=3, cacheRead=8, cacheWrite=4), 'pi')['input'] == 14
    assert normalize_usage(dict(input_tokens=10, output_tokens=3, cached_input_tokens=8), 'codex')['input'] == 10
    assert estimate([dict(type='input_text', text='abcd')]) == 1
    rows = [dict(type='session_meta', timestamp='2026-09-01T00:00:00Z', payload=dict(base_instructions=dict(text='base')))]
    def item(kind, **kw):
        rows.append(dict(type='response_item', timestamp='2026-09-01T00:00:01Z', payload=dict(type=kind, **kw)))
    def usage(i, o, last_i, last_o):
        rows.append(dict(type='event_msg', timestamp='2026-09-01T00:00:02Z', payload=dict(type='token_count', info=dict(total_token_usage=dict(input_tokens=i, output_tokens=o), last_token_usage=dict(input_tokens=last_i, output_tokens=last_o, cached_input_tokens=0)))))
    item('message', role='user', content=[dict(type='input_text', text='test')])
    item('function_call', name='read', arguments='abcd')
    item('function_call_output', output='abcdefgh')
    usage(10, 2, 10, 2)
    usage(10, 2, 10, 2)
    item('message', role='assistant', content=[dict(type='output_text', text='done')])
    usage(30, 5, 20, 3)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)/'test.jsonl'
        p.write_text('\n'.join(json.dumps(r) for r in rows))
        s = parse_file(p, 'codex', stamp('2026-08-01T00:00:00Z'), stamp('2026-10-01T00:00:00Z'), Counter())
        assert len(s['turns']) == 2 and s['cumulative_input'] == 30
        assert sum(s['turns'][1]['new_input'].values()) == 3
        assert s['initial_text_tokens'] == 2
        # Claude's split blocks share a response id and usage, not multiple requests.
        p.write_text('\n'.join(json.dumps(dict(type='assistant', timestamp='2026-09-01T00:00:01Z', message=dict(role='assistant', id='same', content=[dict(type='text', text=x)], usage=dict(input_tokens=10, output_tokens=2)))) for x in ('one', 'two')))
        s = parse_file(p, 'claude', 0, 2e9, Counter())
        assert len(s['turns']) == 1 and s['turns'][0]['text_output'] == 1.5
    print('Self-test passed: zeros, cache normalization, Codex checkpoints/tool-only turns, Claude split records.')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--days', type=int, default=30)
    cli.add_argument('--end', default=datetime.now(UTC).isoformat())
    cli.add_argument('--output', type=Path, default=Path('artifacts/trace-profile'))
    cli.add_argument('--self-test', action='store_true')
    args = cli.parse_args()
    if args.self_test:
        self_test()
        return
    end = stamp(args.end)
    if end is None or args.days < 1:
        cli.error('Provide an ISO timestamp and positive --days')
    begin = end - args.days * 86400
    home = Path.home()
    sources = dict(codex=sorted(set((home/'.codex/sessions/2026').glob('*/*/*.jsonl')) | set((home/'.codex/archived_sessions').glob('*.jsonl'))),
                   claude=sorted((home/'.claude/projects').rglob('*.jsonl')),
                   pi=sorted((home/'.pi/agent/sessions').rglob('*.jsonl')))
    audit = Counter()
    sessions, seen_responses = [], set()
    for src, paths in sources.items():
        audit[src + '_files_discovered'] = len(paths)
        for path in paths:
            # Read ALL files; mtime is not a trustworthy transcript-date filter.
            s = parse_file(path, src, begin, end, audit)
            if s is None:
                continue
            retained = []
            for t in s['turns']:
                rid = (src, t['response_id'])
                if t['response_id'] and rid in seen_responses:
                    audit[src + '_cross_file_duplicate_responses'] += 1
                    continue
                if t['response_id']:
                    seen_responses.add(rid)
                retained.append(t)
            if retained and retained[0] is not s['turns'][0]:
                # The fork replays inherited history, not freshly billed responses.
                # Its first ACTUAL request starts warm with that inherited context.
                s['inherited_prefix'] = True
                s['first_request_input'] = (retained[0]['usage'] or {}).get('input')
                retained[0]['new_input'] = None
            s['turns'] = retained
            s['tool_calls'] = sum(len(t['tool_names']) for t in retained)
            s['subagent_calls'] = sum(t['subagents'] for t in retained)
            if retained:
                sessions.append(s)
    report = dict(window=dict(begin=datetime.fromtimestamp(begin, UTC).isoformat(), end=datetime.fromtimestamp(end, UTC).isoformat(), days=args.days),
                  discovery={src: [str(p) for p in paths] for src,paths in sources.items()}, audit=dict(audit),
                  all=summarize(sessions),
                  agentic=summarize([s for s in sessions if len(s['turns']) > 1 and s['tool_calls']]),
                  by_source={src:summarize([s for s in sessions if s['source']==src]) for src in sources},
                  concurrency=concurrency(sessions, begin, end))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.output/'sessions.jsonl').open('w') as out:
        for s in sessions:
            s['distributions'] = summarize([s])['distributions']
            out.write(json.dumps(s)+'\n')
    lines = [f"Window: {report['window']['begin']} — {report['window']['end']}",
             f"Analyzed {len(sessions)} sessions; " + ', '.join(f"{src}={report['by_source'][src]['sessions']}" for src in sources),
             '| Metric | n | p50 | mean | p90 | p99 | max |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, d in report['all']['distributions'].items():
        lines.append('| '+name+' | '+str(d['n'])+' | '+' | '.join(f"{d[k]:,.3f}" if 'fraction' in name else f"{d[k]:,.0f}" for k in ('p50','mean','p90','p99','max'))+' |' if d['n'] else '| '+name+' | 0 | — | — | — | — | — |')
    (args.output/'distributions.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print('Concurrency:', json.dumps(report['concurrency']))
    print('Audit:', json.dumps(report['audit']))


if __name__ == '__main__':
    main()
