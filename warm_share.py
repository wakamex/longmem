#!/usr/bin/env python3
"""How often does a prompt of mine find the prompt cache warm? From Claude Code session logs.

    uv run --no-project --with arch --with numpy python warm_share.py [--days 30]

For each prompt I typed, the share of the first reply's input that was read from cache, and the
time since the model last spoke in that session. Intervals are 95%, from a stationary block
bootstrap over the prompts in time order with a data-driven block length, since prompts come in
bursts. compare_tools.py uses the first figure as WARM.

Prompts are the user messages of sessions with at least two of them, leaving out tool results,
subagent turns, slash commands and injected notices. A few scripted prompts remain.
"""
import argparse, glob, json, os, time
from datetime import datetime

import numpy as np
from arch.bootstrap import StationaryBootstrap, optimal_block_length

SKIP = ("<command-name>", "<local-command", "<task-notification", "Caveat:", "<bash-", "<user-memory-input>", "[Request interrupted")


def typed(o):
    """The text of a user message I wrote, or None."""
    if o.get("isMeta") or o.get("isSidechain"):
        return None
    c = o.get("message", {}).get("content")
    if isinstance(c, list):
        if any(b.get("type") == "tool_result" for b in c):
            return None
        c = "\n".join(b.get("text", "") for b in c if b.get("type") == "text")
    if not isinstance(c, str) or not c.strip():
        return None
    t = c.lstrip()
    return None if t.startswith(SKIP) or "[SYSTEM NOTIFICATION - NOT USER INPUT]" in t else t


def prompts(days):
    """(time, seconds since the model last spoke or None, share of the first reply's input read from cache or None)."""
    cut = time.time() - days * 86400
    rows = []
    for f in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")):
        if os.path.getmtime(f) < cut:
            continue
        ev = []
        for line in open(f, errors="replace"):
            if '"timestamp"' not in line or ('"type":"user"' not in line and '"type":"assistant"' not in line):
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("isSidechain") or "timestamp" not in o:
                continue
            t = datetime.fromisoformat(o["timestamp"].replace("Z", "+00:00")).timestamp()
            if t < cut:
                continue
            if o.get("type") == "assistant":
                ev.append((t, "a", o.get("message", {}).get("usage")))
            elif o.get("type") == "user" and typed(o):
                ev.append((t, "u", None))
        ev.sort(key=lambda e: e[0])
        if sum(k == "u" for _, k, _ in ev) < 2:
            continue
        last = None
        for n, (t, k, _) in enumerate(ev):
            if k == "u":
                use = ev[n + 1][2] if n + 1 < len(ev) and ev[n + 1][1] == "a" else None
                read = None
                if use:
                    rd, wr, inp = (use.get(x, 0) or 0 for x in ("cache_read_input_tokens", "cache_creation_input_tokens", "input_tokens"))
                    read = rd / (rd + wr + inp) if rd + wr + inp else None
                rows.append((t, None if last is None else t - last, read))
            last = t
    return sorted(rows, key=lambda r: r[0])


def interval(name, x):
    x = np.asarray(x, float)
    block = max(float(optimal_block_length(x)["stationary"].iloc[0]), 1)
    lo, hi = StationaryBootstrap(block, x, seed=0).conf_int(np.mean, 5000, method="percentile").ravel()
    print(f"{name}: {x.mean():.3f}, 95% interval {lo:.3f} to {hi:.3f} (n={len(x)}, block length {block:.0f} prompts)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30)
    rows = prompts(ap.parse_args().days)
    interval("share of the first reply's input read from cache", [r[2] for r in rows if r[2] is not None])
    interval("prompts within 1 hour of the model's last message", [float(r[1] is not None and r[1] <= 3600) for r in rows])
    interval("prompts within 5 minutes", [float(r[1] is not None and r[1] <= 300) for r in rows])


if __name__ == "__main__":
    main()
