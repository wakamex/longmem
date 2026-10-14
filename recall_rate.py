#!/usr/bin/env python3
"""How often does a turn of mine reach back into past sessions? From Claude Code and Codex logs.

    uv run --no-project --with arch --with numpy python recall_rate.py [--days 30] [--exclude PROJECT ...]

A turn is one prompt I typed and everything the agent did until my next prompt. It counts as a
recall turn when the agent searched or read session history in it: ran sch (search-clanker-history) as
a command, or used a shell command, Grep, Glob or Read on a session log under ~/.claude/projects or
~/.codex, not counting a project's memory folder or a session's own saved tool output. That is the closest thing in my real usage to a turn of a never-ending chat needing its
search tool. Interval: 95%, stationary block bootstrap over turns in time order, with a data-driven
block length.

Sessions with fewer than two prompts are left out as scripted runs, and so are sessions of the
endchat evaluation. Each log file is parsed once and its turns kept in a cache, refreshed only when
the file grows, so later runs read just the new sessions.
"""
import argparse, glob, json, os, re, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "recall-rate" / "turns.json"
# sch run as a command: at the start of a command, after a separator, or inside $( )
SCH = re.compile(r"(?:^|[;&|\n]|\$\()\s*(?:\S*/)?(?:sch|search-clanker-history|search-claude-history|search-codex-history)\s+\S")
# a path into session history; a project's memory folder and a session's own spilled tool output are not history
PATH = re.compile(r"[^\s'\"]*(?:\.claude/projects|\.codex/(?:sessions|archived_sessions)|\.(?:claude|codex)/history\.jsonl)[^\s'\"]*")
NOT_HISTORY = ("/memory", "/tool-results/")
READERS = {"Bash": "command", "Read": "file_path", "Grep": "path", "Glob": "path"}
SKIP = ("<command-name>", "<local-command", "<task-notification", "Caveat:", "<bash-", "<user-memory-input>", "[Request interrupted")
CODEX_INJECTED = ("# AGENTS.md instructions", "<environment_context>", "<user_instructions>", "<turn_aborted>", "<skill")


def kind(text):
    """'sch', 'path' or None for the command or path of one tool call."""
    if SCH.search(text):
        return "sch"
    if any(not any(x in m for x in NOT_HISTORY) for m in PATH.findall(text)):
        return "path"
    return None


def stamp(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def claude_events(path):
    """(time, 'u') for prompts I typed, (time, kind) for history access, in file order."""
    for line in open(path, errors="replace"):
        if '"type":"user"' not in line and '"tool_use"' not in line:
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("isSidechain") or "timestamp" not in o:
            continue
        c = o.get("message", {}).get("content")
        if o.get("type") == "user" and not o.get("isMeta"):
            if isinstance(c, list):
                if any(b.get("type") == "tool_result" for b in c):
                    continue
                c = "\n".join(b.get("text", "") for b in c if b.get("type") == "text")
            t = c.lstrip() if isinstance(c, str) else ""
            if t and not t.startswith(SKIP) and "[SYSTEM NOTIFICATION - NOT USER INPUT]" not in t:
                yield stamp(o["timestamp"]), "u"
        elif o.get("type") == "assistant" and isinstance(c, list):
            for b in c:
                if b.get("type") == "tool_use" and b.get("name") in READERS:
                    inp = b.get("input") or {}
                    k = kind(" ".join(str(inp.get(f, "")) for f in (READERS[b["name"]], "pattern")))
                    if k:
                        yield stamp(o["timestamp"]), k


def codex_events(path):
    for line in open(path, errors="replace"):
        if '"role":"user"' not in line and "tool_call" not in line and "local_shell_call" not in line:
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        p = o.get("payload") or {}
        if o.get("type") != "response_item" or "timestamp" not in o:
            continue
        if p.get("type") == "message" and p.get("role") == "user":
            t = "\n".join(b.get("text", "") for b in p.get("content", []) if isinstance(b, dict)).lstrip()
            if t and not t.startswith(CODEX_INJECTED):
                yield stamp(o["timestamp"]), "u"
        elif p.get("type") in ("function_call", "custom_tool_call", "local_shell_call"):
            raw = p.get("arguments") or p.get("input") or p.get("action") or ""
            k = kind(raw if isinstance(raw, str) else json.dumps(raw))
            if k:
                yield stamp(o["timestamp"]), k


def turns(item):
    """[time, kinds] per prompt of one log file; kinds is what history access the turn made."""
    tool, path = item
    out = []
    for t, k in (claude_events if tool == "claude" else codex_events)(path):
        if k == "u":
            out.append([t, ""])
        elif out and k not in out[-1][1]:
            out[-1][1] += k[0]  # s for sch, p for a path
    return out


def project(tool, path):
    if tool == "claude":
        return Path(path).parent.name
    return "codex"


def load(days):
    files = [("claude", f) for f in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))]
    files += [("codex", f) for f in glob.glob(os.path.expanduser("~/.codex/sessions/**/*.jsonl"), recursive=True)]
    files += [("codex", f) for f in glob.glob(os.path.expanduser("~/.codex/archived_sessions/*.jsonl"))]
    cut = time.time() - days * 86400
    files = [(t, f) for t, f in files if os.path.getmtime(f) >= cut and "lme" not in Path(f).parent.name and "claude-tmp" not in Path(f).parent.name]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    stale = [(t, f) for t, f in files if cache.get(f, {}).get("size") != os.path.getsize(f)]
    with ProcessPoolExecutor() as ex:
        for (t, f), rows in zip(stale, ex.map(turns, stale, chunksize=4)):
            cache[f] = {"size": os.path.getsize(f), "tool": t, "turns": rows}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    rows = []
    for t, f in files:
        r = cache[f]["turns"]
        if len(r) >= 2:
            rows += [(ts, kinds, t, project(t, f)) for ts, kinds in r if ts >= cut]
    return sorted(rows), len(stale), len(files)


def interval(x):
    import numpy as np
    from arch.bootstrap import StationaryBootstrap, optimal_block_length
    x = np.asarray(x, float)
    block = max(float(optimal_block_length(x)["stationary"].iloc[0]), 1)
    lo, hi = StationaryBootstrap(block, x, seed=0).conf_int(np.mean, 2000, method="percentile").ravel()
    return lo, hi


def report(name, rows):
    x = [float(bool(k)) for _, k, _, _ in rows]
    if not x:
        return
    lo, hi = interval(x)
    n = sum(x)
    sch, path = sum("s" in k for _, k, _, _ in rows), sum("p" in k and "s" not in k for _, k, _, _ in rows)
    print(f"{name}: {int(n)} of {len(x)} turns reached into past sessions, {n / len(x):.2%} (95% {lo:.2%} to {hi:.2%}), "
          f"one in {len(x) / n:.0f} | by sch {sch}, by reading history files only {path}" if n else f"{name}: none of {len(x)} turns")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--exclude", nargs="*", default=["-code-sch", "-code-clanker-analytics", "-code-endchat", "-code-longmem"],
                    help="Claude Code projects to leave out of the second line, by folder name (default: projects that build or study history search)")
    a = ap.parse_args()
    t0 = time.time()
    rows, parsed, total = load(a.days)
    print(f"last {a.days} days: {total} session files, {parsed} parsed now, {time.time() - t0:.0f} s")
    report("all", rows)
    report(f"leaving out {', '.join(a.exclude)}", [r for r in rows if r[3] not in a.exclude])
    for tool in ("claude", "codex"):
        report(tool, [r for r in rows if r[2] == tool])
    by = {}
    for _, k, _, p in rows:
        by.setdefault(p, [0, 0])
        by[p][0] += 1
        by[p][1] += bool(k)
    print("projects with the most recall turns:", ", ".join(f"{p} {s}/{n}" for p, (n, s) in sorted(by.items(), key=lambda kv: -kv[1][1])[:8]))


if __name__ == "__main__":
    main()
