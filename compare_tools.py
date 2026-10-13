#!/usr/bin/env python3
"""Compare endchat runs that differ in how a turn reads original messages: zoom, or the arms of
eval_endchat_tools.py.

    python3 compare_tools.py

For each run: how many turns called the tool, calls per turn, share of answer-bearing turns in
what the tool returned, judged accuracy, and cost per turn.

Cost is the turn model at Claude Opus 5.5 rates plus Claude Haiku 5.5 for the calls it makes on
the turn model's behalf, with the one-hour prompt cache endchat's runs use: input not yet cached
is written at twice the input price and read back at a twentieth of it. Every run here starts
cold. A warm turn is the same run with its first call's input, less TAIL tokens, read from cache.
WARM is the share of my own prompts that found the cache warm (warm_share.py).
"""
import json, statistics as S
from pathlib import Path

from recall_compare import evidence

# dollars per million tokens
OPUS = {"write": 8.0, "read": 0.20, "out": 20.0}
HAIKU = {"write": 0.20, "read": 0.01, "out": 0.50}
WARM, WARM_LOW, WARM_HIGH = 0.917, 0.908, 0.927  # mean and 95% interval
TAIL = 1000  # tokens of a first call that are new on every turn
ARMS = {  # name: results tag
    "no tool": "none133", "search, whole messages": "search133", "search, excerpts": "excerpt133", "sch": "sch133",
    "search, Haiku reads results": "reader133", "Haiku scouts, one main call": "scout133",
    "search whole + zoom": "searchzoom133", "search excerpts + zoom": "excerptzoom133", "sch + zoom": "schzoom133",
    "no tool, small view": "none_view16k", "sch, small view": "sch_view16k",
}


def rows(path):
    return {json.loads(l)["question_id"]: json.loads(l) for l in open(path)}


def cost(usage, rates, warm=False):
    total = 0
    for n, u in enumerate(usage):
        cached = u["cached_input_tokens"]
        fresh = u["input_tokens"] - cached
        if warm and n == 0:
            moved = max(fresh - TAIL, 0)
            fresh, cached = fresh - moved, cached + moved
        total += fresh * rates["write"] + cached * rates["read"] + u["output_tokens"] * rates["out"]
    return total / 1e6


def turn_cost(run, warm):
    """Mean cost per turn of a run, as {question: stats}."""
    return S.mean(cost(v["turn_usage"], OPUS, warm) + cost(v.get("side_usage", []), HAIKU, warm) for v in run.values())


def at(run, share):
    return share * turn_cost(run, True) + (1 - share) * turn_cost(run, False)


def zoom_run():
    recall = rows("results/zoom_recall_multi133b.jsonl")
    out = {}
    for prof in Path("/nvme/claude-tmp/lme_multib").iterdir():
        led = [json.loads(l) for l in (prof / "ledger.jsonl").read_text().splitlines() if l.startswith("{")]
        usage = [c["usage"] for c in led if c["kind"] == "turn" and c.get("usage")]
        r = recall.get(prof.name)
        out[prof.name] = {"turn_usage": usage, "tool_calls": [None] * (len(usage) - 1), "evidence_recall": r and r["recall"], "messages_returned": r and r["n_read"]}
    return out


def main():
    data = {q["question_id"]: q for q in json.loads(Path("data/longmemeval_s.json").read_text())}
    runs = {"zoom": (zoom_run(), "results/hypotheses_endchat_multi133b_j38.eval.jsonl")}
    for name, tag in ARMS.items():
        if Path(f"results/endchat_stats_{tag}.jsonl").exists():
            runs[name] = (rows(f"results/endchat_stats_{tag}.jsonl"), f"results/hypotheses_endchat_{tag}.eval.jsonl")
    common = set.intersection(*(set(r) for r, _ in runs.values()))
    with_ev = [k for k in common if evidence(data[k])]
    print(f"{len(common)} questions in every run, {len(with_ev)} with answer-bearing turns")
    for name, (r, eval_path) in runs.items():
        used = [k for k in common if r[k]["tool_calls"]]
        label = {k: v["label"] for k, v in rows(eval_path).items()} if Path(eval_path).exists() else {}
        print(f"\n{name}")
        print(f"  turns that called the tool: {len(used)}/{len(common)} ({100 * len(used) / len(common):.0f}%)")
        print(f"  model calls per turn: {S.mean(len(r[k]['turn_usage']) for k in common):.2f}")
        calls = [c for k in common for c in r[k]["tool_calls"]]
        if name != "zoom" and calls:
            print(f"  tool calls per turn: {len(calls) / len(common):.2f}")
            print(f"  characters returned per tool call: {S.mean(c['chars'] for c in calls):.0f}")
        if any("zoomed" in r[k] for k in common):
            z = [k for k in common if r[k].get("zoomed")]
            print(f"  turns that also opened a message with zoom: {len(z)}/{len(common)}, {S.mean(len(r[k]['zoomed']) for k in z) if z else 0:.1f} messages each")
        if used:
            print(f"  evidence recall: {S.mean(r[k]['evidence_recall'] for k in with_ev):.3f} over all, {S.mean(r[k]['evidence_recall'] for k in with_ev if k in used):.3f} over turns that called it")
        print(f"  messages returned per turn: {S.mean(r[k]['messages_returned'] for k in with_ev):.1f}")
        if label:
            print(f"  judged correct: {sum(label[k] for k in common)}/{len(common)} ({100 * sum(label[k] for k in common) / len(common):.1f}%)")
        tok = lambda f: S.mean(sum(f(u) for u in r[k]["turn_usage"]) for k in common)
        print(f"  tokens per turn: {tok(lambda u: u['input_tokens'] - u['cached_input_tokens']):.0f} fresh input, {tok(lambda u: u['cached_input_tokens']):.0f} cache read, {tok(lambda u: u['output_tokens']):.0f} output")
        sub = {k: r[k] for k in common}
        print(f"  cost per turn: ${turn_cost(sub, False):.3f} cold, ${turn_cost(sub, True):.3f} warm, ${at(sub, WARM):.3f} at {WARM:.1%} warm (${at(sub, WARM_HIGH):.3f} to ${at(sub, WARM_LOW):.3f})")


if __name__ == "__main__":
    main()
