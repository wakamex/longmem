#!/usr/bin/env python3
"""Compare endchat runs that differ in the tool a turn reads original messages with: zoom, or the
search and sch commands of eval_endchat_tools.py.

    python3 compare_tools.py

For each run: how many turns called the tool, calls per turn, share of answer-bearing turns in
what the tool returned, judged accuracy, and tokens per turn priced at Claude Opus 5.5 rates.
"""
import json, statistics as S
from pathlib import Path

from recall_compare import evidence

# dollars per million tokens; a cache read also pays the write premium of the call that stored it
INPUT, CACHE_READ, CACHE_WRITE_PREMIUM, OUTPUT = 4.0, 0.20, 1.0, 20.0


def rows(path):
    return {json.loads(l)["question_id"]: json.loads(l) for l in open(path)}


def cost(usage):
    cached = sum(u["cached_input_tokens"] for u in usage)
    fresh = sum(u["input_tokens"] for u in usage) - cached
    out = sum(u["output_tokens"] for u in usage)
    return (fresh * INPUT + cached * (CACHE_READ + CACHE_WRITE_PREMIUM) + out * OUTPUT) / 1e6


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
    runs = {
        "zoom": (zoom_run(), "results/hypotheses_endchat_multi133b_j38.eval.jsonl"),
        "search, zoom off": (rows("results/endchat_stats_search133.jsonl"), "results/hypotheses_endchat_search133.eval.jsonl"),
        "sch, zoom off": (rows("results/endchat_stats_sch133.jsonl"), "results/hypotheses_endchat_sch133.eval.jsonl"),
    }
    for name, tag in (("search with whole messages, zoom on", "searchzoom133"), ("search with excerpts, zoom on", "excerptzoom133")):
        if Path(f"results/endchat_stats_{tag}.jsonl").exists():
            runs[name] = (rows(f"results/endchat_stats_{tag}.jsonl"), f"results/hypotheses_endchat_{tag}.eval.jsonl")
    common = set.intersection(*(set(r) for r, _ in runs.values()))
    with_ev = [k for k in common if evidence(data[k])]
    base = None
    print(f"{len(common)} questions in every run, {len(with_ev)} with answer-bearing turns")
    for name, (r, eval_path) in runs.items():
        used = [k for k in common if r[k]["tool_calls"]]
        label = {k: v["label"] for k, v in rows(eval_path).items()} if Path(eval_path).exists() else {}
        first = S.mean(r[k]["turn_usage"][0]["input_tokens"] for k in common)
        base = base or first * INPUT / 1e6
        c = S.mean(cost(r[k]["turn_usage"]) for k in common)
        print(f"\n{name}")
        print(f"  turns that called the tool: {len(used)}/{len(common)} ({100 * len(used) / len(common):.0f}%)")
        print(f"  model calls per turn: {S.mean(len(r[k]['turn_usage']) for k in common):.2f}")
        if name != "zoom":
            print(f"  tool calls per turn: {S.mean(len(r[k]['tool_calls']) for k in common):.2f}")
            calls = [c for k in common for c in r[k]["tool_calls"]]
            print(f"  characters returned per tool call: {S.mean(c['chars'] for c in calls):.0f}")
        if any("zoomed" in r[k] for k in common):
            z = [k for k in common if r[k].get("zoomed")]
            print(f"  turns that also opened a message with zoom: {len(z)}/{len(common)}, {S.mean(len(r[k]['zoomed']) for k in z) if z else 0:.1f} messages each")
        print(f"  evidence recall: {S.mean(r[k]['evidence_recall'] for k in with_ev):.3f} over all, {S.mean(r[k]['evidence_recall'] for k in with_ev if k in used):.3f} over turns that called it")
        print(f"  messages returned per turn: {S.mean(r[k]['messages_returned'] for k in with_ev):.1f}")
        if label:
            print(f"  judged correct: {sum(label[k] for k in common)}/{len(common)} ({100 * sum(label[k] for k in common) / len(common):.1f}%)")
        tok = lambda f: S.mean(sum(f(u) for u in r[k]["turn_usage"]) for k in common)
        print(f"  tokens per turn: {tok(lambda u: u['input_tokens'] - u['cached_input_tokens']):.0f} fresh input, {tok(lambda u: u['cached_input_tokens']):.0f} cache read, {tok(lambda u: u['output_tokens']):.0f} output")
        print(f"  cost per turn at Opus 5.5 rates: ${c:.3f} ({100 * (c / base - 1):+.0f}% over the first call's input alone, ${base:.3f})")


if __name__ == "__main__":
    main()
