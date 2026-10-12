#!/usr/bin/env python3
"""Recall of the answer-bearing turns (`has_answer` in the dataset) for retrieval runs, endchat's zoom
calls and any other ranking, over the multi-session questions that have such turns.

    python3 recall_compare.py --zoom-workdir /nvme/claude-tmp/lme_multib [--ranking NAME=FILE ...]

A ranking file holds one JSON object per line: {"question_id": ..., "ranked": [message ids]}, with
ids counted over the dataset's own message order, as the retrieval runs count them.
Retrieval is read at its first N ranked messages for several N. Zoom is the set of messages the turn
opened whole, so it has one budget: the number of messages it opened.
"""
import argparse, json, statistics as S
from pathlib import Path

from analyze_zoom import iso
from zoom_recall import zoomed_ids

BUDGETS = (2, 5, 10, 20, 50)


def evidence(q):
    """Dataset-order indexes of turns marked has_answer."""
    ids, i = set(), 0
    for sess in q["haystack_sessions"]:
        for m in sess:
            if m.get("has_answer"):
                ids.add(i)
            i += 1
    return ids


def import_to_dataset(q):
    """Map an endchat message index (date order) to the dataset's own index."""
    starts, i = [], 0
    for sess in q["haystack_sessions"]:
        starts.append(i)
        i += len(sess)
    order = sorted(range(len(q["haystack_sessions"])), key=lambda s: iso(q["haystack_dates"][s]))
    out = []
    for s in order:
        out += [starts[s] + k for k in range(len(q["haystack_sessions"][s]))]
    return out


def summarize(label, per_q, extra=""):
    r = [x for x in per_q.values()]
    print(f"{label:34s} mean_recall={S.mean(r):.3f} all_evidence={sum(x >= 0.999 for x in r):3d}/{len(r)} none={sum(x == 0 for x in r):3d}{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    ap.add_argument("--zoom-workdir", type=Path)
    ap.add_argument("--retrieval", action="append", default=[], metavar="NAME=eval_FILE")
    ap.add_argument("--ranking", action="append", default=[], metavar="NAME=FILE")
    a = ap.parse_args()
    data = {q["question_id"]: q for q in json.loads(a.dataset.read_text()) if q["question_type"] == "multi-session"}
    ev = {k: evidence(q) for k, q in data.items()}
    ev = {k: v for k, v in ev.items() if v}
    print(f"{len(ev)} multi-session questions with answer-bearing turns, mean {S.mean(len(v) for v in ev.values()):.1f} turns each")

    rankings = {}
    for spec in a.retrieval:
        name, path = spec.split("=", 1)
        rankings[name] = {json.loads(l)["question_id"]: json.loads(l)["topk_ids"] for l in open(path)}
    for spec in a.ranking:
        name, path = spec.split("=", 1)
        rankings[name] = {json.loads(l)["question_id"]: json.loads(l)["ranked"] for l in open(path)}
    for name, rk in rankings.items():
        for n in BUDGETS:
            per_q = {k: len(set(rk[k][:n]) & ev[k]) / len(ev[k]) for k in ev if k in rk}
            if per_q:
                summarize(f"{name} first {n}", per_q)
        # the same top 50 with a neighbor on each side, as the answering runs read them
        per_q = {}
        for k in ev:
            if k not in rk:
                continue
            got = set()
            for m in rk[k][:50]:
                got |= {m - 1, m, m + 1}
            per_q[k] = len(got & ev[k]) / len(ev[k])
        if per_q:
            summarize(f"{name} first 50 plus 1 each side", per_q)

    if a.zoom_workdir:
        per_q, spent = {}, []
        for k in ev:
            prof = a.zoom_workdir / k
            if not prof.is_dir():
                continue
            q = data[k]
            n_hist = sum(len(s) for s in q["haystack_sessions"])
            to_ds = import_to_dataset(q)
            read = {to_ds[i] for i in zoomed_ids(prof, n_hist) if i < len(to_ds)}
            per_q[k] = len(read & ev[k]) / len(ev[k])
            spent.append(len(read))
        summarize("endchat zoom (messages opened)", per_q, f" mean_messages_opened={S.mean(spent):.1f}")


if __name__ == "__main__":
    main()
