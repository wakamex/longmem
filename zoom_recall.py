#!/usr/bin/env python3
"""Recall of the messages that hold the answer, measured for endchat's zoom calls.

    python3 zoom_recall.py --workdir /nvme/claude-tmp/lme_multib

The evidence set is the same one the retrieval runs score: the turns the dataset marks
`has_answer`. A message counts as read when a turn zoomed it whole, zoom(id, 1). Messages read
is the budget spent, so it is shown beside recall the way retrieval shows precision.
"""
import argparse, json, statistics as S
from pathlib import Path

from analyze_zoom import iso, read_log


def oracle_ids(q):
    """Indexes, in import order, of the turns marked has_answer."""
    order = sorted(zip(q["haystack_dates"], q["haystack_sessions"]), key=lambda p: iso(p[0]))
    ids, i = set(), 0
    for _, sess in order:
        for m in sess:
            if m.get("has_answer"):
                ids.add(i)
            i += 1
    return ids, i


def zoomed_ids(profile, n_hist):
    read = set()
    for m in read_log(profile):
        if m["i"] >= n_hist and m["kind"] == "tool" and m["text"].startswith("zoom "):
            try:
                z = json.loads(m["text"][5:])
            except ValueError:
                continue
            if z["n"] == 1:
                read.add(z["id"])
    return read


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    data = {q["question_id"]: q for q in json.loads(a.dataset.read_text())}
    rows = []
    for prof in sorted(p for p in a.workdir.iterdir() if p.is_dir() and p.name in data):
        ev, n_hist = oracle_ids(data[prof.name])
        if not ev:
            continue
        read = zoomed_ids(prof, n_hist)
        hit = len(read & ev)
        rows.append({"question_id": prof.name, "n_oracle": len(ev), "n_read": len(read), "hit": hit, "recall": hit / len(ev),
                     "precision": hit / len(read) if read else None})
    r = [x["recall"] for x in rows]
    print(f"questions={len(rows)} mean_recall={S.mean(r):.3f} all_evidence={sum(x['recall'] >= 0.999 for x in rows)} "
          f"no_evidence_read={sum(x['hit'] == 0 for x in rows)} mean_messages_read={S.mean(x['n_read'] for x in rows):.1f} "
          f"mean_oracle={S.mean(x['n_oracle'] for x in rows):.1f}")
    if a.out:
        a.out.write_text("".join(json.dumps(x) + "\n" for x in rows))


if __name__ == "__main__":
    main()
