#!/usr/bin/env python3
"""For kept endchat profiles, show how each question's turn used zoom against the evidence sessions.

    python3 analyze_zoom.py --tag wrong31 --workdir /nvme/claude-tmp/lme_wrong31

Each question lands in one class:
  no_zoom          the turn answered from the view alone
  missed_evidence  it zoomed, but no zoom reached a message of an evidence session
  reached_evidence a zoom reached an evidence message and the answer was still wrong
  fixed            the rerun answered correctly (only with a judged .eval.jsonl)
"""
import argparse, json, re, collections
from pathlib import Path

DATE = re.compile(r"(\d{4})/(\d\d)/(\d\d) \(\w+\) (\d\d):(\d\d)")
iso = lambda d: "{}-{}-{}T{}:{}".format(*DATE.match(d).groups())

def evidence_ids(q):
    """Message indexes (in import order) that belong to the question's evidence sessions."""
    order = sorted(zip(q["haystack_dates"], q["haystack_session_ids"], q["haystack_sessions"]), key=lambda p: iso(p[0]))
    ev, i, total = set(), 0, 0
    for _, sid, sess in order:
        for _ in sess:
            if sid in q["answer_session_ids"]: ev.add(i)
            i += 1
    return ev, i

def read_log(profile):
    msgs = []
    for f in sorted((profile / "main").glob("*.jsonl")):
        msgs += [json.loads(l) for l in f.read_text().split("\n") if l.startswith("{")]
    return sorted(msgs, key=lambda m: m["i"])

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    a = ap.parse_args()
    data = {q["question_id"]: q for q in json.loads(a.dataset.read_text())}
    evalp = Path("results") / f"hypotheses_endchat_{a.tag}.eval.jsonl"
    judged = {json.loads(l)["question_id"]: json.loads(l)["label"] for l in evalp.read_text().splitlines()} if evalp.exists() else {}
    rows, classes = [], collections.Counter()
    for prof in sorted(p for p in a.workdir.iterdir() if p.is_dir()):
        q = data.get(prof.name)
        if not q: continue
        ev, n_hist = evidence_ids(q)
        log = read_log(prof)
        turn = [m for m in log if m["i"] >= n_hist]
        zooms = []
        for m in turn:
            if m["kind"] == "tool" and m["text"].startswith("zoom "):
                try: zooms.append(json.loads(m["text"][5:]))
                except ValueError: pass
        reached = sorted({e for z in zooms for e in ev if z["id"] <= e < z["id"] + z["n"] and z["n"] == 1})
        covered = sorted({e for z in zooms for e in ev if z["id"] <= e < z["id"] + z["n"]})
        answer = " ".join(m["text"] for m in turn if m["kind"] == "agent")
        if judged.get(prof.name): cls = "fixed"
        elif not zooms: cls = "no_zoom"
        elif not reached: cls = "missed_evidence"
        else: cls = "reached_evidence"
        classes[cls] += 1
        rows.append((prof.name, cls, len(zooms), len(ev), len(reached), len(covered), q["answer"], answer))
    print(f"{'question':20s} {'class':17s} zooms evid reached covered")
    for r in rows: print(f"{r[0]:20s} {r[1]:17s} {r[2]:5d} {r[3]:4d} {r[4]:7d} {r[5]:7d}  expected={str(r[6])[:40]!r} got={r[7][:60]!r}")
    print()
    for k, v in classes.most_common(): print(f"{k}: {v}")

if __name__ == "__main__":
    main()
