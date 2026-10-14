#!/usr/bin/env python3
"""Accuracy of the endchat arms over their replicate runs on the 133 multi-session questions.

    python3 replicates.py

For each arm: its runs' scores, the mean, and a 95% interval for how the arm does on these
questions, from resampling its runs within each question. Then each pair of arms next to each
other in score: the difference, its interval, and whether it is settled: the interval leaves out
zero, or lies within MARGIN questions either way.
"""
import json, random, statistics as S
from pathlib import Path

from replicate_arms import ARMS, SIZE

FIRST = {arm: f"results/hypotheses_endchat_{arm}.eval.jsonl" for arm in ARMS if arm not in ("zoom133", "scout_sch", "scout_zoom", "sch_noview", "sch_noview_reworded")}  # run 1, made under the same settings
MARGIN, DRAWS = 3, 5000


def runs(arm):
    paths = [FIRST[arm]] if arm in FIRST else []
    paths += sorted(str(p) for p in Path("results").glob(f"hypotheses_endchat_{arm}_r*.eval.jsonl"))
    out = []
    for p in paths:
        lab = {json.loads(l)["question_id"]: json.loads(l)["label"] for l in open(p)}
        if len(lab) == SIZE[arm]:
            out.append(lab)
    return out


def draws(r, rng):
    """Totals from resampling the arm's runs separately for each question."""
    qs, k = list(r[0]), len(r)
    return [sum(S.mean(r[rng.randrange(k)][q] for _ in range(k)) for q in qs) for _ in range(DRAWS)]


def span(x):
    x = sorted(x)
    return x[int(0.025 * len(x))], x[int(0.975 * len(x))]


def main():
    rng = random.Random(0)
    arms = {a: runs(a) for a in ARMS}
    arms = {a: r for a, r in arms.items() if r}
    mean = {a: S.mean(sum(run.values()) for run in r) for a, r in arms.items()}
    boot = {a: draws(r, rng) for a, r in arms.items()}
    for size in sorted({SIZE[a] for a in arms}, reverse=True):  # arms are compared only on the same questions
        order = sorted((a for a in arms if SIZE[a] == size), key=mean.get)
        print(f"\n{size} questions")
        for a in order:
            lo, hi = span(boot[a])
            scores = [sum(run.values()) for run in arms[a]]
            print(f"{a:20s} {len(scores):2d} runs  mean {mean[a]:6.1f}  95% {lo:6.1f} to {hi:6.1f}  runs: {' '.join(map(str, scores))}")
        print()
        for a, b in zip(order, order[1:]):
            d = [y - x for x, y in zip(boot[a], boot[b])]
            lo, hi = span(d)
            state = "differ" if lo > 0 or hi < 0 else "equivalent" if -MARGIN <= lo and hi <= MARGIN else "open"
            print(f"{b} minus {a}: {mean[b] - mean[a]:+.1f}  95% {lo:+.1f} to {hi:+.1f}  {state}")

if __name__ == "__main__":
    main()
