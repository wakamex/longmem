#!/usr/bin/env python3
"""Plot the replicated endchat arms as clouds of plausible (cost, questions correct) points.

    uv run --no-project --with matplotlib python plot_tool_arms_cloud.py [OUT]   # default results/tool_arms_cloud.png

Arms with fewer than two completed runs are left out.

Each dot is one bootstrap draw. Accuracy: the arm's runs resampled separately for each of the 133
questions, so the cloud's height is how well the arm's average on these questions is pinned down.
Cost: the arm's runs resampled, each priced cold and warm with compare_tools.py and mixed at a warm
share drawn from its measured interval. Draws for different arms are paired by index, so each
index is one possible world; the legend gives the share of worlds in which the arm is on the
efficient frontier (no other arm cheaper and at least as accurate, or as cheap and more accurate).
"""
import json, random, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_tools as c
from replicates import runs as label_runs

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = {  # tag: (name, color)
    "none133": ("no tool", "#2a78d6"), "zoom133": ("zoom", "#2a78d6"),
    "scout133": ("Haiku scouts with search", "#1baf7a"), "scout_sch": ("Haiku scouts with sch", "#1baf7a"),
    "scout_zoom": ("Haiku scouts with zoom", "#1baf7a"), "reader133": ("search, Haiku reads results", "#1baf7a"),
    "sch133": ("sch", "#eb6834"), "sch_view16k": ("sch, small view", "#eb6834"),
}
DRAWS, N = 3000, 133


def cost_runs(tag):
    """Each run's turns, with the first call's cached input counted as fresh, so that every run is
    priced as if it had started cold (replicates found the cache warm from the run before)."""
    paths = ([] if tag == "zoom133" else [f"results/endchat_stats_{tag}.jsonl"]) + sorted(str(p) for p in Path("results").glob(f"endchat_stats_{tag}_r*.jsonl"))
    out = []
    for p in paths:
        r = c.rows(p)
        if len(r) < N:
            continue
        for v in r.values():
            if v["turn_usage"]:
                u = dict(v["turn_usage"][0]); u["cached_input_tokens"] = 0
                v["turn_usage"] = [u] + v["turn_usage"][1:]
        out.append((c.turn_cost(r, False), c.turn_cost(r, True)))
    return out


def main():
    rng = random.Random(0)
    warm_sd = (c.WARM_HIGH - c.WARM_LOW) / 3.92
    cloud = {}
    arms = {t: v for t, v in ARMS.items() if len(label_runs(t)) >= 2 and len(cost_runs(t)) >= 2}
    for tag in arms:
        labs, costs = label_runs(tag), cost_runs(tag)
        qs, k = list(labs[0]), len(labs)
        pts = []
        for _ in range(DRAWS):
            acc = sum(sum(labs[rng.randrange(k)][q] for _ in range(k)) / k for q in qs)
            w = min(max(rng.gauss(c.WARM, warm_sd), 0), 1)
            pick = [costs[rng.randrange(len(costs))] for _ in costs]
            pts.append((sum(w * warm + (1 - w) * cold for cold, warm in pick) / len(pick), acc))
        cloud[tag] = pts
    front = {t: 0 for t in arms}
    for i in range(DRAWS):
        for t in arms:
            x, y = cloud[t][i]
            if not any((cloud[o][i][0] <= x and cloud[o][i][1] > y) or (cloud[o][i][0] < x and cloud[o][i][1] >= y) for o in arms if o != t):
                front[t] += 1
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for tag, (name, color) in arms.items():
        xs, ys = zip(*cloud[tag])
        ax.scatter(xs, ys, s=3, color=color, alpha=0.06, linewidths=0, zorder=2)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        ax.scatter(mx, my, s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        share = front[tag] / DRAWS
        dy = {"reader133": 16, "sch133": -16, "scout_zoom": -16, "scout_sch": 16}.get(tag, 0)
        ax.annotate(f"{name}\non the frontier in {share:.0%} of draws", (mx, my), xytext=(12, dy), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center")
        print(f"{name:30s} cost ${mx:.4f} correct {my:.1f} frontier {share:.0%} runs {len(label_runs(tag))}")
    ax.set_xlim(0.030, 0.090)
    ax.set_ylim(97, 118)
    ax.set_xlabel("Cost per turn at Claude Opus 5.5 prices, one-hour prompt cache, 92% of turns warm (lower is better)", fontsize=9, color=MUTED)
    ax.set_ylabel(f"Questions correct, of {N}, mean over runs (higher is better)", fontsize=9.5, color=MUTED)
    ax.xaxis.set_major_formatter(lambda x, _: f"${x:.2f}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}  ({100 * y / N:.0f}%)")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.set_title("Reading old messages in a never-ending chat: accuracy against cost", fontsize=12, color=INK, loc="left", pad=22)
    runs = sorted(len(label_runs(t)) for t in arms)
    ax.text(0, 1.03, f"Each cloud is 3,000 bootstrap draws over {runs[0]} to {runs[-1]} runs per arm on 133 multi-session LongMemEval questions, Claude Sonnet 5.5",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    out = sys.argv[1] if len(sys.argv) > 1 else "results/tool_arms_cloud.png"
    ax.text(1, -0.11, time.strftime("updated %Y-%m-%d %H:%M"), transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    main()
