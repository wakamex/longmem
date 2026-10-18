#!/usr/bin/env python3
"""Plot the replicated endchat arms as clouds of plausible (cost, questions correct) points.

    uv run --no-project --with matplotlib python plot_tool_arms_cloud.py [OUT]              # cost of a turn that needs recall
    uv run --no-project --with matplotlib python plot_tool_arms_cloud.py --per-turn [OUT]   # cost of an average turn

Arms with fewer than two completed runs are left out.

Each dot is one bootstrap draw. Accuracy: the arm's runs resampled separately for each of the 133
questions, so the cloud's height is how well the arm's average on these questions is pinned down.
Cost: the arm's runs resampled, each priced cold and warm with compare_tools.py and mixed at a warm
share drawn from its measured interval. Draws for different arms are paired by index, so each
index is one possible world; the legend gives the share of worlds in which the arm is on the
efficient frontier (no other arm cheaper and at least as accurate, or as cheap and more accurate).

With --per-turn, cost is that of an average turn: a share RECALL of turns, drawn from its measured
interval (recall_rate.py), needs the chat's history and costs what the arm's runs cost; the rest
cost a turn that needs none. Such a turn is assumed to make no tool call: one call over the view
(the no-tool arm's runs, or the small-view no-tool run for the small view), plus for a scout the
first Haiku call, in which it reads the view and decides there is nothing to look up.
"""
import json, random, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_tools as c
from replicates import runs as label_runs

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TOOLS = {"none": "#8a8985", "zoom": "#2a78d6", "sch": "#eb6834", "search": "#1baf7a", "full": "#3b3a37"}  # color: the tool that reads old messages
ROLES = {"main": ("o", "the main model reads"), "scout": ("^", "a Haiku scout reads first, the main model answers once"),
         "reader": ("s", "the main model searches, Haiku condenses the results")}  # shape: who reads
VIEWS = {"sch_view16k": "small", "sch_noview": "none"}  # fill: how much of the summary view each turn carries; the rest carry all of it
FILLS = {"full": ("full", "carries the full view of summaries, or the whole history"), "small": ("left", "carries a view cut to an eighth"),
         "none": ("none", "carries no view")}
ARMS = {  # tag: (name, tool, role)
    "none133": ("no tool", "none", "main"), "zoom133": ("zoom", "zoom", "main"),
    "sch133": ("sch", "sch", "main"), "sch_view16k": ("sch, small view", "sch", "main"),
    "reader133": ("search, Haiku reads results", "search", "reader"),
    "scout133": ("Haiku scouts with search", "search", "scout"), "scout_sch": ("Haiku scouts with sch", "sch", "scout"),
    "scout_zoom": ("Haiku scouts with zoom", "zoom", "scout"),
    "sch_noview": ("sch, no view", "sch", "main"),
    "full133": ("whole history in context", "full", "main"),
}
DRAWS, N = 3000, 133
RECALL, RECALL_LOW, RECALL_HIGH = 0.0363, 0.0315, 0.0411  # share of my turns that reached into past sessions, 30 days to 2026-10-14


def cost_runs(tag, first_side_call=False, first_call_only=False):
    """Each run's turns, with the first call's cached input counted as fresh, so that every run is
    priced as if it had started cold (replicates found the cache warm from the run before). With
    first_side_call, only the first of a turn's Haiku calls is counted."""
    first = f"results/endchat_stats_{tag}.jsonl"  # run 1 under the same settings, where it exists (not for zoom, whose run 1 used another harness)
    paths = ([first] if tag != "zoom133" and Path(first).exists() else []) + sorted(str(p) for p in Path("results").glob(f"endchat_stats_{tag}_r*.jsonl"))
    out = []
    for p in paths:
        r = c.rows(p)
        if len(r) < N:
            continue
        for v in r.values():
            if v["turn_usage"]:
                u = dict(v["turn_usage"][0]); u["cached_input_tokens"] = 0
                v["turn_usage"] = [u] + v["turn_usage"][1:]
            if first_call_only:
                v["turn_usage"], v["side_usage"] = v["turn_usage"][:1], []
            if first_side_call:
                v["side_usage"] = v.get("side_usage", [])[:1]
        out.append((c.turn_cost(r, False), c.turn_cost(r, True)))
    return out


def quiet_costs(tag):
    """Cost of a turn of this arm that needs no recall, per run."""
    if tag == "full133":  # every turn is the same single call
        return cost_runs(tag)
    if tag.startswith("scout"):  # the scout's first Haiku call and the main call
        return cost_runs(tag, first_side_call=True)
    if tag == "sch_noview":  # a turn without the view or a tool call: priced from the no-view runs' first calls
        return cost_runs(tag, first_call_only=True)
    return cost_runs("none_view16k" if tag == "sch_view16k" else "none133")


def main():
    per_turn = "--per-turn" in sys.argv
    argv = [x for x in sys.argv[1:] if x != "--per-turn"]
    rng = random.Random(0)
    recall_sd = (RECALL_HIGH - RECALL_LOW) / 3.92
    warm_sd = (c.WARM_HIGH - c.WARM_LOW) / 3.92
    cloud = {}
    arms = {t: v for t, v in ARMS.items() if len(label_runs(t)) >= 2 and len(cost_runs(t)) >= 2}
    for tag in arms:
        labs, costs = label_runs(tag), cost_runs(tag)
        quiet = quiet_costs(tag) if per_turn else None
        qs, k = list(labs[0]), len(labs)
        pts = []
        for _ in range(DRAWS):
            acc = sum(sum(labs[rng.randrange(k)][q] for _ in range(k)) / k for q in qs)
            w = min(max(rng.gauss(c.WARM, warm_sd), 0), 1)
            pick = [costs[rng.randrange(len(costs))] for _ in costs]
            cost = sum(w * warm + (1 - w) * cold for cold, warm in pick) / len(pick)
            if per_turn:
                r = min(max(rng.gauss(RECALL, recall_sd), 0), 1)
                calm = [quiet[rng.randrange(len(quiet))] for _ in quiet]
                cost = r * cost + (1 - r) * sum(w * warm + (1 - w) * cold for cold, warm in calm) / len(calm)
            pts.append((cost, acc))
        cloud[tag] = pts
    front = {t: 0 for t in arms}
    for i in range(DRAWS):
        for t in arms:
            x, y = cloud[t][i]
            if not any((cloud[o][i][0] <= x and cloud[o][i][1] > y) or (cloud[o][i][0] < x and cloud[o][i][1] >= y) for o in arms if o != t):
                front[t] += 1
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for tag, (name, tool, role) in arms.items():
        color, marker = TOOLS[tool], ROLES[role][0]
        xs, ys = zip(*cloud[tag])
        ax.scatter(xs, ys, s=3, color=color, alpha=0.06, linewidths=0, zorder=2)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        fill = FILLS[VIEWS.get(tag, "full")][0]
        if fill == "full":
            ax.scatter(mx, my, s=75, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        else:  # half-filled or hollow, outlined in the series color
            ax.plot(mx, my, marker=marker, markersize=9, fillstyle=fill, color=color, markerfacecoloralt=SURFACE,
                    markeredgecolor=color, markeredgewidth=2, linestyle="none", zorder=3)
        share = front[tag] / DRAWS
        # beside the mark by default; arms whose marks sit close together are labelled on opposite sides or above and below
        # beside the mark by default; marks that sit close together are labelled on opposite sides, or above and below
        dx, dy, ha = {"scout133": (0, 16, "center"), "scout_sch": (0, -16, "center"), "scout_zoom": (0, 16, "center"),
                      "none133": (0, -16, "center"),
                      **({"zoom133": (20, 0, "left")} if per_turn else {"zoom133": (-12, 0, "right"), "reader133": (-12, 0, "right")})}.get(tag, (12, 0, "left"))
        ax.annotate(f"{name} ({share:.0%})", (mx, my), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center", ha=ha)
        print(f"{name:30s} cost ${mx:.4f} correct {my:.1f} frontier {share:.0%} runs {len(label_runs(tag))}")
    # costs span more than a factor of ten, so the axis is logarithmic: equal distances are equal ratios
    xs = sorted(x for pts in cloud.values() for x, _ in pts)
    ax.set_xscale("log")
    right = xs[-1] * (3.2 if per_turn else 1.9)  # room for the legend
    ax.set_xlim(xs[0] / 1.15, right)
    ticks = [t for t in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5) if xs[0] / 1.15 <= t <= right]
    ax.set_xticks(ticks)
    ax.minorticks_off()
    ax.set_ylim(97, 118)
    what = f"Cost of an average turn ({RECALL:.1%} need recall)" if per_turn else "Cost of a turn that needs recall"
    ax.set_xlabel(f"{what}, Opus 5.5 prices, cache warm on 92±1% of turns, log scale (lower is better)", fontsize=9, color=MUTED)
    ax.set_ylabel(f"Recall questions correct, of {N}, mean over runs (higher is better)", fontsize=9.5, color=MUTED)
    ax.xaxis.set_major_formatter(lambda x, _: f"${x:.3f}".rstrip("0").rstrip(".") if x < 0.1 else f"${x:.2f}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}  ({100 * y / N:.0f}%)")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.set_title("Reading old messages in a never-ending chat: accuracy against cost", fontsize=12, color=INK, loc="left", pad=22)
    for tool, color in TOOLS.items():
        ax.scatter([], [], s=55, color=color, marker="o", label={"none": "reads nothing: summaries only", "full": "reads the whole history, no summaries"}.get(tool, f"reads with {tool}"))
    for marker, text in ROLES.values():
        ax.scatter([], [], s=55, color=MUTED, marker=marker, label=text)
    for fill, text in FILLS.values():
        ax.plot([], [], marker="o", markersize=8, fillstyle=fill, color=MUTED, markerfacecoloralt=SURFACE, markeredgecolor=MUTED,
                markeredgewidth=2 if fill != "full" else 0, linestyle="none", label=text)
    ax.scatter([], [], s=0, label="(n%): share of draws in which the arm is on the efficient frontier")
    # in an empty part of the chart: per turn, right of center below the whole history; per recall turn, the lower right corner
    ax.legend(loc="center right" if per_turn else "lower right", bbox_to_anchor=(1, 0.38) if per_turn else None,
              fontsize=7.5, frameon=False, labelcolor=MUTED, ncol=1)
    runs = sorted(len(label_runs(t)) for t in arms)
    ax.text(0, 1.03, f"3,000 bootstrap draws per arm, {runs[0]} to {runs[-1]} runs each, 133 LongMemEval questions, answered by Sonnet 5.5",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    out = argv[0] if argv else "results/tool_arms_cloud.png"
    fig.text(0.99, 0.99, time.strftime("updated %Y-%m-%d %H:%M"), ha="right", va="top", fontsize=7.5, color=MUTED)
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    main()
