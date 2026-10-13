#!/usr/bin/env python3
"""Plot questions correct against cost per turn for the endchat tool arms, each cost priced at the share
of my turns that find the one-hour prompt cache warm, with the 95% interval of that share.

    uv run --no-project --with matplotlib python plot_tool_arms_range.py   # writes results/tool_arms_range.png

Vertical bars are the run-to-run variation of one run: two runs of sch that differed only in an
unused zoom tool disagreed on 11 of 133 questions.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_tools as c

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ARMS = {  # name: (color, y shift so equal scores do not overlap, label side)
    "no tool": (BLUE, 0, "right"), "zoom": (BLUE, 0, "right"),
    "search, whole messages": (ORANGE, 0, "right"), "search, excerpts": (ORANGE, 0, "right"), "sch": (ORANGE, 0, "right"),
    "sch, small view": (ORANGE, 0, "right"),
    "search, Haiku reads results": (AQUA, 0, "right"), "Haiku scouts, one main call": (AQUA, 0, "right"),
}
N, WARM = 133, 92
RUN_NOISE = 4.6  # 95% half-width of one run on these questions: two near-identical sch runs differed on 11 of 133

d = {}
for name in ARMS:
    run = c.zoom_run() if name == "zoom" else c.rows(f"results/endchat_stats_{c.ARMS[name]}.jsonl")
    evals = "results/hypotheses_endchat_multi133b_j38.eval.jsonl" if name == "zoom" else f"results/hypotheses_endchat_{c.ARMS[name]}.eval.jsonl"
    d[name] = {"correct": sum(v["label"] for v in c.rows(evals).values()), "expected": c.at(run, c.WARM), "lo": c.at(run, c.WARM_HIGH),
               "hi": c.at(run, c.WARM_LOW), "small": "small view" in name}
fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=160, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
pts = sorted(((d[n]["expected"], d[n]["correct"]) for n in ARMS), key=lambda p: (p[0], -p[1]))
front, best = [], -1
for x, y in pts:
    if y > best:
        front.append((x, y)); best = y
ax.plot(*zip(*front), color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1, label=f"Efficient frontier at {WARM}% warm")
for name, (color, dy, side) in ARMS.items():
    v = d[name]; y = v["correct"] + dy
    small = v["small"]
    ax.plot([v["lo"], v["hi"]], [y, y], color=color, lw=7, alpha=0.35, solid_capstyle="butt", zorder=2)
    ax.errorbar(v["expected"], y, yerr=RUN_NOISE, color=color, lw=1, alpha=0.55, capsize=2.5, zorder=2)
    ax.scatter(v["expected"], y, s=70, color=color, marker="s" if small else "o", edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.annotate(name, (v["hi"], y), xytext=(8, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center")
ax.annotate("no tool, small view: 48 correct, \\$0.014", (0.0165, 96.6), fontsize=8.5, color=INK, va="center")
ax.annotate("", xy=(0.014, 96.05), xytext=(0.014, 97.2), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
ax.scatter([], [], s=55, color=MUTED, label=f"Cost when {WARM}% of turns find the cache warm, as mine do")
ax.plot([], [], color=MUTED, lw=7, alpha=0.35, label="95% interval of that share, 91% to 93%")
ax.scatter([], [], s=55, color=MUTED, marker="s", label="View cut from 32,000 tokens to about 4,000")
ax.errorbar([], [], yerr=[], color=MUTED, lw=1, capsize=2.5, label="Run-to-run variation of one run, 95%")
for label, color in (("Summaries, with zoom or no tool", BLUE), ("Main model searches", ORANGE), ("Haiku helps", AQUA)):
    ax.scatter([], [], s=55, color=color, label=label)
ax.set_xlim(0, 0.135)
ax.set_ylim(96, 121.5)
ax.set_xlabel("Cost per turn at Claude Opus 5.5 prices with the one-hour prompt cache (lower is better)", fontsize=9.5, color=MUTED)
ax.set_ylabel(f"Questions correct, of {N} (higher is better)", fontsize=9.5, color=MUTED)
ax.xaxis.set_major_formatter(lambda x, _: f"${x:.2f}")
ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}  ({100 * y / N:.0f}%)")
ax.set_yticks([100, 105, 110, 115])
ax.grid(color=GRID, lw=0.7)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=MUTED, ncol=2, columnspacing=1.5, bbox_to_anchor=(0.0, 1.0))
ax.set_title("Reading old messages in a never-ending chat: accuracy against cost", fontsize=12, color=INK, loc="left", pad=22)
ax.text(0, 1.03, "133 multi-session LongMemEval questions, Claude Sonnet 5.5 answering, one chat of about 490 messages each", transform=ax.transAxes, fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig("results/tool_arms_range.png", facecolor=SURFACE)
print("frontier:", front)
