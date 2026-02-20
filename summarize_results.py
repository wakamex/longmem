#!/usr/bin/env python3
"""Summarize retrieval and accuracy metrics from eval result files.

Usage:
    python summarize_results.py results/eval_colbert_*.jsonl
    python summarize_results.py results/eval_*.jsonl
    python summarize_results.py results/eval_small_k10_mr0.jsonl results/eval_colbert_lightonai_ColBERT-Zero_k10_mr0.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def summarize(path: Path):
    results = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print(f"  (empty)")
        return

    n = len(results)

    # Session hit
    hits = sum(1 for r in results if r.get("topk_session_hit"))
    exp_hits = sum(1 for r in results if r.get("expanded_session_hit"))

    print(f"  Questions: {n}")
    print(f"  Session-hit@K:    {hits}/{n} = {hits/n*100:.1f}%")
    print(f"  Session-hit+MR:   {exp_hits}/{n} = {exp_hits/n*100:.1f}%")

    # Precision/recall (if present)
    if "topk_precision" in results[0]:
        avg_p = sum(r.get("topk_precision", 0) for r in results) / n
        avg_r = sum(r.get("topk_recall", 0) for r in results) / n
        avg_mr = sum(r.get("max_recall", 0) for r in results) / n
        print(f"  Precision:        {avg_p:.3f}")
        print(f"  Recall:           {avg_r:.3f}  (max possible: {avg_mr:.3f})")

    # LLM accuracy (if present)
    if "correct" in results[0]:
        correct = sum(1 for r in results if r.get("correct"))
        print(f"  LLM accuracy:     {correct}/{n} = {correct/n*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Summarize eval result files")
    parser.add_argument("files", nargs="+", type=Path, help="Eval JSONL files")
    args = parser.parse_args()

    for path in sorted(args.files):
        if not path.exists():
            print(f"{path}: not found", file=sys.stderr)
            continue
        name = path.stem.replace("eval_", "")
        print(f"\n{name}")
        print("─" * 50)
        summarize(path)

    print()


if __name__ == "__main__":
    main()
