#!/usr/bin/env python3
"""Diff two eval result files: show wins, losses, and both-wrong.

Usage:
    python diff_results.py results/eval_small_k50_mr1_obs0.jsonl results/eval_colbert_lightonai_ColBERT-Zero_k50_mr1_obs0.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    results = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line.strip())
            if "correct" in r:
                results[r["question_id"]] = r
    return results


def main():
    parser = argparse.ArgumentParser(description="Diff two eval result files")
    parser.add_argument("file_a", type=Path, help="Baseline result file")
    parser.add_argument("file_b", type=Path, help="Comparison result file")
    args = parser.parse_args()

    a = load(args.file_a)
    b = load(args.file_b)

    if not a:
        print(f"{args.file_a}: no LLM-scored results found", file=sys.stderr)
        sys.exit(1)
    if not b:
        print(f"{args.file_b}: no LLM-scored results found", file=sys.stderr)
        sys.exit(1)

    name_a = args.file_a.stem.replace("eval_", "")
    name_b = args.file_b.stem.replace("eval_", "")

    common = sorted(set(a) & set(b))
    a_correct = sum(1 for q in common if a[q]["correct"])
    b_correct = sum(1 for q in common if b[q]["correct"])

    print(f"A: {name_a}  ({a_correct}/{len(common)} = {a_correct/len(common)*100:.1f}%)")
    print(f"B: {name_b}  ({b_correct}/{len(common)} = {b_correct/len(common)*100:.1f}%)")

    wins = [q for q in common if b[q]["correct"] and not a[q]["correct"]]
    losses = [q for q in common if a[q]["correct"] and not b[q]["correct"]]
    both_wrong = [q for q in common if not a[q]["correct"] and not b[q]["correct"]]

    def print_section(label, qids):
        print(f"\n{label} ({len(qids)}):")
        print("─" * 90)
        for qid in qids:
            q = a[qid]["question"][:65]
            gt = a[qid]["answer"][:30]
            aa = a[qid].get("llm_answer", "")[:30]
            ba = b[qid].get("llm_answer", "")[:30]
            print(f"  {qid}  {q}")
            print(f"    truth: {gt}")
            print(f"    A: {aa}")
            print(f"    B: {ba}")

    print_section("B wins (B right, A wrong)", wins)
    print_section("B losses (A right, B wrong)", losses)
    print_section("Both wrong", both_wrong)

    print(f"\nNet: B is {'+' if len(wins) >= len(losses) else ''}{len(wins) - len(losses)} vs A")


if __name__ == "__main__":
    main()
