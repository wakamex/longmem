#!/usr/bin/env python3
"""Compare retrieval results for a single question across eval runs.

Usage:
    python compare_question.py 6d550036
    python compare_question.py 6d550036 --show-sessions
    python compare_question.py 6d550036 --show-rankings --runs colbert small --top 20
    python compare_question.py 37f165cf --key 102 435 --show-rankings --runs obs0
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval results for a question")
    parser.add_argument("question_id", help="Question ID to look up")
    parser.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--show-sessions", action="store_true",
                        help="Print answer session content")
    parser.add_argument("--show-rankings", action="store_true",
                        help="Print top-K retrieved messages with oracle highlights")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Filter runs by substring match (e.g. 'colbert' 'small')")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top retrieved messages to show (default: 10)")
    parser.add_argument("--key", nargs="*", type=int, default=None,
                        help="Key oracle message IDs (highlighted with *** in output)")
    args = parser.parse_args()

    key_ids = set(args.key) if args.key else None

    # Load oracle
    oracle = None
    if args.dataset.exists():
        with args.dataset.open() as f:
            for q in json.load(f):
                if q["question_id"] == args.question_id:
                    oracle = q
                    break

    if not oracle:
        print(f"Question {args.question_id} not found in dataset")
        return

    # Load key message IDs from sidecar file if not overridden via CLI
    if key_ids is None:
        key_file = args.dataset.parent / "key_messages.json"
        if key_file.exists():
            with key_file.open() as f:
                all_keys = json.load(f)
            dataset_keys = all_keys.get(args.question_id, [])
        else:
            dataset_keys = oracle.get("key_message_ids", [])
        key_ids = set(dataset_keys) if dataset_keys else set()

    # Build message-id -> (text, session_id) lookup
    answer_sids = set(oracle.get("answer_session_ids", []))
    sids = oracle.get("haystack_session_ids", [])
    msg_lookup = {}  # id -> (text, session_id, is_oracle)
    msg_id = 0
    for si, session in enumerate(oracle.get("haystack_sessions", [])):
        sid = sids[si] if si < len(sids) else None
        for turn in session:
            text = f"[{turn['role']}] {turn['content']}"
            msg_lookup[msg_id] = (text, sid, sid in answer_sids)
            msg_id += 1

    n_oracle_msgs = sum(1 for _, _, is_o in msg_lookup.values() if is_o)
    print(f"Question:  {oracle['question']}")
    print(f"Answer:    {oracle['answer']}")
    print(f"Type:      {oracle['question_type']}")
    print(f"Answer sessions: {len(answer_sids)}  {sorted(answer_sids)}")
    n_messages = sum(len(s) for s in oracle.get("haystack_sessions", []))
    n_sessions = len(oracle.get("haystack_sessions", []))
    print(f"Haystack:  {n_messages} messages across {n_sessions} sessions")
    print(f"Oracle messages: {n_oracle_msgs}")
    if key_ids:
        print(f"Key messages: {sorted(key_ids)}")

    # Always show oracle messages
    print(f"\n  {'id':<6} {'session':<25} message")
    print(f"  {'─' * 70}")
    for mid in sorted(msg_lookup):
        text, sid, is_oracle = msg_lookup[mid]
        if is_oracle:
            text_preview = text[:80].replace("\n", " ")
            marker = " ***" if mid in key_ids else ""
            print(f"  {mid:<6} {sid or '':25} {text_preview}{marker}")

    if args.show_sessions:
        for i, session in enumerate(oracle.get("haystack_sessions", [])):
            sid = sids[i] if i < len(sids) else None
            if sid in answer_sids:
                print(f"\n--- Answer session: {sid} ({len(session)} turns) ---")
                for turn in session:
                    text = turn["content"][:150].replace("\n", " ")
                    print(f"  [{turn['role']}] {text}")

    # Collect matching eval files
    all_runs = {}
    for path in sorted(args.results_dir.glob("eval_*.jsonl")):
        name = path.stem.replace("eval_", "")
        if args.runs and not any(r in name for r in args.runs):
            continue
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if r["question_id"] == args.question_id:
                    all_runs[name] = r
                    break

    # Summary table
    header = f"  {'Run':<45} {'P':>6} {'R':>6} {'MaxR':>6} {'R/MR':>6} {'hits':>6}"
    if key_ids:
        header += f" {'key':>5}"
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"  {'':45} {'':>6} {'':>6} {'':>6} {'':>6} {'(of ' + str(n_oracle_msgs) + ')':>6}")
    print(f"{'─' * len(header)}")
    for name, r in all_runs.items():
        p = r.get("topk_precision")
        rc = r.get("topk_recall")
        mr = r.get("max_recall")
        hits = r.get("topk_oracle_hits")
        topk_ids = r.get("topk_ids", [])
        if p is None and topk_ids:
            oracle_ids = {mid for mid, (_, _, is_o) in msg_lookup.items() if is_o}
            hits = sum(1 for mid in topk_ids if mid in oracle_ids)
            k = len(topk_ids)
            p = hits / k if k else 0
            rc = hits / n_oracle_msgs if n_oracle_msgs else 0
            mr = min(k, n_oracle_msgs) / n_oracle_msgs if n_oracle_msgs else 0
        if p is not None:
            nr = (rc / mr) if mr else 0
            line = f"  {name:<45} {p:>6.3f} {rc:>6.3f} {mr:>6.3f} {nr:>6.3f} {hits:>5}"
            if key_ids:
                key_hits = sum(1 for mid in topk_ids if mid in key_ids)
                key_ranks = [str(topk_ids.index(mid) + 1) if mid in topk_ids else "-"
                             for mid in sorted(key_ids)]
                line += f" {key_hits}/{len(key_ids)} @{','.join(key_ranks)}"
            print(line)
        else:
            hit = "HIT" if r.get("topk_session_hit") else "MISS"
            print(f"  {name:<45} {hit:>6}     -      -      -     -")
    if not all_runs:
        print(f"  (no results found for {args.question_id})")
    print(f"{'─' * len(header)}")

    # Show rankings
    if args.show_rankings:
        for name, r in all_runs.items():
            topk_ids = r.get("topk_ids")
            if not topk_ids:
                print(f"\n  {name}: no topk_ids saved (re-run to generate)")
                continue
            print(f"\n  {name}  (top {args.top} of {len(topk_ids)})")
            print(f"  {'rank':<5} {'id':<6} {'':>6}  message")
            print(f"  {'─' * 66}")
            for rank, mid in enumerate(topk_ids[:args.top], 1):
                info = msg_lookup.get(mid)
                if info:
                    text, sid, is_oracle = info
                    if mid in key_ids:
                        marker = " *** "
                    elif is_oracle:
                        marker = " <<< "
                    else:
                        marker = "     "
                    text_preview = text[:80].replace("\n", " ")
                    print(f"  {rank:<5} {mid:<6} {marker}  {text_preview}")
                else:
                    print(f"  {rank:<5} {mid:<6}         (message not found)")
            n_oracle = sum(1 for mid in topk_ids[:args.top] if msg_lookup.get(mid, (None, None, False))[2])
            n_key = sum(1 for mid in topk_ids[:args.top] if mid in key_ids) if key_ids else 0
            summary = f"  oracle messages in top-{args.top}: {n_oracle}"
            if key_ids:
                summary += f"  |  key messages: {n_key}/{len(key_ids)}"
            print(summary)


if __name__ == "__main__":
    main()
