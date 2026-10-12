#!/usr/bin/env python3
"""Run LongMemEval through endchat: import each question's haystack as a chat, let endchat
summarize it into its memory tree, then ask the question as a turn.

Writes hypotheses in the same format as eval_retrieval.py, so the strict judge scores them:

    python3 eval_endchat.py --question-type multi-session --limit 10 --tag pilot
    python3 eval_retrieval.py --hypotheses results/hypotheses_endchat_pilot.jsonl \
        --judge openai --judge-model gemini-3.8-flash

Sessions are imported in date order, as a real chat would be (211 of the 500 haystacks are
shuffled). Each question uses a throwaway endchat profile.
"""

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ANSWER_SUFFIX = (
    "\n\nGive ONLY the answer — a number, name, or short phrase. "
    "Do not explain your reasoning or list supporting evidence. "
    "If your memory doesn't contain enough information, say \"I don't know\"."
)
AOP = shutil.which("aop") or sys.exit("aop is not on PATH")
DATE = re.compile(r"(\d{4})/(\d\d)/(\d\d) \(\w+\) (\d\d):(\d\d)")


def iso(date: str) -> str:
    y, mo, d, h, mi = DATE.match(date).groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:00Z"


def history(q: dict) -> list[dict]:
    sessions = sorted(zip(q["haystack_dates"], q["haystack_sessions"]), key=lambda p: iso(p[0]))
    out = []
    for date, session in sessions:
        for m in session:
            out.append({"kind": "user" if m["role"] == "user" else "agent", "text": m["content"], "date": iso(date)})
    return out


def run_question(q: dict, args, write_lock: threading.Lock, hyp_path: Path, stats_path: Path) -> None:
    qid = q["question_id"]
    profile = args.workdir / qid
    shutil.rmtree(profile, ignore_errors=True)
    profile.mkdir(parents=True)
    (profile / "config.toml").write_text(
        f'aop_bin = "{AOP}"\n[compactor]\nmodel = "{args.compactor_model}"\neffort = "{args.compactor_effort}"\n'
        f'[turn]\nmodel = "{args.turn_model}"\neffort = "{args.turn_effort}"\n'
    )
    hist = profile / "history.json"
    hist.write_text(json.dumps(history(q)))
    imp = subprocess.run([str(args.endchat), "import", str(profile), str(hist), "--pace", str(args.pace), "--timeout-secs", str(args.import_timeout)], capture_output=True, text=True)
    hist.unlink()
    try:
        imported = json.loads(imp.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        imported = {"complete": False, "error": (imp.stderr or imp.stdout)[-300:]}
    if not imported.get("complete"):
        print(f"[{qid}] import incomplete: {imported}", file=sys.stderr, flush=True)
        return
    ask = subprocess.run(
        [str(args.endchat), "say", str(profile), "--date", iso(q["question_date"]), q["question"] + ANSWER_SUFFIX],
        capture_output=True, text=True,
    )
    if ask.returncode != 0:
        print(f"[{qid}] say failed: {ask.stderr[-300:]}", file=sys.stderr, flush=True)
        return
    turns = [json.loads(l) for l in (profile / "ledger.jsonl").read_text().splitlines() if l.startswith("{") and l.endswith("}") and l.count("{\"date\"") == 1]
    turns = [t["usage"] for t in turns if t["kind"] == "turn" and t.get("usage")]
    stats = {
        "question_id": qid,
        "import": imported,
        "turn_calls": len(turns),
        "turn_input_tokens": sum(t["input_tokens"] for t in turns),
        "turn_cached_input_tokens": sum(t["cached_input_tokens"] for t in turns),
    }
    with write_lock:
        with hyp_path.open("a") as f:
            f.write(json.dumps({"question_id": qid, "hypothesis": ask.stdout.strip()}) + "\n")
        with stats_path.open("a") as f:
            f.write(json.dumps(stats) + "\n")
    print(f"[{qid}] {ask.stdout.strip()[:80]!r} (expected {str(q['answer'])[:60]!r}) {imported['seconds']}s", flush=True)
    if not args.keep:
        shutil.rmtree(profile, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    p.add_argument("--endchat", type=Path, default=Path("/code/endchat/target/release/endchat"))
    p.add_argument("--workdir", type=Path, default=Path("/nvme/claude-tmp/lme"))
    p.add_argument("--tag", default="endchat")
    p.add_argument("--question-type")
    p.add_argument("--question-id", action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--jobs", type=int, default=1, help="questions at once; each runs 8 compaction workers")
    p.add_argument("--compactor-model", default="haiku")
    p.add_argument("--compactor-effort", default="medium")
    p.add_argument("--turn-model", default="sonnet")
    p.add_argument("--turn-effort", default="medium")
    p.add_argument("--pace", type=int, default=0, help="import this many messages at a time, waiting for summaries between batches")
    p.add_argument("--import-timeout", type=int, default=1800, help="give up on an import batch that has not been summarized after this many seconds")
    p.add_argument("--keep", action="store_true", help="keep each question's profile directory")
    args = p.parse_args()

    data = json.loads(args.dataset.read_text())
    if args.question_type:
        data = [q for q in data if q["question_type"] == args.question_type]
    if args.question_id:
        data = [q for q in data if q["question_id"] in args.question_id]
    if args.limit:
        data = data[: args.limit]
    hyp_path = Path("results") / f"hypotheses_endchat_{args.tag}.jsonl"
    stats_path = Path("results") / f"endchat_stats_{args.tag}.jsonl"
    done = {json.loads(l)["question_id"] for l in hyp_path.read_text().splitlines()} if hyp_path.exists() else set()
    todo = [q for q in data if q["question_id"] not in done]
    print(f"{len(todo)} questions to run ({len(done)} already done)", flush=True)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_question, q, args, lock, hyp_path, stats_path): q["question_id"] for q in todo}
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:  # one bad question must not end the run; rerun resumes
                print(f"[{futures[f]}] failed: {e!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
