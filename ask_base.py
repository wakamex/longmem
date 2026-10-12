#!/usr/bin/env python3
"""Build imported-but-unasked endchat profiles once, then ask their questions again and again.

    python3 ask_base.py build --base DIR --id ID [--id ID ...]
    python3 ask_base.py ask --base DIR --tag T --id ID ...

build imports each question's haystack into DIR/ID and stops before the question. ask copies
each base profile to a scratch directory, asks the question, and writes
results/hypotheses_endchat_T.jsonl for eval_retrieval.py. The copy keeps the base unchanged, so
changes to the system prompt or the turn code can be compared on the same memory.
"""
import argparse, concurrent.futures, json, shutil, subprocess, sys, threading
from pathlib import Path

import eval_endchat as ee


def build(q, args):
    qid = q["question_id"]
    profile = args.base / qid
    if (profile / "BUILT").exists():
        return
    shutil.rmtree(profile, ignore_errors=True)
    profile.mkdir(parents=True)
    (profile / "config.toml").write_text(
        f'aop_bin = "{ee.AOP}"\n[compactor]\nmodel = "haiku"\neffort = "medium"\n[turn]\nmodel = "sonnet"\neffort = "medium"\n'
    )
    hist = profile / "history.json"
    hist.write_text(json.dumps(ee.history(q)))
    imp = subprocess.run([str(args.endchat), "import", str(profile), str(hist), "--pace", str(args.pace), "--timeout-secs", "1800"], capture_output=True, text=True)
    hist.unlink()
    try:
        done = json.loads(imp.stdout.strip().splitlines()[-1]).get("complete")
    except (IndexError, json.JSONDecodeError):
        done = False
    print(f"[{qid}] import {'complete' if done else 'INCOMPLETE'}", flush=True)
    if done:
        (profile / "BUILT").write_text("")


def ask(q, args, lock):
    qid = q["question_id"]
    scratch = args.scratch / qid
    shutil.rmtree(scratch, ignore_errors=True)
    shutil.copytree(args.base / qid, scratch)
    if args.effort:
        cfg = scratch / "config.toml"
        cfg.write_text(cfg.read_text().replace('[turn]\nmodel = "sonnet"\neffort = "medium"', f'[turn]\nmodel = "sonnet"\neffort = "{args.effort}"'))
    r = subprocess.run(
        [str(args.endchat), "say", str(scratch), "--date", ee.iso(q["question_date"]), q["question"] + ee.ANSWER_SUFFIX],
        capture_output=True, text=True,
    )
    log = (scratch / "main").glob("*.jsonl")
    zooms = sum(1 for f in log for l in f.read_text().splitlines() if '"kind":"tool"' in l)
    with lock:
        with (Path("results") / f"hypotheses_endchat_{args.tag}.jsonl").open("a") as f:
            f.write(json.dumps({"question_id": qid, "hypothesis": r.stdout.strip(), "tool_calls": zooms}) + "\n")
    print(f"[{qid}] zooms={zooms} {r.stdout.strip()[:70]!r} (expected {str(q['answer'])[:40]!r})", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["build", "ask"])
    p.add_argument("--base", type=Path, default=Path("/nvme/claude-tmp/lme_base"))
    p.add_argument("--scratch", type=Path, default=Path("/nvme/claude-tmp/lme_ask"))
    p.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    p.add_argument("--endchat", type=Path, default=Path("/code/endchat/target/release/endchat"))
    p.add_argument("--id", action="append", required=True)
    p.add_argument("--tag", default="ask")
    p.add_argument("--pace", type=int, default=50)
    p.add_argument("--effort", help="turn effort for ask, replacing the profile's")
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    data = [q for q in json.loads(args.dataset.read_text()) if q["question_id"] in args.id]
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        fs = [pool.submit(build if args.mode == "build" else ask, q, args, *([] if args.mode == "build" else [lock])) for q in data]
        for f in fs:
            f.result()


if __name__ == "__main__":
    main()
