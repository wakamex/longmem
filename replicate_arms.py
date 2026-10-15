#!/usr/bin/env python3
"""Run further replicates of the endchat arms on the 133 multi-session questions and judge them.

    python3 replicate_arms.py --first 2 --last 5 [--arms sch reader ...]   # needs search_server.py running

Replicate N of an arm is tagged ARM_rN; the run already made under the same settings is replicate 1,
except for zoom, whose earlier run came from another harness, so its replicates start at 1. Rounds
go arm by arm so that no arm's runs are bunched in time, and so that an arm's next run starts
within the hour its views stay in the prompt cache. A run that stops short, as when the login
expires, is resumed up to three times; if it still falls short everything stops. When the
subscription's session limit is reached the run waits, 20 minutes at a time for up to 6 hours.
"""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, time
from pathlib import Path

ARMS = {  # tag of replicate 1: arguments of eval_endchat_tools.py
    "none133": ["--tool", "none"],
    "scout133": ["--tool", "search", "--scout"],
    "reader133": ["--tool", "search", "--reader"],
    "sch133": ["--tool", "sch"],
    "zoom133": ["--tool", "zoom"],
    "sch_view16k": ["--tool", "sch", "--view", "16000", "8000"],
    "scout_sch": ["--tool", "sch", "--scout"],
    "scout_zoom": ["--tool", "zoom", "--scout"],
    "sch_noview": ["--tool", "sch", "--no-view"],
    "full133": ["--tool", "full"],
    "sch_reworded": ["--tool", "sch", "--questions", "data/reworded_multi_session.json"],
    "sch_noview_reworded": ["--tool", "sch", "--no-view", "--questions", "data/reworded_multi_session.json"],
}
SIZE = {arm: 125 if "reworded" in arm else 133 for arm in ARMS}  # questions per run


def done(tag):
    p = Path(f"results/hypotheses_endchat_{tag}.jsonl")
    return len(p.read_text().splitlines()) if p.exists() else 0


def run(arm, n, jobs):
    tag = f"{arm}_r{n}"
    log = Path(f"/nvme/claude-tmp/lme-rep-{tag}.out")
    attempt = waits = 0
    while done(tag) < SIZE[arm] and attempt < 4 and waits < 18:
        seen = log.stat().st_size if log.exists() else 0
        subprocess.run([sys.executable, "eval_endchat_tools.py", *ARMS[arm], "--tag", tag, "--jobs", str(jobs)], stdout=log.open("a"), stderr=subprocess.STDOUT)
        if done(tag) >= SIZE[arm]:
            break
        if any(w in log.read_text()[seen:] for w in ("session limit", "weekly limit")):  # the subscription window is used up: wait for it to reset
            waits += 1
            time.sleep(1200)
        else:
            attempt += 1
            time.sleep(120)
    if done(tag) < SIZE[arm]:
        return tag, f"only {done(tag)} of {SIZE[arm]} answered"
    env = dict(os.environ)
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip().removeprefix("export ")] = v.strip().strip("'\"")
    j = subprocess.run([sys.executable, "eval_retrieval.py", "--hypotheses", f"results/hypotheses_endchat_{tag}.jsonl", "--judge", "openai"], capture_output=True, text=True, env=env)
    return tag, j.stdout.strip().splitlines()[-1] if j.stdout.strip() else j.stderr[-200:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    ap.add_argument("--parallel", type=int, default=3, help="arms at once")
    ap.add_argument("--jobs", type=int, default=4, help="questions at once within an arm")
    a = ap.parse_args()
    for n in range(a.first, a.last + 1):
        arms = [arm for arm in a.arms if not Path(f"results/hypotheses_endchat_{arm}_r{n}.eval.jsonl").exists() or done(f"{arm}_r{n}") < SIZE[arm]]
        with cf.ThreadPoolExecutor(a.parallel) as ex:
            for tag, result in ex.map(lambda arm: run(arm, n, a.jobs), arms):
                print(time.strftime("%H:%M"), tag, result, flush=True)
                if "of" in result and "answered" in result:
                    sys.exit(f"stopping: {tag} {result}")


if __name__ == "__main__":
    main()
