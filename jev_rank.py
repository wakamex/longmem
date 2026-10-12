#!/usr/bin/env python3
"""Rank every message of a LongMemEval haystack by TypeSafe Jev's yes/no probability that it helps
answer the question, and write the ranking for recall_compare.py.

    python3 jev_rank.py --question-type multi-session --out results/jev_ranking_multi.jsonl [--limit 2]

One Jev request per message: the state is the message, and the question is a noul (yes/no) read.
The API key is read from JEV_API_KEY in the environment or in --env-file, and never printed.
Questions already in the output file are skipped, so a rerun resumes.
"""
import argparse, concurrent.futures, json, os, threading, time, urllib.error, urllib.request
from pathlib import Path

URL = "https://api.typesafe.ai/v1/systemone"  # --url points it at a Jev-compatible server such as djev
MAX_CHARS = 6000
INSTRUCTIONS = (
    "Does this chat message contain information needed to answer the question below? Answer yes when it states "
    "a fact, event, quantity, name or date that the question asks about, even in passing.\nQuestion: {question}"
)


def key_from(env_file: Path) -> str:
    if os.environ.get("JEV_API_KEY"):
        return os.environ["JEV_API_KEY"]
    for line in env_file.read_text().splitlines():
        if line.startswith("JEV_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("JEV_API_KEY not found")


def ask(url: str, key: str, state: str, question: str, stats: dict) -> float:
    body = json.dumps({
        "model": "jev-latest",
        "state": state,
        "questions": {"relevant": {"type": "noul", "instructions": INSTRUCTIONS.format(question=question)}},
    }).encode()
    for attempt in range(8):
        req = urllib.request.Request(url, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            with stats["lock"]:
                stats["input"] += d.get("usage", {}).get("input_tokens", 0)
                stats["calls"] += 1
            return d["answers"]["relevant"]["noul"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 2 ** attempt))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}")
        except (urllib.error.URLError, TimeoutError):
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("gave up after 8 attempts")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    ap.add_argument("--env-file", type=Path, default=Path("/code/aipocalypse/.env"))
    ap.add_argument("--question-type", default="multi-session")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--url", default=URL, help="Jev endpoint; a local server needs no real key")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    key = "local" if a.url != URL else key_from(a.env_file)
    data = [q for q in json.loads(a.dataset.read_text()) if q["question_type"] == a.question_type]
    done = {json.loads(l)["question_id"] for l in a.out.read_text().splitlines()} if a.out.exists() else set()
    todo = [q for q in data if q["question_id"] not in done][: a.limit]
    stats = {"lock": threading.Lock(), "input": 0, "calls": 0}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for q in todo:
            msgs = [f"[{m['role']}] {m['content']}"[:MAX_CHARS] for s in q["haystack_sessions"] for m in s]
            scores = list(pool.map(lambda t: ask(a.url, key, t, q["question"], stats), msgs))
            ranked = sorted(range(len(msgs)), key=lambda i: -scores[i])
            with a.out.open("a") as f:
                f.write(json.dumps({"question_id": q["question_id"], "ranked": ranked, "scores": [round(s, 4) for s in scores]}) + "\n")
            print(f"[{q['question_id']}] {len(msgs)} messages, {stats['calls']} calls, {stats['input']} input tokens, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
