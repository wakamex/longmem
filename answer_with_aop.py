#!/usr/bin/env python3
"""Re-answer a saved retrieval run with a different answerer, through AOP.

Takes the retrieved message ids stored in an eval_*.jsonl file, rebuilds the observations and the
answer prompt exactly as eval_retrieval.py does, and sends them to a Claude model in a sealed AOP
run with no tools. Writes hypotheses in the format eval_retrieval.py --hypotheses scores:

    python3 answer_with_aop.py results/eval_small_k50_mr1_obs0.jsonl --tag small_sonnet
    python3 eval_retrieval.py --hypotheses results/hypotheses_small_sonnet.jsonl --judge openai
"""
import argparse, concurrent.futures as cf, json, os, shutil, subprocess, sys, tempfile, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from eval_retrieval import ANSWER_PROMPT, load_messages

AOP = shutil.which("aop") or sys.exit("aop is not on PATH")

def observations(oracle, topk_ids, message_range):
    msgs = load_messages(oracle)
    by_id = {m.id: m for m in msgs}
    max_id = max(by_id)
    ids = set()
    for oid in topk_ids:
        for off in range(-message_range, message_range + 1):
            if 0 <= oid + off <= max_id:
                ids.add(oid + off)
    return "\n".join(f"[{i+1}] {by_id[m].text}" for i, m in enumerate(sorted(ids)))

def answer(prompt, model, effort, workdir):
    with tempfile.NamedTemporaryFile("w", dir=workdir, suffix=".txt", delete=False) as f:
        f.write(prompt); path = f.name
    try:
        p = subprocess.run([AOP, "run", "--agent", "claude", "--model", model, "--effort", effort, "--profile", "sealed",
                            "--no-web", "--tools", "", "--system-prompt", "Answer the user's question.", "--prompt-file", path,
                            "--timeout", "300", "--json"], capture_output=True, text=True, cwd=workdir)
        r = json.loads(p.stdout)
        if not r.get("succeeded"):
            raise RuntimeError(f"run failed: {r.get('error')}")
        subprocess.run([AOP, "cleanup", r["run_id"]], capture_output=True, cwd=workdir)
        return r["final_message"].strip(), r.get("usage") or {}
    finally:
        os.unlink(path)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_file", type=Path)
    ap.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    ap.add_argument("--message-range", type=int, default=1)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--workdir", type=Path, default=Path("/nvme/claude-tmp/answer_aop"))
    a = ap.parse_args()
    a.workdir.mkdir(parents=True, exist_ok=True)
    oracle = {q["question_id"]: q for q in json.loads(a.dataset.read_text())}
    rows = [json.loads(l) for l in a.eval_file.read_text().splitlines()]
    out = Path("results") / f"hypotheses_{a.tag}.jsonl"
    done = {json.loads(l)["question_id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in rows if r["question_id"] not in done]
    print(f"{len(todo)} to answer ({len(done)} done)", flush=True)
    lock = threading.Lock()
    def one(r):
        q = oracle[r["question_id"]]
        prompt = ANSWER_PROMPT.format(observations=observations(q, r["topk_ids"], a.message_range), question=q["question"])
        text, usage = answer(prompt, a.model, a.effort, a.workdir)
        with lock, out.open("a") as f:
            f.write(json.dumps({"question_id": r["question_id"], "hypothesis": text, "usage": usage}) + "\n")
        print(r["question_id"], repr(text[:60]), usage.get("input_tokens"), flush=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(one, r): r["question_id"] for r in todo}
        for f in cf.as_completed(futs):
            try: f.result()
            except Exception as e: print(f"[{futs[f]}] failed: {e!r}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
