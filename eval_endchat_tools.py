#!/usr/bin/env python3
"""Ask LongMemEval questions again on already imported endchat profiles, with a search command
in place of zoom or beside it.

    python3 eval_endchat_tools.py --tool search --tag search133   # needs search_server.py running
    python3 eval_endchat_tools.py --tool sch --tag sch133
    python3 eval_endchat_tools.py --tool search --zoom --tag searchzoom133
    python3 eval_endchat_tools.py --tool search --zoom --excerpt --tag excerptzoom133
    python3 eval_endchat_tools.py --tool search --reader --tag reader133
    python3 eval_endchat_tools.py --tool search --scout --tag scout133
    python3 eval_endchat_tools.py --tool none --tag none133
    python3 eval_endchat_tools.py --tool sch --view 16000 8000 --tag sch_view16k
    python3 eval_retrieval.py --hypotheses results/hypotheses_endchat_search133.jsonl --judge openai

Each profile under --base already holds the question and its answer from the run that built it.
A copy is cut back to the messages before the question, so the turn sees the memory that run saw.
The system prompt and the note sent with the question come from prompts/NAME/, where NAME is the
tool, with _zoom, _excerpt and _reader added for those options. With --reader, Claude Haiku reads
each search's messages and passes on what it quotes. With --scout, Haiku takes the turn first with
the search command and reports passages instead of answering, and the turn model then answers in
one call without tools, from the view and that report (prompts/scout/).

Writes results/hypotheses_endchat_TAG.jsonl and results/endchat_stats_TAG.jsonl. The stats hold
each call's token usage, each tool call with the dataset indexes of the messages it returned, the
messages zoom opened whole, and the share of the answer-bearing turns among all of those.
"""
import argparse, concurrent.futures, json, re, shutil, subprocess, sys, threading
from pathlib import Path

import eval_endchat as ee
from recall_compare import evidence, import_to_dataset

HERE = Path(__file__).resolve().parent
COMMANDS = {"search": HERE / "search_tool.py", "sch": HERE / "sch_tool.py"}


def read_log(profile):
    msgs = [json.loads(l) for f in sorted((profile / "main").glob("*.jsonl")) for l in f.read_text().split("\n") if l.startswith("{")]
    return sorted(msgs, key=lambda m: m["i"])


def rewind(profile, n):
    """Drop every message from `n` on, with the tree nodes and view lines that cover them."""
    for f in (profile / "main").glob("*.jsonl"):
        keep = [l for l in f.read_text().split("\n") if l.startswith("{") and json.loads(l)["i"] < n]
        f.write_text("".join(l + "\n" for l in keep)) if keep else f.unlink()
    for f in (profile / "tree").glob("*.jsonl"):
        keep = [l for l in f.read_text().split("\n") if l.startswith("{") and (json.loads(l)["i"] + 1) << json.loads(l)["l"] <= n]
        f.write_text("".join(l + "\n" for l in keep)) if keep else f.unlink()
    for name in ("main.idx", "tree.idx", "ledger.jsonl", "lock"):
        (profile / name).unlink(missing_ok=True)
    view = json.loads((profile / "view.json").read_text())
    exact = True
    for key in ("chat", "compaction"):
        lines = [x for x in view[key]["lines"] if (x[1] + 1) << x[0] <= n]
        exact &= bool(lines) and (lines[-1][1] + 1) << lines[-1][0] == n
        view[key]["lines"] = lines
    view["len"] = n
    # When the turn's messages caused a merge across the cut, endchat replays the view from the log.
    (profile / "view.json").write_text(json.dumps(view)) if exact else (profile / "view.json").unlink()
    return exact


def log_to_dataset(q, hist):
    """Dataset index of each log message. A long message is logged as several pages."""
    to_ds, out, j = import_to_dataset(q), {}, 0
    contents = [m["text"] for m in ee.history(q)]
    for p, text in enumerate(contents):
        got = ""
        while got != text:
            if j >= len(hist) or not text.startswith(got + hist[j]["text"]):
                raise ValueError(f"log message {j} does not continue dataset message {p}")
            got += hist[j]["text"]
            out[j] = to_ds[p]
            j += 1
    return out


def prepare(q, args, profile):
    """A copy of the question's profile cut back to before the question, with the search inputs."""
    shutil.rmtree(profile, ignore_errors=True)
    shutil.copytree(args.base / q["question_id"], profile, ignore=shutil.ignore_patterns(".aop", ".tmp"))
    asked = q["question"] + ee.ANSWER_SUFFIX
    if args.view:
        n, same_view = small_view(q, args, profile), False
    else:
        n = max(m["i"] for m in read_log(profile) if m["kind"] == "user" and m["text"] == asked)
        same_view = rewind(profile, n)
    hist = read_log(profile)
    (profile / "messages.jsonl").write_text("".join(json.dumps(m) + "\n" for m in hist))
    session = profile / "sch" / "projects" / "chat"
    session.mkdir(parents=True)
    with (session / "endchat.jsonl").open("w") as f:
        for m in hist:
            role = "user" if m["kind"] == "user" else "assistant"
            content = m["text"] if role == "user" else [{"type": "text", "text": m["text"]}]
            f.write(json.dumps({"type": role, "timestamp": m["date"], "i": m["i"], "message": {"role": role, "content": content}}, ensure_ascii=False) + "\n")
    return n, same_view, log_to_dataset(q, hist)


def small_view(q, args, profile):
    """Replace the profile with one whose tree is built to the root, so that endchat can lay out a
    view within --view bytes. The built tree is kept under --view-base and shared between runs."""
    base = args.view_base / q["question_id"]
    if not (base / "BUILT").exists():
        shutil.rmtree(base, ignore_errors=True)
        shutil.copytree(profile, base)
        asked = q["question"] + ee.ANSWER_SUFFIX
        rewind(base, max(m["i"] for m in read_log(base) if m["kind"] == "user" and m["text"] == asked))
        (base / "view.json").unlink(missing_ok=True)
        (base / "config.toml").write_text(f'aop_bin = "{ee.AOP}"\n[compactor]\nmodel = "haiku"\neffort = "medium"\n')
        (base / "empty.json").write_text("[]")
        # an import of nothing waits until every summary the log allows has been written
        r = subprocess.run([str(args.endchat), "import", str(base), str(base / "empty.json"), "--timeout-secs", "1800"], capture_output=True, text=True)
        done = json.loads(r.stdout.strip().splitlines()[-1])
        if not done.get("complete"):
            raise RuntimeError(f"summaries not built: {done}")
        (base / "build.json").write_text(json.dumps(done))
        (base / "BUILT").write_text("")
    shutil.rmtree(profile)
    shutil.copytree(base, profile, ignore=shutil.ignore_patterns(".aop", ".tmp", "lock", "ledger.jsonl", "view.json", "BUILT", "empty.json"))
    return len(read_log(profile))


def configure(profile, args, system, tail, model, effort, tool=None, zoom=False):
    commands = f'[turn.commands]\n{tool} = "{COMMANDS[tool]}"\n' if tool else ""
    if args.view:  # after [turn.commands], a new table
        commands += f"[limits]\nview_high = {args.view[0]}\nview_low = {args.view[1]}\n"
    (profile / "config.toml").write_text(
        f'aop_bin = "{ee.AOP}"\nsystem_prompt = "{system}"\n'
        f'[compactor]\nmodel = "haiku"\neffort = "medium"\n'
        f'[turn]\nmodel = "{model}"\neffort = "{effort}"\nzoom = {str(zoom).lower()}\n'
        f"tail_note = {json.dumps(tail.strip(), ensure_ascii=False)}\n" + commands
    )


def say(profile, args, q, text):
    r = subprocess.run([str(args.endchat), "say", str(profile), "--date", ee.iso(q["question_date"]), text], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"say failed: {r.stderr[-300:]}")
    return r.stdout.strip()


def turn_usage(profile):
    ledger = [json.loads(l) for l in (profile / "ledger.jsonl").read_text().splitlines() if l.startswith("{")]
    return [c["usage"] for c in ledger if c["kind"] == "turn" and c.get("usage")]


def run_question(q, args, lock, hyp_path, stats_path):
    qid = q["question_id"]
    profile = args.workdir / qid
    asked = q["question"] + ee.ANSWER_SUFFIX
    n, same_view, to_ds = prepare(q, args, profile)
    side = []  # token usage of Haiku calls made for the turn model
    if args.scout:
        prompts = HERE / "prompts" / "scout"
        (profile / "search_mode").write_text("whole")
        configure(profile, args, prompts / "system.md", (prompts / "tail.txt").read_text(), "haiku", "medium", tool=args.tool)
        found = say(profile, args, q, q["question"])
        side = turn_usage(profile)
        searched = profile  # tool calls and zooms are read from the scout's profile
        profile = args.workdir / f"{qid}.answer"
        prepare(q, args, profile)
        none = found.strip().lower().strip(".") in ("", "none")
        tail = (prompts / "answer_tail_none.txt").read_text() if none else (prompts / "answer_tail.txt").read_text().format(found=found)
        configure(profile, args, prompts / "answer_system.md", tail, args.turn_model, args.turn_effort)
    else:
        prompts = HERE / "prompts" / (args.tool + "_zoom" * args.zoom + "_excerpt" * args.excerpt + "_reader" * args.reader)
        (profile / "search_mode").write_text("reader" if args.reader else "excerpt" if args.excerpt else "whole")
        configure(profile, args, prompts / "system.md", (prompts / "tail.txt").read_text(), args.turn_model, args.turn_effort, tool=None if args.tool == "none" else args.tool, zoom=args.zoom)
        searched = profile
    answer = say(profile, args, q, asked)
    calls_path = searched / "tool_calls.jsonl"
    tools = [json.loads(l) for l in calls_path.read_text().splitlines()] if calls_path.exists() else []
    for t in tools:
        t["ids"] = sorted({to_ds[i] for i in t["ids"]})
        if "searched" in t:
            t["searched"] = sorted({to_ds[i] for i in t["searched"]})
        side += [t.pop("reader_usage")] if "reader_usage" in t else []
    zoomed = set()
    for m in read_log(searched)[n:]:
        if m["kind"] == "tool" and m["text"].startswith("zoom "):
            z = json.loads(m["text"][5:])
            if z["n"] == 1 and z["id"] in to_ds:
                zoomed.add(to_ds[z["id"]])
    ev = evidence(q)
    seen = {i for t in tools for i in t["ids"]} | zoomed
    if args.scout:  # the answering model sees only the messages the scout quoted
        quoted = {to_ds[int(i)] for i in re.findall(r"^(\d+) \|", found, re.M) if int(i) in to_ds}
        searched_ids, seen = seen, seen & quoted
    stats = {
        "question_id": qid,
        "same_view": same_view,
        "turn_usage": turn_usage(profile),
        "side_usage": side,
        "tool_calls": tools,
        "zoomed": sorted(zoomed),
        "messages_returned": len(seen),
        "evidence_recall": len(seen & ev) / len(ev) if ev else None,
    }
    if args.scout:
        stats["scout_report_chars"] = len(found)
        stats["searched_recall"] = len(searched_ids & ev) / len(ev) if ev else None
    elif args.reader:
        looked = {i for t in tools for i in t.get("searched", [])}
        stats["searched_recall"] = len(looked & ev) / len(ev) if ev else None
    with lock:
        with hyp_path.open("a") as f:
            f.write(json.dumps({"question_id": qid, "hypothesis": answer}) + "\n")
        with stats_path.open("a") as f:
            f.write(json.dumps(stats) + "\n")
    print(f"[{qid}] calls={len(tools)} recall={stats['evidence_recall']} {answer[:60]!r} (expected {str(q['answer'])[:40]!r})", flush=True)
    if not args.keep:
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(searched, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", choices=[*sorted(COMMANDS), "none"], required=True, help="none answers from the view in one call")
    p.add_argument("--tag", required=True)
    p.add_argument("--zoom", action="store_true", help="leave zoom and date on beside the tool")
    p.add_argument("--excerpt", action="store_true", help="search returns the matching sentences of long messages")
    p.add_argument("--reader", action="store_true", help="Haiku reads each search's messages and passes on what it quotes")
    p.add_argument("--view", type=int, nargs=2, metavar=("HIGH", "LOW"), help="bound the view to these bytes instead of endchat's 128000 and 64000")
    p.add_argument("--view-base", type=Path, default=Path("/nvme/claude-tmp/lme_fulltree"), help="where profiles with a fully built tree are kept")
    p.add_argument("--scout", action="store_true", help="Haiku searches first and reports; the turn model answers in one call without tools")
    p.add_argument("--base", type=Path, default=Path("/nvme/claude-tmp/lme_multib"))
    p.add_argument("--workdir", type=Path)
    p.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    p.add_argument("--endchat", type=Path, default=Path("/nvme/claude-tmp/endchat-search/target/release/endchat"))
    p.add_argument("--question-id", action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--turn-model", default="sonnet")
    p.add_argument("--turn-effort", default="medium")
    p.add_argument("--keep", action="store_true", help="keep each question's profile directory")
    args = p.parse_args()
    args.workdir = args.workdir or Path(f"/nvme/claude-tmp/lme_{args.tag}")
    data = [q for q in json.loads(args.dataset.read_text()) if (args.base / q["question_id"]).is_dir()]
    if args.question_id:
        data = [q for q in data if q["question_id"] in args.question_id]
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
