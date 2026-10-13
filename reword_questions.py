#!/usr/bin/env python3
"""Reword LongMemEval questions so that they share no content word with the turns that answer them.

    python3 reword_questions.py --question-type multi-session --out data/reworded_multi_session.json

The benchmark's questions were written from their evidence and repeat its words, which suits keyword
and regex search. Each question here is rewritten by Claude Sonnet, through AOP, to ask the same
thing in other words, as someone half remembering would. A rewrite is retried with the shared
words named while any remain. The output lists, per question, the rewrite, the original, and the
content words still shared with the evidence.
"""
import argparse, concurrent.futures as cf, json, re, sys
from pathlib import Path

from answer_with_aop import answer

STOP = set("""a about after again all also am an and any are as at be because been before being both but by can could did do does doing
down each few for from further get got had has have having he her here hers him his how i if in into is it its just like many may me
might more most much my no nor not now of off on once one only or other our out over own s same she should so some such t than that
the their them then there these they this those through to too under until up us very was we were what when where which while who
whom why will with would you your yours last ago since ever""".split())


def stems(text):
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP or len(w) < 3:
            continue
        for suffix in ("ing", "ies", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def shared(question, evidence):
    ev = stems(evidence)
    return sorted({w for w in re.findall(r"[a-z0-9]+", question.lower()) if stems(w) & ev})


PROMPT = """Rewrite a question a user asks about their own past chat with an assistant.

The rewrite must ask for exactly the same thing and have the same answer, but share no content word with the messages that hold the answer: no noun, verb, adjective, name or number that appears in them, in any inflection. Use synonyms, descriptions or broader terms, the way someone half remembering would ask. Keep it one natural question in the first person. Do not include or hint at the answer.

Question: {question}
Answer (for your understanding only): {answer}

The messages that hold the answer:
{evidence}
{retry}
Reply with the rewritten question only."""


def reword(q, model, effort, workdir):
    evidence = "\n".join(f"- {m['content']}" for s in q["haystack_sessions"] for m in s if m.get("has_answer"))
    best, retry = None, ""
    for _ in range(4):
        text, _ = answer(PROMPT.format(question=q["question"], answer=q["answer"], evidence=evidence, retry=retry), model, effort, workdir)
        text = text.strip().strip('"')
        left = shared(text, evidence)
        if best is None or len(left) < len(best[1]):
            best = (text, left)
        if not left:
            break
        retry = f"\nYour last rewrite was: {text}\nIt still shares these words with the messages: {', '.join(left)}. Replace every one.\n"
    return {"question_id": q["question_id"], "question": best[0], "original": q["question"], "shared_words": best[1],
            "original_shared_words": shared(q["question"], evidence)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    ap.add_argument("--question-type", default="multi-session")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--workdir", type=Path, default=Path("/nvme/claude-tmp/answer_aop"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.workdir.mkdir(parents=True, exist_ok=True)
    data = [q for q in json.loads(a.dataset.read_text()) if q["question_type"] == a.question_type]
    done = {r["question_id"]: r for r in json.loads(a.out.read_text())} if a.out.exists() else {}
    todo = [q for q in data if q["question_id"] not in done and any(m.get("has_answer") for s in q["haystack_sessions"] for m in s)]
    print(f"{len(todo)} to reword ({len(done)} done)", flush=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for f in cf.as_completed([ex.submit(reword, q, a.model, a.effort, a.workdir) for q in todo]):
            try:
                r = f.result()
            except Exception as e:
                print(f"failed: {e!r}", file=sys.stderr, flush=True)
                continue
            done[r["question_id"]] = r
            print(r["question_id"], len(r["original_shared_words"]), "->", len(r["shared_words"]), "|", r["question"][:90], flush=True)
    order = {q["question_id"]: n for n, q in enumerate(data)}
    a.out.write_text(json.dumps(sorted(done.values(), key=lambda r: order[r["question_id"]]), indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
