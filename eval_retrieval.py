#!/usr/bin/env python3
"""Evaluate embedding retrieval on LongMemEval.

For each question:
  1. Load raw messages from oracle haystack_sessions
  2. Rank by cosine similarity to question using embedding model
  3. Give top-K messages (with ±N surrounding context) to an LLM answerer
  4. Judge answer correctness (generic inline judge + optional strict LongMemEval judge)

Usage:
    # Best config: K=50, message-range=±1, terse prompt
    python3 scripts/eval_retrieval.py \
        --dataset LongMemEval/data/longmemeval_s_cleaned.json \
        --question-type multi-session

    # Also run strict LongMemEval judge (GPT-4o) in one command
    python3 scripts/eval_retrieval.py \
        --dataset LongMemEval/data/longmemeval_s_cleaned.json \
        --question-type multi-session \
        --judge openai --judge-model gpt-4o

    # Re-score existing hypotheses with strict judge only (no retrieval)
    python3 scripts/eval_retrieval.py \
        --dataset LongMemEval/data/longmemeval_s_cleaned.json \
        --hypotheses results/hypotheses_small_k50_mr1.jsonl \
        --judge openai --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    id: int
    text: str


# ---------------------------------------------------------------------------
# Load messages from oracle
# ---------------------------------------------------------------------------

def load_messages(oracle_entry: dict) -> list[Message]:
    """Load raw chat messages from oracle haystack_sessions."""
    messages: list[Message] = []
    msg_id = 0
    for session in oracle_entry.get("haystack_sessions", []):
        for turn in session:
            text = f"[{turn['role']}] {turn['content']}"
            messages.append(Message(id=msg_id, text=text))
            msg_id += 1
    return messages


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _truncate_for_embedding(text: str, max_chars: int = 28000) -> str:
    """Truncate text to fit within embedding model token limit (~7K tokens)."""
    return text[:max_chars] if len(text) > max_chars else text


def _embed_texts_openai(
    texts: list[str], client, model: str = "text-embedding-3-small",
    batch_size: int = 512, max_retries: int = 5,
) -> list[list[float]]:
    """Embed texts using OpenAI embeddings API."""
    truncated = [_truncate_for_embedding(t) for t in texts]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                all_embeddings.extend([d.embedding for d in sorted_data])
                break
            except Exception as e:
                if attempt < max_retries - 1 and ("429" in str(e) or "rate" in str(e).lower()):
                    time.sleep(2 ** attempt + 1)
                else:
                    raise
        else:
            raise RuntimeError(f"Embedding failed after {max_retries} retries")
    return all_embeddings


def _embed_texts_cohere(
    texts: list[str], model: str = "embed-v4.0",
    input_type: str = "search_document", batch_size: int = 96,
    max_retries: int = 5,
) -> list[list[float]]:
    """Embed texts using Cohere embeddings API."""
    api_key = os.environ.get("CO_API_KEY") or os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("CO_API_KEY or COHERE_API_KEY not set")
    truncated = [_truncate_for_embedding(t) for t in texts]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://api.cohere.com/v2/embed",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"texts": batch, "model": model,
                          "input_type": input_type, "embedding_types": ["float"]},
                    timeout=60,
                )
                resp.raise_for_status()
                all_embeddings.extend(resp.json()["embeddings"]["float"])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                else:
                    raise
        else:
            raise RuntimeError(f"Cohere embedding failed after {max_retries} retries")
    return all_embeddings


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, embed_model: str, question_id: str) -> Path:
    model_short = embed_model.replace("text-embedding-3-", "").replace("/", "_")
    return cache_dir / model_short / f"{question_id}.json"


def _load_cache(path: Path, expected_count: int) -> list[list[float]] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        return data if len(data) == expected_count else None
    except Exception:
        return None


def _save_cache(path: Path, embeddings: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(embeddings, f)


# ---------------------------------------------------------------------------
# Cosine similarity ranking
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def embedding_rank(
    messages: list[Message], query: str, client,
    embed_model: str, cache_dir: Path | None = None, question_id: str = "",
) -> list[int]:
    """Rank messages by cosine similarity to query. Returns message IDs."""
    is_cohere = embed_model.startswith("embed-")

    # Get message embeddings (cached)
    msg_embeddings = None
    cp = None
    if cache_dir and question_id:
        cp = _cache_path(cache_dir, embed_model, question_id)
        msg_embeddings = _load_cache(cp, len(messages))

    if msg_embeddings is None:
        texts = [m.text for m in messages]
        if is_cohere:
            msg_embeddings = _embed_texts_cohere(texts, model=embed_model, input_type="search_document")
        else:
            msg_embeddings = _embed_texts_openai(texts, client, embed_model)
        if cp:
            _save_cache(cp, msg_embeddings)

    # Embed query
    if is_cohere:
        q_emb = _embed_texts_cohere([query], model=embed_model, input_type="search_query")[0]
    else:
        q_emb = _embed_texts_openai([query], client, embed_model)[0]

    # Score and sort
    scores = [(m.id, _cosine_similarity(q_emb, emb)) for m, emb in zip(messages, msg_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in scores]


# ---------------------------------------------------------------------------
# LLM answerer + inline judge
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
You are answering a question using ONLY the observations below from a chat \
memory system. These observations are retrieved memories from past conversations.

Observations:
{observations}

Question: {question}

Give ONLY the answer — a number, name, or short phrase. \
Do not explain your reasoning or list supporting evidence. \
If the observations don't contain enough information, say "I don't know"."""

JUDGE_PROMPT = """\
You are judging whether a model's answer is correct given a ground truth answer.

Question: {question}
Ground truth answer: {answer}
Model answer: {hypothesis}

Is the model's answer correct? It doesn't need to be word-for-word identical, \
but it must convey the same essential information. Answer "yes" or "no" only."""


# ---------------------------------------------------------------------------
# LongMemEval strict judge prompts (from official evaluate_qa.py)
# ---------------------------------------------------------------------------

_STRICT_BASE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
    "\n\nIs the model response correct? Answer yes or no only."
)

STRICT_JUDGE_PROMPTS = {
    "single-session-user": _STRICT_BASE,
    "single-session-assistant": _STRICT_BASE,
    "multi-session": _STRICT_BASE,
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model makes "
        "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
        "response is still correct. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. "
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal "
        "information correctly."
        "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {hypothesis}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is "
        "given but the asked information is not."
        "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {hypothesis}"
        "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}


def _get_strict_judge_prompt(question_type: str, question_id: str,
                             question: str, answer: str, hypothesis: str) -> str:
    if "_abs" in question_id:
        template = STRICT_JUDGE_PROMPTS["abstention"]
    else:
        template = STRICT_JUDGE_PROMPTS.get(question_type, _STRICT_BASE)
    return template.format(question=question, answer=answer, hypothesis=hypothesis)


class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            earliest = self._last + self.interval
            if now < earliest:
                time.sleep(earliest - now)
            self._last = time.monotonic()


_rate_limiter: RateLimiter | None = None


def _llm_call(client, model, messages, max_tokens, max_retries=8):
    for attempt in range(max_retries):
        if _rate_limiter:
            _rate_limiter.wait()
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0 if "gpt-4" in model or "gemini" in model else 1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if ("429" in str(e) or "rate" in err or "quota" in err
                    or "resource_exhausted" in err) and attempt < max_retries - 1:
                time.sleep(2 ** attempt + 5)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {max_retries} retries")


def answer_and_judge(
    top_messages: list[Message], oracle_entry: dict, client, model: str,
) -> tuple[str, bool]:
    """Give retrieved messages to LLM, get answer, judge correctness."""
    question = oracle_entry["question"]
    answer = str(oracle_entry["answer"])

    obs_text = "\n".join(
        f"[{i+1}] {m.text[:500]}" for i, m in enumerate(top_messages)
    )
    prompt = ANSWER_PROMPT.format(observations=obs_text, question=question)

    try:
        llm_answer = _llm_call(client, model, [{"role": "user", "content": prompt}], 1024)
    except Exception as e:
        return f"ERROR: {e}", False

    judge_prompt = JUDGE_PROMPT.format(question=question, answer=answer, hypothesis=llm_answer)
    try:
        judge_resp = _llm_call(client, model, [{"role": "user", "content": judge_prompt}], 256)
        correct = "yes" in judge_resp.lower()
    except Exception:
        return llm_answer, False

    return llm_answer, correct


# ---------------------------------------------------------------------------
# Strict judge (LongMemEval per-question-type prompts)
# ---------------------------------------------------------------------------

BAR_WIDTH = 20


def _bar(fraction: float) -> str:
    filled = round(fraction * BAR_WIDTH)
    return "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)


def _display_strict_results(
    results: dict[str, bool],
    ref_lookup: dict[str, dict],
    judge_label: str,
) -> None:
    type_counts: dict[str, tuple[int, int]] = {}
    for qid, correct in results.items():
        ref = ref_lookup.get(qid)
        if ref is None:
            continue
        qtype = ref["question_type"]
        c, t = type_counts.get(qtype, (0, 0))
        type_counts[qtype] = (c + (1 if correct else 0), t + 1)

    total_correct = sum(c for c, _ in type_counts.values())
    total_count = sum(t for _, t in type_counts.values())
    overall = total_correct / total_count * 100 if total_count else 0

    print(f"\nStrict LongMemEval Judge: {judge_label}")
    print("\u2500" * 68)
    max_type_len = max(len(t) for t in type_counts) if type_counts else 0
    for qtype in sorted(type_counts):
        c, t = type_counts[qtype]
        pct = c / t * 100 if t else 0
        print(f"  {qtype:<{max_type_len}} : {pct:5.1f}% [{_bar(pct / 100)}] ({c}/{t})")
    print(f"\n  LongMemEval Accuracy: {overall:.1f}% ({total_correct}/{total_count})")


def _run_strict_judge(
    hypotheses: list[dict],
    oracle_lookup: dict[str, dict],
    judge_model: str,
    eval_cache_path: Path,
    max_workers: int = 4,
) -> dict[str, bool]:
    """Run strict LongMemEval judge on hypotheses. Returns {qid: correct}."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set (needed for strict judge).", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Load eval cache
    cache: dict[str, bool] = {}
    if eval_cache_path.exists():
        with eval_cache_path.open() as fh:
            for raw in fh:
                try:
                    entry = json.loads(raw.strip())
                    cache[entry["question_id"]] = entry["label"]
                except (json.JSONDecodeError, KeyError):
                    pass

    to_judge = [h for h in hypotheses
                if h["question_id"] not in cache and h["question_id"] in oracle_lookup]

    if to_judge:
        print(f"\nStrict judging {len(to_judge)} questions ({len(cache)} cached) "
              f"with {judge_model}...", flush=True)

        eval_fh = eval_cache_path.open("a")
        done = [0]

        def judge(hyp: dict) -> dict:
            ref = oracle_lookup[hyp["question_id"]]
            prompt = _get_strict_judge_prompt(
                ref["question_type"], hyp["question_id"],
                ref["question"], str(ref["answer"]), hyp["hypothesis"],
            )
            try:
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=16, temperature=0,
                )
                response = resp.choices[0].message.content.strip()
                label = "yes" in response.lower()
            except Exception as exc:
                response = str(exc)
                label = False
            return {"question_id": hyp["question_id"], "label": label,
                    "judge_response": response}

        workers = max(1, min(max_workers, len(to_judge)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_qid = {pool.submit(judge, h): h["question_id"] for h in to_judge}
            for future in concurrent.futures.as_completed(future_to_qid):
                qid = future_to_qid[future]
                try:
                    result = future.result()
                    cache[result["question_id"]] = result["label"]
                    eval_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    eval_fh.flush()
                    done[0] += 1
                    status = "correct" if result["label"] else "wrong"
                    print(f"  [{done[0]}/{len(to_judge)}] {qid}: {status}", flush=True)
                except Exception as exc:
                    print(f"  [{qid}] ERROR: {exc}", file=sys.stderr, flush=True)
        eval_fh.close()

    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate embedding retrieval on LongMemEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="LongMemEval dataset JSON (e.g. longmemeval_s_cleaned.json)")
    parser.add_argument("--embed-model", default="text-embedding-3-small",
                        help="Embedding model (OpenAI text-embedding-3-* or Cohere embed-*)")
    parser.add_argument("--answerer", default="gemini-3-flash-preview",
                        help="LLM model for answering + inline judging")
    parser.add_argument("--retrieval-k", type=int, default=50,
                        help="Number of top messages to retrieve (default: 50)")
    parser.add_argument("--message-range", type=int, default=1,
                        help="Include ±N surrounding messages around each hit (default: 1)")
    parser.add_argument("--question-type", type=str, default=None,
                        help="Filter to question type (e.g. multi-session)")
    parser.add_argument("--question-id", type=str, default=None,
                        help="Run a single question")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--scorer-rpm", type=int, default=0,
                        help="Rate limit LLM calls (requests/min, 0=unlimited)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output path")
    parser.add_argument("--judge", default=None,
                        help='Strict LongMemEval judge: "openai" for OpenAI API')
    parser.add_argument("--judge-model", default="gpt-4o",
                        help="Model for strict judge (default: gpt-4o)")
    parser.add_argument("--hypotheses", type=Path, default=None,
                        help="Re-score existing hypotheses file (skip retrieval)")
    args = parser.parse_args()

    # Load dataset
    with args.dataset.open() as f:
        oracle_data = json.load(f)
    oracle_lookup = {q["question_id"]: q for q in oracle_data}

    # Mode: re-score existing hypotheses
    if args.hypotheses:
        if not args.judge:
            parser.error("--judge is required when using --hypotheses")
        hypotheses = []
        with args.hypotheses.open() as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    hypotheses.append(json.loads(line))
        if not hypotheses:
            print("No hypotheses found.", file=sys.stderr)
            sys.exit(1)
        eval_cache_path = args.hypotheses.with_suffix(".eval.jsonl")
        cache = _run_strict_judge(
            hypotheses, oracle_lookup, args.judge_model,
            eval_cache_path, max_workers=args.max_workers,
        )
        judge_label = f"openai/{args.judge_model}" if args.judge == "openai" else args.judge
        _display_strict_results(cache, oracle_lookup, judge_label)
        return

    # Output paths
    model_short = args.embed_model.replace("text-embedding-3-", "")
    if args.output:
        out_jsonl = args.output
    else:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = args.results_dir / f"eval_{model_short}_k{args.retrieval_k}_mr{args.message_range}.jsonl"
    hyp_stem = out_jsonl.stem.replace("eval_", "hypotheses_")
    if hyp_stem == out_jsonl.stem:
        hyp_stem = out_jsonl.stem + "_hypotheses"
    out_hypotheses = out_jsonl.with_name(hyp_stem + ".jsonl")

    question_ids = sorted(oracle_lookup.keys())
    if args.question_id:
        question_ids = [args.question_id]
    if args.question_type:
        question_ids = [qid for qid in question_ids
                        if oracle_lookup.get(qid, {}).get("question_type") == args.question_type]
    if args.limit:
        question_ids = question_ids[:args.limit]

    # API clients
    from openai import OpenAI
    embed_client = None
    scorer_client = None

    is_cohere = args.embed_model.startswith("embed-")
    if not is_cohere:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        embed_client = OpenAI(api_key=api_key)

    if "gemini" in args.answerer:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            print("GEMINI_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        scorer_client = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        scorer_client = embed_client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    embed_cache_dir = args.results_dir / "embed_cache"

    global _rate_limiter
    if args.scorer_rpm > 0:
        _rate_limiter = RateLimiter(args.scorer_rpm)

    print(f"Embedding: {args.embed_model}")
    print(f"Answerer: {args.answerer}")
    print(f"Retrieval: K={args.retrieval_k}, MR=±{args.message_range}")
    print(f"Questions: {len(question_ids)}")

    t_start = time.monotonic()
    write_lock = threading.Lock()
    counter = [0]

    def process_question(qid: str) -> dict | None:
        oracle = oracle_lookup.get(qid)
        if oracle is None:
            return None

        messages = load_messages(oracle)
        if not messages:
            return None
        msg_by_id = {m.id: m for m in messages}

        # Rank by embedding similarity
        ranking = embedding_rank(
            messages, oracle["question"], embed_client,
            embed_model=args.embed_model,
            cache_dir=embed_cache_dir, question_id=qid,
        )

        # Expand with message range
        hit_ids = ranking[:args.retrieval_k]
        if args.message_range > 0:
            max_id = max(m.id for m in messages)
            expanded = set()
            for oid in hit_ids:
                for offset in range(-args.message_range, args.message_range + 1):
                    nid = oid + offset
                    if 0 <= nid <= max_id:
                        expanded.add(nid)
            hit_ids = sorted(expanded)

        top_messages = [msg_by_id[oid] for oid in hit_ids if oid in msg_by_id]

        # Answer + judge
        llm_answer, correct = answer_and_judge(
            top_messages, oracle, scorer_client, model=args.answerer,
        )

        with write_lock:
            counter[0] += 1
            mark = "\u2705" if correct else "\u274c"
            print(f"[{counter[0]}/{len(question_ids)}] {qid}  {mark}", flush=True)

        return {
            "question_id": qid,
            "question_type": oracle["question_type"],
            "question": oracle["question"],
            "answer": str(oracle["answer"]),
            "n_messages": len(messages),
            "llm_answer": llm_answer,
            "correct": correct,
        }

    # Process
    if args.max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_question, qid): qid for qid in question_ids}
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
    else:
        results = [r for qid in question_ids if (r := process_question(qid))]

    results.sort(key=lambda r: r["question_id"])

    # Write JSONL
    with out_jsonl.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write hypotheses
    with out_hypotheses.open("w") as f:
        for r in results:
            f.write(json.dumps({"question_id": r["question_id"],
                                "hypothesis": r["llm_answer"]}) + "\n")

    # Generic summary
    elapsed = time.monotonic() - t_start
    n_correct = sum(1 for r in results if r["correct"])
    pct = n_correct / len(results) * 100 if results else 0

    print(f"\nGeneric accuracy: {n_correct}/{len(results)} = {pct:.1f}%")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Results: {out_jsonl}")
    print(f"Hypotheses: {out_hypotheses}")

    # Strict judge (optional)
    if args.judge:
        hypotheses = [{"question_id": r["question_id"], "hypothesis": r["llm_answer"]}
                      for r in results]
        eval_cache_path = out_hypotheses.with_suffix(".eval.jsonl")
        cache = _run_strict_judge(
            hypotheses, oracle_lookup, args.judge_model,
            eval_cache_path, max_workers=args.max_workers,
        )
        judge_label = f"openai/{args.judge_model}" if args.judge == "openai" else args.judge
        _display_strict_results(cache, oracle_lookup, judge_label)
    else:
        print(f"\nTo also run strict LongMemEval judge, add: --judge openai")


if __name__ == "__main__":
    main()
