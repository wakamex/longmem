#!/usr/bin/env python3
"""Rank every message of a LongMemEval haystack with a local embedding model and write the ranking
for recall_compare.py.

    python3 embed_rank.py --kind colbert --model lightonai/LateOn --out results/rank_lateon.jsonl
    python3 embed_rank.py --kind dense --model lightonai/DenseOn --query-prompt query \
        --doc-prompt document --out results/rank_denseon.jsonl

--kind dense is one vector per message, ranked by cosine. --kind colbert is one vector per token,
ranked by MaxSim through PyLate. Messages are the same "[role] content" strings eval_retrieval.py
ranks, and each model truncates them at its own limit unless --max-tokens is given.
"""
import argparse, json, time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"))
    ap.add_argument("--question-type", default="multi-session")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--kind", choices=("dense", "colbert"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--query-prompt", help="prompt name for questions")
    ap.add_argument("--doc-prompt", help="prompt name for messages")
    ap.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--max-tokens", type=int, help="override the model's message length limit")
    ap.add_argument("--text-only", action="store_true", help="skip a multimodal model's vision and audio encoders")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    data = [q for q in json.loads(a.dataset.read_text()) if q["question_type"] == a.question_type][: a.limit]
    haystacks = [[f"[{m['role']}] {m['content']}" for s in q["haystack_sessions"] for m in s] for q in data]

    t0 = time.time()
    if a.kind == "colbert":
        from pylate import models, rank
        model = models.ColBERT(model_name_or_path=a.model, device="cuda", **({"document_length": a.max_tokens} if a.max_tokens else {}))
        model.to(dtype)
        limit = model.document_length
    else:
        from sentence_transformers import SentenceTransformer
        kw = {"config_kwargs": {"vision_config": None, "audio_config": None}} if a.text_only else {}
        model = SentenceTransformer(a.model, device="cuda", model_kwargs={"torch_dtype": dtype}, **kw)
        if a.max_tokens:
            model.max_seq_length = a.max_tokens
        limit = model.max_seq_length
    print(f"loaded {a.model} in {time.time() - t0:.0f}s, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, "
          f"message limit {limit} tokens, {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

    t0 = time.time()
    rows = []
    if a.kind == "colbert":
        for q, msgs in zip(data, haystacks):
            qe = model.encode([q["question"]], is_query=True, prompt_name=a.query_prompt, batch_size=1, show_progress_bar=False)
            de = model.encode(msgs, is_query=False, prompt_name=a.doc_prompt, batch_size=a.batch_size, show_progress_bar=False)
            out = rank.rerank(documents_ids=[list(range(len(msgs)))], queries_embeddings=qe, documents_embeddings=[de])
            rows.append([e["id"] for e in out[0]])
    else:
        # sessions recur across haystacks, so each distinct message is encoded once
        texts = sorted({m for msgs in haystacks for m in msgs})
        enc = lambda x, prompt: model.encode(x, prompt_name=prompt, batch_size=a.batch_size, convert_to_tensor=True,
                                             normalize_embeddings=True, show_progress_bar=True).float()
        emb = enc(texts, a.doc_prompt)
        if torch.isnan(emb).any():
            raise SystemExit(f"{int(torch.isnan(emb).any(1).sum())} messages embedded as NaN; rerun with --dtype float32")
        at = {t: i for i, t in enumerate(texts)}
        qemb = enc([q["question"] for q in data], a.query_prompt)
        print(f"{len(texts)} distinct messages of {sum(map(len, haystacks))}", flush=True)
        for k, msgs in enumerate(haystacks):
            scores = emb[[at[m] for m in msgs]] @ qemb[k]
            rows.append(torch.argsort(scores, descending=True, stable=True).tolist())
    secs = time.time() - t0
    print(f"ranked {len(data)} questions in {secs:.0f}s ({secs / len(data):.2f}s each), peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    with a.out.open("w") as f:
        for q, ranked in zip(data, rows):
            f.write(json.dumps({"question_id": q["question_id"], "ranked": ranked}) + "\n")


if __name__ == "__main__":
    main()
