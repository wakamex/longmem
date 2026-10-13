#!/usr/bin/env python3
"""Serve search over endchat profiles for eval_endchat_tools.py: rank a profile's messages with
ColBERT-Zero and return the best ones whole.

    .venv/bin/python search_server.py [--port 8765]

POST / with {"profile": DIR, "query": TEXT} answers {"ids": [...], "text": ...}. A profile's
messages are read from DIR/messages.jsonl and embedded on its first query. Messages are ranked as
the "[role] content" strings embed_rank.py ranks, with the model's query and document prompts.
"""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pylate import models, rank

MODEL = "lightonai/ColBERT-Zero"
TOP = 10
lock = threading.Lock()
cache = {}


def search(model, profile, query):
    with lock:
        if profile not in cache:
            msgs = [json.loads(l) for l in (Path(profile) / "messages.jsonl").read_text().splitlines()]
            docs = [f"[{'user' if m['kind'] == 'user' else 'assistant'}] {m['text']}" for m in msgs]
            cache[profile] = msgs, model.encode(docs, is_query=False, prompt_name="document", batch_size=32, show_progress_bar=False)
        msgs, de = cache[profile]
        qe = model.encode([query], is_query=True, prompt_name="query", batch_size=1, show_progress_bar=False)
        out = rank.rerank(documents_ids=[list(range(len(msgs)))], queries_embeddings=qe, documents_embeddings=[de])
    best = [msgs[e["id"]] for e in out[0][:TOP]]
    return {"ids": [m["i"] for m in best], "text": "\n\n".join(f"{m['i']} | {m['date'][:10]} | {m['kind']}: {m['text']}" for m in best) + "\n"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18765)
    a = ap.parse_args()
    model = models.ColBERT(model_name_or_path=MODEL, device="cuda")
    model.half()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(search(model, req["profile"], req["query"])).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    print(f"serving {MODEL} on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
