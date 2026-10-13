#!/usr/bin/env python3
"""Serve search over endchat profiles for eval_endchat_tools.py: rank a profile's messages with
ColBERT-Zero and return the best ones, whole or cut to the sentences that match.

    .venv/bin/python search_server.py [--port 18765]

POST / with {"profile": DIR, "query": TEXT, "mode": "whole" | "excerpt"} answers
{"ids": [...], "text": ...}. A profile's messages are read from DIR/messages.jsonl and embedded on
its first query, as the "[role] content" strings embed_rank.py ranks. Profiles with the same
directory name hold the same chat and share one index. See excerpt.py for the ranking and the cut.
"""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import excerpt

TOP = 10
SHARE, NEIGHBORS = 0.9, 1  # chosen on the single-session-assistant questions
lock = threading.Lock()
cache = {}


def search(model, profile, query, mode):
    with lock:
        key = Path(profile).name
        if key not in cache:
            msgs = [json.loads(l) for l in (Path(profile) / "messages.jsonl").read_text().splitlines()]
            roles = ["user" if m["kind"] == "user" else "assistant" for m in msgs]
            cache[key] = msgs, excerpt.Index(model, [f"[{r}] {m['text']}" for r, m in zip(roles, msgs)], [len(r) + 3 for r in roles])
        msgs, index = cache[key]
        hits, q = index.search(query, TOP)
        shown = index.excerpts(hits, q, share=SHARE, neighbors=NEIGHBORS) if mode == "excerpt" else {k: msgs[k]["text"] for k in hits}
    text = "\n\n".join(f"{msgs[k]['i']} | {msgs[k]['date'][:10]} | {msgs[k]['kind']}: {shown[k]}" for k in hits) + "\n"
    return {"ids": [msgs[k]["i"] for k in hits], "text": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18765)
    a = ap.parse_args()
    model = excerpt.load()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(search(model, req["profile"], req["query"], req.get("mode", "whole"))).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    print(f"serving {excerpt.MODEL} on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
