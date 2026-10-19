#!/usr/bin/python3
"""endchat turn command for search("QUERY"): ask search_server.py and print the messages.

    search_tool.py PROFILE_DIR ARG

search("QUERY", k=N) returns the N best messages (at most 200); without k, PROFILE_DIR/search_k gives N, and 10 when
it is absent. PROFILE_DIR/search_chars, when present, makes the default k 200 and keeps only the best messages
that fit in that many characters. PROFILE_DIR/search_mode, when present, holds "whole", "excerpt" or "reader". With "reader" the
whole messages go to Claude Haiku with the user's message, and what it quotes from them is printed
instead (prompts/search_reader/reader_*.md). Each call is appended to PROFILE_DIR/tool_calls.jsonl
with the ids of the messages shown, and for a reader call the ids searched and Haiku's token usage.
"""
import json, os, re, subprocess, sys, tempfile, urllib.request
from pathlib import Path

URL = os.environ.get("SEARCH_SERVER", "http://127.0.0.1:18765")
PROMPTS = Path(__file__).resolve().parent / "prompts" / "search_reader"


def read(profile, query, found):
    """What Haiku quotes from `found` for the user's latest message, and the call's usage."""
    msgs = [json.loads(l) for f in sorted(Path(profile, "main").glob("*.jsonl")) for l in f.read_text().split("\n") if l.startswith("{")]
    question = max((m for m in msgs if m["kind"] == "user"), key=lambda m: m["i"])["text"]
    aop = re.search(r'aop_bin = "(.*)"', Path(profile, "config.toml").read_text()).group(1)
    with tempfile.NamedTemporaryFile("w", dir=profile, suffix=".txt", delete=False) as f:
        f.write((PROMPTS / "reader_prompt.md").read_text().format(question=question, query=query, messages=found))
    try:
        p = subprocess.run([aop, "run", "--agent", "claude", "--model", (Path(profile, "reader_model").read_text().strip() if Path(profile, "reader_model").exists() else "haiku"), "--effort", "medium", "--profile", "sealed", "--no-web",
                            "--tools", "", "--system-prompt-file", str(PROMPTS / "reader_system.md"), "--prompt-file", f.name,
                            "--timeout", "300", "--json"], capture_output=True, text=True, cwd=profile)
        r = json.loads(p.stdout)
        if not r.get("succeeded"):
            sys.exit(f"the reader failed: {r.get('error')}")
        subprocess.run([aop, "cleanup", r["run_id"]], capture_output=True, cwd=profile)
        return r["final_message"].strip() + "\n", r.get("usage") or {}
    finally:
        os.unlink(f.name)


def main():
    profile, arg = sys.argv[1], sys.argv[2].strip()
    k_file = os.path.join(profile, "search_k")
    chars_file = os.path.join(profile, "search_chars")
    chars = int(open(chars_file).read()) if os.path.exists(chars_file) else None
    k = int(open(k_file).read()) if os.path.exists(k_file) else 200 if chars else 10
    m = re.match(r"""^(.*?)\s*,\s*k\s*=\s*(\d+)$""", arg, re.S)
    if m:
        arg, k = m.group(1).strip(), min(int(m.group(2)), 200)
    query = arg[1:-1] if len(arg) > 1 and arg[0] == arg[-1] and arg[0] in "\"'" else arg
    if not query.strip():
        sys.exit('usage: search("QUERY") or search("QUERY", k=N)')
    mode_file = os.path.join(profile, "search_mode")
    mode = open(mode_file).read().strip() if os.path.exists(mode_file) else "whole"
    req = urllib.request.Request(URL, json.dumps({"profile": profile, "query": query, "mode": "whole" if mode == "reader" else mode, "k": k, "chars": chars}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    call = {"tool": "search", "arg": query, "ids": out["ids"]}
    if mode == "reader":
        out["text"], call["reader_usage"] = read(profile, query, out["text"])
        quoted = {int(m) for m in re.findall(r"^(\d+) \|", out["text"], re.M)}
        call["searched"], call["ids"] = out["ids"], [i for i in out["ids"] if i in quoted]
    call["chars"] = len(out["text"])
    with open(os.path.join(profile, "tool_calls.jsonl"), "a") as f:
        f.write(json.dumps(call) + "\n")
    sys.stdout.write(out["text"])


if __name__ == "__main__":
    main()
