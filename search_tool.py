#!/usr/bin/python3
"""endchat turn command for search("QUERY"): ask search_server.py and print the messages.

    search_tool.py PROFILE_DIR ARG

Each call is appended to PROFILE_DIR/tool_calls.jsonl with the ids of the messages returned.
"""
import json, os, sys, urllib.request

URL = os.environ.get("SEARCH_SERVER", "http://127.0.0.1:18765")


def main():
    profile, arg = sys.argv[1], sys.argv[2].strip()
    query = arg[1:-1] if len(arg) > 1 and arg[0] == arg[-1] and arg[0] in "\"'" else arg
    if not query.strip():
        sys.exit('usage: search("QUERY")')
    req = urllib.request.Request(URL, json.dumps({"profile": profile, "query": query}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    with open(os.path.join(profile, "tool_calls.jsonl"), "a") as f:
        f.write(json.dumps({"tool": "search", "arg": query, "ids": out["ids"], "chars": len(out["text"])}) + "\n")
    sys.stdout.write(out["text"])


if __name__ == "__main__":
    main()
