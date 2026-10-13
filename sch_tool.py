#!/home/mihai/.local/share/uv/tools/search-clanker-history/bin/python
"""endchat turn command for `sch PATTERN`: run sch itself over the profile's messages.

    sch_tool.py PROFILE_DIR ARG

The messages are in PROFILE_DIR/sch/projects/chat/endchat.jsonl as a Claude Code session, which
sch reads through CLAUDE_CONFIG_DIR. Only the options named in the prompt are accepted. Each call
is appended to PROFILE_DIR/tool_calls.jsonl with the ids of the messages sch printed.
"""
import argparse, contextlib, io, json, os, shlex, sys

USAGE = "usage: sch [-t user|assistant] [-A N] [-B N] [-n N] [-C CHARS] [--since TIME] [--until TIME] PATTERN"


def parse(arg):
    p = argparse.ArgumentParser(prog="sch", add_help=False, allow_abbrev=False, exit_on_error=False)
    p.add_argument("pattern")
    p.add_argument("-t", "--type", choices=["user", "assistant"])
    for flag in ("-A", "-B", "-n", "-C"):
        p.add_argument(flag, type=int)
    p.add_argument("--since")
    p.add_argument("--until")
    a, extra = p.parse_known_args(shlex.split(arg))
    if extra:
        raise ValueError(f"unknown arguments: {' '.join(extra)}")
    argv = [a.pattern]
    for flag, v in (("-t", a.type), ("-A", a.A), ("-B", a.B), ("-n", a.n), ("-C", a.C), ("--since", a.since), ("--until", a.until)):
        if v is not None:
            argv += [flag, str(v)]
    return argv


def main():
    profile, arg = sys.argv[1], sys.argv[2]
    try:
        argv = parse(arg)
    except (argparse.ArgumentError, ValueError, SystemExit) as e:
        sys.exit(f"{e}\n{USAGE}")
    os.environ["CLAUDE_CONFIG_DIR"] = os.path.join(profile, "sch")
    from search_clanker_history import cli

    ids_of = {}
    with open(os.path.join(profile, "sch", "projects", "chat", "endchat.jsonl")) as f:
        for line in f:
            m = json.loads(line)
            ids_of.setdefault(cli.classify_line(m)[2], []).append(m["i"])
    shown = []
    cut = cli.truncate_around_match

    def recording(text, pattern, chars):  # called once for every message sch prints
        shown.extend(ids_of.get(text, []))
        return cut(text, pattern, chars)

    cli.truncate_around_match = recording
    sys.argv = ["sch", "--agent", "claude", "--no-color", *argv]
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli.main()
    except SystemExit as e:
        if e.code:
            sys.exit(f"{err.getvalue()}{USAGE}")
    with open(os.path.join(profile, "tool_calls.jsonl"), "a") as f:
        f.write(json.dumps({"tool": "sch", "arg": arg, "ids": sorted(set(shown)), "chars": len(out.getvalue())}) + "\n")
    sys.stdout.write(out.getvalue())


if __name__ == "__main__":
    main()
