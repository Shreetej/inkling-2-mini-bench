"""Agentic bug-fix test: a small multi-file repo with a seeded tiered-pricing bug.

The model gets two tools (bash, write_file) and a 30-turn budget. Every command runs under
macOS sandbox-exec: no network, writes only inside a fresh copy of the workspace.
Success = the visible tests AND a hidden test file pass, and the visible tests were not edited.

  python agent.py inkling-2-mini qwen3.8-27b ...   # labels from run.MODELS; 3 trials each
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from run import BASE, MODELS, OUT, http, unload_all

LOCAL = "127.0.0.1" in BASE

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("BENCH_TMP", tempfile.gettempdir())) / "inkling-bench"
SCRATCH.mkdir(parents=True, exist_ok=True)
MAX_TURNS = 30
MAX_WALL = 1200
TRIALS = 3
PROMPT = ("You are working in a Python repo (current directory). A customer reports that solar savings quotes are "
          "wrong, and `python3 -m pytest -q` fails. Find the root cause and fix the source code so all tests pass. "
          "Do not edit the tests. Use the tools to inspect files and run commands. When the tests pass, reply with "
          "a one-sentence summary of the fix and stop calling tools.")
TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command in the repo directory. Returns stdout+stderr.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Overwrite a file (path relative to the repo) with the given content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]


def sandboxed(ws: Path, cmd: str) -> str:
    try:
        r = subprocess.run(["sandbox-exec", "-f", str(HERE / "sandbox.sb"), "-D", f"WS={ws}", "/bin/bash", "-c", cmd],
                           cwd=ws, capture_output=True, text=True, timeout=60)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = "error: command timed out after 60s"
    return out[-6000:] or "(no output)"


def write_file(ws: Path, path: str, content: str) -> str:
    target = (ws / path).resolve()
    if not str(target).startswith(str(ws.resolve())):
        return "error: path outside repo"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


def verify(ws: Path, test_hash: str) -> dict:
    shutil.copy(HERE / "hidden_test.py", ws / "hidden_test.py")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ws,
                       capture_output=True, text=True, timeout=120)
    untouched = hashlib.sha256((ws / "test_quote.py").read_bytes()).hexdigest() == test_hash
    return {"pass": r.returncode == 0 and untouched, "tests_untouched": untouched,
            "pytest_tail": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]}


def trial(model: str, extra: dict) -> dict:
    ws = Path(tempfile.mkdtemp(prefix="agent-", dir=SCRATCH))
    shutil.copytree(HERE / "agent_ws", ws, dirs_exist_ok=True)
    test_hash = hashlib.sha256((ws / "test_quote.py").read_bytes()).hexdigest()
    msgs: list[dict] = [{"role": "user", "content": PROMPT}]
    t0 = time.monotonic()
    out_tokens = turns = tool_calls = 0
    commands: list[str] = []
    stop = "turn_limit"
    while turns < MAX_TURNS:
        if time.monotonic() - t0 > MAX_WALL:
            stop = "time_limit"
            break
        turns += 1
        try:
            r = http("POST", "/v1/chat/completions",
                     {"model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 16384, **extra})
        except Exception as e:  # noqa: BLE001
            stop = f"error: {e!r}"[:200]
            break
        out_tokens += r.get("usage", {}).get("completion_tokens", 0)
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls} if calls
                    else {"role": "assistant", "content": m.get("content") or ""})
        if not calls:
            stop = "answered"
            break
        for c in calls:
            tool_calls += 1
            fn = c["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if fn["name"] == "bash" and "command" in args:
                commands.append(args["command"])
                res = sandboxed(ws, args["command"])
            elif fn["name"] == "write_file" and {"path", "content"} <= set(args):
                commands.append(f"write_file {args['path']}")
                res = write_file(ws, args["path"], args["content"])
            else:
                res = f"error: bad tool call {fn.get('name')} {fn.get('arguments')!r}"[:500]
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": res})
    v = verify(ws, test_hash)
    repeats = len(commands) - len(set(commands))
    return {**v, "stop": stop, "turns": turns, "tool_calls": tool_calls, "out_tokens": out_tokens,
            "wall": round(time.monotonic() - t0, 1), "repeated_commands": repeats, "commands": commands,
            "final": msgs[-1].get("content", "")[:500], "billing_py": (ws / "solar/billing.py").read_text()}


def main() -> None:
    for arg in sys.argv[1:]:
        label, _, remote_model = arg.partition("=")  # remote: label=model-name
        model, extra = (remote_model, {}) if remote_model else MODELS[label][:2]
        path = OUT / f"agent-{label}.json"
        if path.exists():
            print("skip", label)
            continue
        if LOCAL:
            unload_all()
            http("POST", f"/v1/models/{model}/load")
        runs = []
        for i in range(TRIALS):
            r = trial(model, extra)
            runs.append(r)
            print(f"{label} #{i}: pass={r['pass']} stop={r['stop']} turns={r['turns']} wall={r['wall']}s "
                  f"tokens={r['out_tokens']} repeats={r['repeated_commands']} :: {r['pytest_tail']}", flush=True)
        path.write_text(json.dumps({"label": label, "model": model, "extra": extra, "runs": runs}, indent=1))
    if LOCAL:
        unload_all()


if __name__ == "__main__":
    main()
