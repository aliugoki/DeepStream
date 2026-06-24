#!/usr/bin/env python3
"""
Host-side pipeline agent.

Runs on the GPU host (where docker + the deepstream repo live). Polls the
dashboard for launch/stop jobs and executes them, reporting status back. This
keeps docker OFF the web app — the dashboard only enqueues bounded jobs
(start/stop a known company's pipeline), the agent (holding AGENT_TOKEN) is the
only thing that can run them.

Env:
    DASHBOARD_URL   default http://localhost:5002
    AGENT_TOKEN     must match the dashboard's AGENT_TOKEN
Run:
    AGENT_TOKEN=... python3 tools/pipeline_agent.py
    (or as a systemd service / nohup background process)
"""
import os
import re
import time
import subprocess

import requests

DASH = os.getenv("DASHBOARD_URL", "http://localhost:5002").rstrip("/")
TOKEN = os.getenv("AGENT_TOKEN", "")
DS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE = re.compile(r"^[A-Za-z0-9_-]+$")     # bound the username to a safe charset
POLL_SECONDS = 5

if not TOKEN:
    raise SystemExit("AGENT_TOKEN env is required")


def run(cmd):
    try:
        p = subprocess.run(cmd, cwd=DS, capture_output=True, text=True, timeout=600)
        return p.returncode, (p.stdout + p.stderr)[-1800:]
    except Exception as e:
        return 1, f"agent exec error: {e}"


def report(jid, status, log=None):
    try:
        requests.post(f"{DASH}/api/pipeline/agent/jobs/{jid}",
                      json={"token": TOKEN, "status": status, "log": log}, timeout=10)
    except Exception:
        pass


def main():
    print(f"pipeline agent → {DASH} (repo {DS})")
    while True:
        try:
            jobs = requests.get(f"{DASH}/api/pipeline/agent/jobs",
                                params={"token": TOKEN}, timeout=10).json()
            for j in jobs:
                u, idx, act = j["username"], int(j["index"]), j["action"]
                if not SAFE.match(u):
                    report(j["id"], "failed", "unsafe username"); continue
                report(j["id"], "running")
                if act == "start":
                    rc, log = run(["bash", "tools/run_company_pipeline.sh", u, str(idx)])
                elif act == "stop":
                    rc, log = run(["docker", "rm", "-f", f"deepstream-{u}"])
                else:
                    rc, log = 1, "unknown action"
                report(j["id"], "done" if rc == 0 else "failed", log)
        except Exception as e:
            print("poll error:", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
