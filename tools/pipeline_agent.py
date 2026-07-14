#!/usr/bin/env python3
"""
Host-side pipeline agent.

Runs on the GPU host (where docker + the deepstream repo live). Each cycle it:
  1. polls the dashboard for launch/stop/restart jobs and executes them, and
  2. pushes a heartbeat: deepstream-* container state/logs + GPU telemetry.

This keeps docker OFF the web app — the dashboard only enqueues bounded jobs and
reads the heartbeat; the agent (holding AGENT_TOKEN) is the only thing that runs
docker. The control panel uses the heartbeat for container/GPU info and reads
live frame-flow health straight off each pipeline's :9108+index endpoint.

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


def run(cmd, timeout=600):
    try:
        p = subprocess.run(cmd, cwd=DS, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-1800:]
    except Exception as e:
        return 1, f"agent exec error: {e}"


def report(jid, status, log=None):
    try:
        requests.post(f"{DASH}/api/pipeline/agent/jobs/{jid}",
                      json={"token": TOKEN, "status": status, "log": log}, timeout=10)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Heartbeat — container state/logs + GPU telemetry
# --------------------------------------------------------------------------- #
def docker_containers():
    rc, out = run(["docker", "ps", "-a", "--filter", "name=deepstream-",
                   "--format", "{{.Names}}\t{{.State}}\t{{.Status}}\t{{.RunningFor}}"], timeout=20)
    items = []
    if rc != 0:
        return items
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].startswith("deepstream-"):
            continue
        name = parts[0]
        _, logs = run(["docker", "logs", "--tail", "40", name], timeout=20)
        items.append({
            "name": name, "state": parts[1], "status": parts[2],
            "running_for": parts[3] if len(parts) > 3 else "",
            "log": logs[-1600:],
        })
    return items


def gpu_stats():
    rc, out = run(["nvidia-smi",
                   "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                   "--format=csv,noheader,nounits"], timeout=15)
    gpus = []
    if rc != 0:
        return gpus
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6:
            continue
        try:
            gpus.append({"index": int(f[0]), "name": f[1],
                         "mem_used": int(float(f[2])), "mem_total": int(float(f[3])),
                         "util": int(float(f[4])), "temp": int(float(f[5]))})
        except ValueError:
            pass
    return gpus


def heartbeat():
    try:
        requests.post(f"{DASH}/api/pipeline/agent/heartbeat",
                      json={"token": TOKEN, "containers": docker_containers(), "gpus": gpu_stats()},
                      timeout=15)
    except Exception as e:
        print("heartbeat error:", e)


# --------------------------------------------------------------------------- #
# Job execution
# --------------------------------------------------------------------------- #
def execute(j):
    u, idx, act = j["username"], int(j["index"]), j["action"]
    if not SAFE.match(u):
        report(j["id"], "failed", "unsafe username")
        return
    report(j["id"], "running")
    if act in ("start", "restart"):
        # run_company_pipeline.sh re-provisions config from the DB cameras and
        # recreates the container (docker rm -f first), so it doubles as restart.
        rc, log = run(["bash", "tools/run_company_pipeline.sh", u, str(idx)])
    elif act == "stop":
        rc, log = run(["docker", "rm", "-f", f"deepstream-{u}"], timeout=60)
    elif act == "mediamtx_sync":
        # Regenerate the MediaMTX paths block (per-company recordDeleteAfter from the
        # DB). No pipeline touched; MediaMTX hot-reloads the rewritten config file.
        rc, log = run(["python3", "tools/gen_mediamtx_paths.py"], timeout=60)
    elif act == "backfill":
        # Recover a live-stream gap: fetch the missed window from the NVR and
        # reprocess it fast, stamped at its recording time (run_backfill.sh).
        import json as _json
        p = _json.loads(j.get("payload") or "{}")
        rc, log = run(["bash", "tools/run_backfill.sh", u, str(p.get("channel", "")),
                       str(p.get("start", "")), str(p.get("end", "")),
                       str(p.get("gap_id", ""))], timeout=3600)
    else:
        rc, log = 1, "unknown action"
    report(j["id"], "done" if rc == 0 else "failed", log)


def main():
    print(f"pipeline agent → {DASH} (repo {DS})")
    while True:
        try:
            jobs = requests.get(f"{DASH}/api/pipeline/agent/jobs",
                                params={"token": TOKEN}, timeout=10).json()
            for j in jobs:
                execute(j)
        except Exception as e:
            print("poll error:", e)
        heartbeat()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
