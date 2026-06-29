#!/usr/bin/env python3
"""
Generate a company-specific pipeline config from the dashboard DB.

Reads the company's cameras (managed in the dashboard) and writes
config/companies/<admin_username>.toml with that company's [[sources]] and
unique RTSP/health ports. COMPANY_ID + api_key are passed at launch via env
(see run_company_pipeline.sh), not baked into the config.

Usage:
    python3 tools/gen_company_config.py <admin_username> [--index N]
"""
import os
import sys
import argparse

import toml
import psycopg2
from dotenv import load_dotenv

load_dotenv("/home/meta/deploy/attendance-system/backend/.env")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username")
    ap.add_argument("--index", type=int, default=0,
                    help="port offset so multiple companies don't collide (RTSP 8555+N, health 9108+N)")
    args = ap.parse_args()

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("SELECT company_id, company_name FROM companies WHERE admin_username=%s", (args.username,))
    row = cur.fetchone()
    if not row:
        sys.exit(f"No company with admin_username='{args.username}'")
    company_id, company_name = str(row[0]), row[1]
    cur.execute("""SELECT name, type, rtsp_url FROM cameras
                   WHERE company_id=%s AND enabled=TRUE AND rtsp_url IS NOT NULL AND rtsp_url<>''
                   ORDER BY name""", (company_id,))
    cams = cur.fetchall()
    conn.close()
    if not cams:
        print(f"WARNING: no enabled cameras with an RTSP url for {company_name}. "
              f"Add them in the dashboard Cameras page.")

    # Start from the existing config as a template (inherits model paths/sections).
    cfg = toml.load(os.path.join(BASE, "config", "config_pipeline.toml"))
    cfg["pipeline"]["num_sources"] = max(len(cams), 1)
    cfg["pipeline"]["muxer_batch_size"] = max(len(cams), 1)
    cfg["pipeline"]["known_face_dir"] = "/workspace/data/known_faces"
    cfg["pipeline"]["health_port"] = 9108 + args.index
    # Local display: render the annotated output in a window on the host's
    # monitor. The launcher sets PIPELINE_DISPLAY=0 when no X server is reachable
    # so the pipeline falls back to fakesink instead of crashing on nveglglessink.
    cfg["pipeline"]["display"] = int(os.environ.get("PIPELINE_DISPLAY", "1"))
    cfg.setdefault("streammux", {})["batch-size"] = max(len(cams), 1)

    # One annotated RTSP mount per camera: /cam0, /cam1, … on this company's
    # RTSP port (8555+index). MediaMTX pulls each into a "{username}_cam{i}"
    # path for browser HLS/WebRTC (see tools/gen_mediamtx_paths.py). UDP ports
    # are offset by index so concurrently-running company pipelines don't clash.
    n = max(len(cams), 1)
    rs = cfg.setdefault("rtsp_server", {})
    rs["port"] = 8555 + args.index
    rs["enable_rtsp_streaming"] = True
    rs["codec"] = rs.get("codec", "H264")
    rs["udpsink-host"] = "127.0.0.1"
    rs["mount-points"] = [f"/cam{i}" for i in range(n)]
    rs["udpsink-ports"] = [5400 + args.index * 16 + i for i in range(n)]
    rs.pop("mount-point", None)               # remove the single-mount fallback
    rs.pop("udpsink-port", None)

    cfg["sources"] = [{
        "id": c[0], "uri": c[2], "type": c[1] or "general",
        "num-retry": 5, "rtsp-reconnect-interval-sec": 10, "latency": 200,
    } for c in cams]

    out_dir = os.path.join(BASE, "config", "companies")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.username}.toml")
    with open(out, "w") as f:
        f.write(f"# Generated for {company_name} ({args.username}). Do not hand-edit; re-run the generator.\n")
        toml.dump(cfg, f)
    print(f"Wrote {out}")
    print(f"  company: {company_name}  cameras: {len(cams)}  "
          f"rtsp_port: {8555 + args.index}  health_port: {9108 + args.index}")
    for i, c in enumerate(cams):
        print(f"   - cam{i} {c[0]} ({c[1]}): {c[2]}  ->  rtsp://127.0.0.1:{8555 + args.index}/cam{i}")


if __name__ == "__main__":
    main()
