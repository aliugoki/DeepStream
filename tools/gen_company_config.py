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

import json
import toml
import psycopg2
from dotenv import load_dotenv

load_dotenv("/home/meta/deploy/attendance-system/backend/.env")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username")
    ap.add_argument("--backfill-uri", help="reprocess this file:// clip instead of live cameras")
    ap.add_argument("--clip-start", help="ISO datetime the clip starts (stamps attendance)")
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
    cur.execute("""SELECT name, type, rtsp_url, detection_area FROM cameras
                   WHERE company_id=%s AND enabled=TRUE AND rtsp_url IS NOT NULL AND rtsp_url<>''
                   ORDER BY name""", (company_id,))
    cams = cur.fetchall()
    # Per-company recognition tuning (tolerant of the column not existing yet, e.g.
    # if the dashboard migration hasn't run — then we keep the template defaults).
    rec = None
    try:
        cur.execute("SELECT rec_threshold, rec_margin, rec_min_votes "
                    "FROM tenant_settings WHERE company_id=%s", (company_id,))
        rec = cur.fetchone()
    except Exception:
        rec = None
    conn.close()
    if not cams:
        print(f"WARNING: no enabled cameras with an RTSP url for {company_name}. "
              f"Add them in the dashboard Cameras page.")

    # Start from the existing config as a template (inherits model paths/sections).
    # Fall back to the committed .example if the live template is missing or empty
    # (e.g. clobbered by a container mount) so we never generate a config with no
    # [pipeline] section.
    tmpl = os.path.join(BASE, "config", "config_pipeline.toml")
    if not os.path.exists(tmpl) or os.path.getsize(tmpl) == 0:
        tmpl = os.path.join(BASE, "config", "config_pipeline.example.toml")
        print(f"WARNING: config_pipeline.toml missing/empty; using {os.path.basename(tmpl)}")
    cfg = toml.load(tmpl)
    if "pipeline" not in cfg:
        sys.exit(f"Template {tmpl} has no [pipeline] section — cannot generate config.")
    cfg["pipeline"]["num_sources"] = max(len(cams), 1)
    cfg["pipeline"]["muxer_batch_size"] = max(len(cams), 1)
    cfg["pipeline"]["known_face_dir"] = "/workspace/data/known_faces"
    cfg["pipeline"]["health_port"] = 9108 + args.index
    # Local display: render the annotated output in a window on the host's
    # monitor. The launcher sets PIPELINE_DISPLAY=0 when no X server is reachable
    # so the pipeline falls back to fakesink instead of crashing on nveglglessink.
    cfg["pipeline"]["display"] = int(os.environ.get("PIPELINE_DISPLAY", "1"))
    # Per-company face-recognition tuning from tenant policy (unset -> template
    # defaults). main_enterprise reads rec_threshold / rec_margin / track_min_votes.
    if rec:
        if rec[0] is not None:
            cfg["pipeline"]["rec_threshold"] = float(rec[0])
        if rec[1] is not None:
            cfg["pipeline"]["rec_margin"] = float(rec[1])
        if rec[2] is not None:
            cfg["pipeline"]["track_min_votes"] = int(rec[2])
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
    # Keep the singular keys too: main_udp.add_udp_rtsp_branches reads the plural
    # lists, but its `.get("...-ports", [int(rtsp_cfg["udpsink-port"])])` default
    # is evaluated eagerly and KeyErrors if the singular keys are absent.
    rs["mount-point"] = rs["mount-points"][0]
    rs["udpsink-port"] = rs["udpsink-ports"][0]

    def _area(c):
        """Per-camera detection zone: a polygon of [x,y] points normalized 0..1,
        stored as a JSON string in cameras.detection_area. Empty -> whole frame."""
        raw = c[3] if len(c) > 3 else None
        if not raw:
            return None
        try:
            pts = raw if isinstance(raw, list) else json.loads(raw)
            pts = [[float(x), float(y)] for x, y in pts]
            return pts if len(pts) >= 3 else None
        except Exception:
            return None

    cfg["sources"] = []
    for c in cams:
        src = {"id": c[0], "uri": c[2], "type": c[1] or "general",
               "num-retry": 5, "rtsp-reconnect-interval-sec": 10, "latency": 200}
        area = _area(c)
        if area:
            src["detection_area"] = area
        cfg["sources"].append(src)

    # BACKFILL: reprocess a single recorded clip (file source) instead of the live
    # cameras, stamped at its recording time. Runs unthrottled (NVDEC); to sample
    # frames for >24x-realtime, raise `interval=` in config_yolo.txt.
    if args.backfill_uri:
        cam0 = cfg["sources"][0] if cfg["sources"] else {"type": "entrance"}
        one = {"id": cam0.get("id", "backfill"), "uri": args.backfill_uri,
               "type": cam0.get("type", "entrance")}
        if "detection_area" in cam0:
            one["detection_area"] = cam0["detection_area"]
        cfg["sources"] = [one]
        cfg["pipeline"]["num_sources"] = 1
        cfg["pipeline"]["muxer_batch_size"] = 1
        cfg.setdefault("streammux", {})["batch-size"] = 1
        cfg["streammux"]["live-source"] = 0
        cfg.setdefault("rtsp_server", {})["enable_rtsp_streaming"] = False
        if args.clip_start:
            cfg.setdefault("backfill", {})["clip_start"] = args.clip_start

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
