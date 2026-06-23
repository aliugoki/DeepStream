#!/usr/bin/env python3
"""
Print VisionTrack tenant + camera UUIDs as a ready-to-paste [visiontrack]
config block for config_pipeline.toml.

The VisionTrack identity feed (utils/visiontrack_publisher.py) needs, per
pipeline source, the matching VisionTrack `cameras.id` and `tenants.id` UUIDs.
This queries VisionTrack's Postgres and emits the [[visiontrack.cameras]]
entries so you only have to map each pipeline source_id to the right camera.

VisionTrack's DB must be running. Connection defaults match its
docker-compose (override via env): VT_DB_HOST, VT_DB_PORT, VT_DB_NAME,
VT_DB_USER, VT_DB_PASSWORD.

Usage:
    python3 tools/vt_uuids.py
"""
import os
import sys

import psycopg2

CFG = {
    "host": os.getenv("VT_DB_HOST", "localhost"),
    "port": os.getenv("VT_DB_PORT", "5432"),
    "dbname": os.getenv("VT_DB_NAME", "visiontrack"),
    "user": os.getenv("VT_DB_USER", "visiontrack"),
    "password": os.getenv("VT_DB_PASSWORD", "visiontrack_dev_password"),
}


def main():
    try:
        conn = psycopg2.connect(**CFG)
    except Exception as e:
        sys.exit(f"Could not connect to VisionTrack DB ({CFG['host']}:{CFG['port']}/"
                 f"{CFG['dbname']}): {e}\nIs the VisionTrack stack running?")

    with conn, conn.cursor() as cur:
        cur.execute("SELECT id, name FROM tenants ORDER BY name")
        tenants = cur.fetchall()
        cur.execute("SELECT id, name, tenant_id FROM cameras ORDER BY name")
        cameras = cur.fetchall()

    print("# --- VisionTrack tenants ---")
    for tid, name in tenants:
        print(f"#   {name}: {tid}")
    if not cameras:
        print("\n# No cameras enrolled in VisionTrack yet. Add cameras there first.")
        return

    print("\n# Paste into config/config_pipeline.toml, then set source_id to the")
    print("# matching [[sources]] index and flip [visiontrack].enabled = true.\n")
    for i, (cid, name, tid) in enumerate(cameras):
        print(f"[[visiontrack.cameras]]")
        print(f"source_id = {i}   # <-- map to the right [[sources]] index ({name})")
        print(f'camera_id = "{cid}"')
        print(f'tenant_id = "{tid}"')
        print()


if __name__ == "__main__":
    main()
