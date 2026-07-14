#!/usr/bin/env bash
# Backfill attendance from Hikvision NVR footage for a live-stream gap. Invoked by
# the host agent (pipeline_agent.py, action=backfill). Fetches the missed window and
# reprocesses each recorded segment FAST (file source, NVDEC, unthrottled), stamped
# at its real recording time. See docs/BACKFILL.md.
#   run_backfill.sh <username> <channel> <start_iso> <end_iso> [gap_id]
set -euo pipefail

USER="${1:?username}"; CH="${2:?channel}"; START="${3:?start}"; END="${4:?end}"; GAP="${5:-}"
DS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASH_ENV=/home/meta/deploy/attendance-system/backend/.env
IMAGES_ROOT=/home/meta/deploy/test/data/company_images
IMG=deepstream-facepipe:latest
OUT="$DS/data/backfill/$USER"; mkdir -p "$OUT"

_set_gap() {  # $1 = status
  [ -n "$GAP" ] || return 0
  python3 - "$GAP" "$1" <<PY || true
import sys, os, psycopg2
from dotenv import load_dotenv; load_dotenv("$DASH_ENV")
c = psycopg2.connect(os.getenv("DATABASE_URL")); cur = c.cursor()
cur.execute("UPDATE stream_gaps SET status=%s WHERE id=%s", (sys.argv[2], int(sys.argv[1])))
c.commit(); c.close()
PY
}
trap '_set_gap failed' ERR

# 1) Resolve company + gallery folder + NVR creds + pipeline index (tab-separated).
IFS=$'\t' read -r CID APIKEY FOLDER IDX NHOST NPORT NUSER NPASS < <(python3 - "$USER" "$CH" <<PY
import sys, os, psycopg2
from dotenv import load_dotenv; load_dotenv("$DASH_ENV")
user, ch = sys.argv[1], int(sys.argv[2])
c = psycopg2.connect(os.getenv("DATABASE_URL")); cur = c.cursor()
cur.execute("SELECT company_id, api_key, company_image_folder FROM companies WHERE admin_username=%s", (user,))
comp = cur.fetchone()
cur.execute("SELECT admin_username FROM companies ORDER BY company_name")
idx = next((i for i, (x,) in enumerate(cur.fetchall()) if x == user), 0)
cur.execute('''SELECT cm.nvr_host, cm.nvr_port, cm.nvr_user, cm.nvr_password
               FROM cameras cm JOIN companies co ON cm.company_id = co.company_id::text
               WHERE co.admin_username=%s AND cm.nvr_channel=%s LIMIT 1''', (user, ch))
n = cur.fetchone(); c.close()
if not comp or not n or not n[0]:
    sys.exit("missing company or NVR config")
print(comp[0], comp[1] or "", comp[2] or user, idx, n[0], n[1] or 80, n[2] or "", n[3] or "", sep="\t")
PY
)

# DB env-file for the container (reuse the live one if present).
DB_ENV="$DS/config/companies/.db.env"
[ -f "$DB_ENV" ] || python3 - "$DB_ENV" <<PY
import os, sys
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv; load_dotenv("$DASH_ENV")
u = urlparse(os.getenv("DATABASE_URL"))
open(sys.argv[1], "w").write(
    f"DB_HOST={u.hostname or 'localhost'}\nDB_PORT={u.port or 5432}\n"
    f"DB_NAME={(u.path or '/facial_recognition_db').lstrip('/')}\n"
    f"DB_USER={unquote(u.username or 'postgres')}\nDB_PASSWORD={unquote(u.password or '')}\n")
PY

case "$FOLDER" in /*) GALLERY_SRC="$FOLDER" ;; *) GALLERY_SRC="$IMAGES_ROOT/$FOLDER" ;; esac

# 2) Fetch the missed window from the NVR.
mapfile -t SEGS < <(python3 "$DS/tools/nvr_fetch.py" --host "$NHOST" --port "$NPORT" \
  --user "$NUSER" --password "$NPASS" --channel "$CH" --start "$START" --end "$END" --out "$OUT")
if [ "${#SEGS[@]}" -eq 0 ] || [[ "${SEGS[0]:-}" != *"/"* ]]; then
  echo "no recordings for $USER ch$CH ($START..$END)"; _set_gap done; exit 0
fi
echo "fetched ${#SEGS[@]} segment(s) for $USER ch$CH"

# 3) Reprocess each segment (single-shot pipeline over the file, quits at EOS).
for line in "${SEGS[@]}"; do
  path="${line%%$'\t'*}"; rest="${line#*$'\t'}"; cstart="${rest%%$'\t'*}"
  [ -f "$path" ] || { echo "skip missing $path"; continue; }
  echo ">> backfill $path (clip_start=$cstart)"
  python3 "$DS/tools/gen_company_config.py" "$USER" --index "$IDX" \
      --backfill-uri "file://$path" --clip-start "$cstart"
  docker run --rm --runtime nvidia --gpus all --network host \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e COMPANY_ID="$CID" -e WEBHOOK_TOKEN="$APIKEY" \
    -e WEBHOOK_URL="http://localhost:5002/api/attendance/entry" -e PIPELINE_DISPLAY=0 \
    --env-file "$DB_ENV" \
    -v "$DS:/workspace" \
    -v "$GALLERY_SRC:/workspace/data/known_faces" \
    -v "$DS/config/companies/$USER.toml:/workspace/config/config_pipeline.toml" \
    -w /workspace "$IMG" bash /workspace/tools/pipeline_entry.sh \
    || echo "  segment run returned nonzero"
done
_set_gap done
echo "backfill complete: $USER ch$CH gap=$GAP"
