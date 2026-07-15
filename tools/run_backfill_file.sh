#!/usr/bin/env bash
# Reprocess an operator-UPLOADED video clip (offline, high-speed) — no NVR involved.
# Invoked by the host agent (pipeline_agent.py, action=backfill, payload.upload=true).
# Runs a single-shot pipeline over the file (file source, NVDEC, unthrottled), stamping
# attendance from the provided clip_start. Sibling of run_backfill.sh, which instead
# pulls the window from the Hikvision NVR. See docs/BACKFILL.md.
#   run_backfill_file.sh <username> <filename> <clip_start_iso> [gap_id]
set -euo pipefail

USER="${1:?username}"; FILENAME="${2:?filename}"; CLIP_START="${3:?clip_start}"; GAP="${4:-}"
DS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASH_ENV=/home/meta/deploy/attendance-system/backend/.env
IMAGES_ROOT=/home/meta/deploy/test/data/company_images
# The dashboard writes uploads here (CAPTURES_DIR/backfill_uploads, bind-mounted to host).
UPLOAD_DIR=/home/meta/deploy/attendance-system/captures/backfill_uploads
IMG=deepstream-facepipe:latest

# basename() the payload filename to prevent any path traversal.
FILE="$UPLOAD_DIR/$(basename "$FILENAME")"
EXT=".${FILE##*.}"; [ "$EXT" = ".$FILE" ] && EXT=".mp4"

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

[ -f "$FILE" ] || { echo "upload not found: $FILE"; _set_gap failed; exit 1; }

# Resolve company + gallery folder + pipeline index (tab-separated). No NVR needed.
IFS=$'\t' read -r CID APIKEY FOLDER IDX < <(python3 - "$USER" <<PY
import sys, os, psycopg2
from dotenv import load_dotenv; load_dotenv("$DASH_ENV")
user = sys.argv[1]
c = psycopg2.connect(os.getenv("DATABASE_URL")); cur = c.cursor()
cur.execute("SELECT company_id, api_key, company_image_folder FROM companies WHERE admin_username=%s", (user,))
comp = cur.fetchone()
cur.execute("SELECT admin_username FROM companies ORDER BY company_name")
idx = next((i for i, (x,) in enumerate(cur.fetchall()) if x == user), 0)
c.close()
if not comp:
    sys.exit("missing company")
print(comp[0], comp[1] or "", comp[2] or user, idx, sep="\t")
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

# Reprocess the single uploaded file (quits at EOS), stamped at clip_start. The file is
# mounted into the workspace; gen_company_config points the one source at it (live-source=0).
echo ">> backfill upload $FILE (clip_start=$CLIP_START)"
python3 "$DS/tools/gen_company_config.py" "$USER" --index "$IDX" \
    --backfill-uri "file:///workspace/data/upload_clip$EXT" --clip-start "$CLIP_START"
if docker run --rm --runtime nvidia --gpus all --network host \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e COMPANY_ID="$CID" -e WEBHOOK_TOKEN="$APIKEY" \
    -e WEBHOOK_URL="http://localhost:5002/api/attendance/entry" -e PIPELINE_DISPLAY=0 \
    --env-file "$DB_ENV" \
    -v "$DS:/workspace" \
    -v "$GALLERY_SRC:/workspace/data/known_faces" \
    -v "$FILE:/workspace/data/upload_clip$EXT:ro" \
    -v "$DS/config/companies/$USER.toml:/workspace/config/config_pipeline.toml" \
    -w /workspace "$IMG" bash /workspace/tools/pipeline_entry.sh; then
  _set_gap done
else
  echo "  run returned nonzero"; _set_gap failed
fi
# One-shot upload: drop the clip so captures/backfill_uploads doesn't grow unbounded.
rm -f "$FILE"
echo "backfill upload complete: $USER file=$(basename "$FILE") gap=$GAP"
