#!/usr/bin/env bash
# Launch a company-specific DeepStream pipeline container.
#   ./run_company_pipeline.sh <admin_username> [index] [webhook_base]
# Resolves COMPANY_ID + api_key from the dashboard DB, mounts that company's
# gallery + generated config, and runs main_enterprise.py.
set -euo pipefail

USER="${1:?usage: run_company_pipeline.sh <admin_username> [index] [webhook_base]}"
INDEX="${2:-0}"
WEBHOOK_BASE="${3:-http://localhost:5002}"
DS=/home/meta/deploy/deepstream
IMG=nvcr.io/nvidia/deepstream:7.1-gc-triton-devel

# 0) local-display detection. nveglglessink needs a reachable X server; if there
#    isn't one, fall back to fakesink (PIPELINE_DISPLAY=0) so we don't crash.
DISPLAY_VAL="${DISPLAY:-:0}"
DISP_NUM="${DISPLAY_VAL#*:}"; DISP_NUM="${DISP_NUM%%.*}"
DISPLAY_ARGS=()
if [ -S "/tmp/.X11-unix/X${DISP_NUM}" ]; then
  export PIPELINE_DISPLAY=1
  XAUTH="${XAUTHORITY:-$HOME/.Xauthority}"
  DISPLAY_ARGS=(-e DISPLAY="$DISPLAY_VAL" -v /tmp/.X11-unix:/tmp/.X11-unix)
  [ -f "$XAUTH" ] && DISPLAY_ARGS+=(-e XAUTHORITY=/root/.Xauthority -v "$XAUTH:/root/.Xauthority:ro")
  command -v xhost >/dev/null 2>&1 && DISPLAY="$DISPLAY_VAL" xhost +local:root >/dev/null 2>&1 || true
  echo "local display: ON ($DISPLAY_VAL)"
else
  export PIPELINE_DISPLAY=0
  echo "local display: OFF (no X server at $DISPLAY_VAL) -> fakesink"
fi

# 1) generate the company config from the DB cameras
python3 "$DS/tools/gen_company_config.py" "$USER" --index "$INDEX"

# record this company's pipeline index so the MediaMTX path generator knows
# which RTSP port (8555+index) to pull each camera's annotated stream from.
python3 - "$USER" "$INDEX" <<'PY'
import json, os, sys
p = "/home/meta/deploy/deepstream/config/companies/_indices.json"
d = {}
if os.path.exists(p):
    try: d = json.load(open(p))
    except Exception: d = {}
d[sys.argv[1]] = int(sys.argv[2])
json.dump(d, open(p, "w"), indent=2)
PY

# 2) resolve company_id, gallery folder, api_key
read -r CID FOLDER APIKEY < <(python3 - "$USER" <<'PY'
import sys, os, psycopg2
from dotenv import load_dotenv
load_dotenv('/home/meta/deploy/attendance-system/backend/.env')
c = psycopg2.connect(os.getenv('DATABASE_URL')); cur = c.cursor()
cur.execute("SELECT company_id, company_image_folder, api_key FROM companies WHERE admin_username=%s", (sys.argv[1],))
r = cur.fetchone(); print(r[0], r[1] or sys.argv[1], r[2]); c.close()
PY
)
echo "company_id=$CID folder=$FOLDER rtsp=$((8555+INDEX)) health=$((9108+INDEX))"

# 3) launch (one container per company)
docker rm -f "deepstream-$USER" 2>/dev/null || true
docker run -d --name "deepstream-$USER" --restart always --runtime nvidia --gpus all --network host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e COMPANY_ID="$CID" -e WEBHOOK_TOKEN="$APIKEY" \
  -e WEBHOOK_URL="$WEBHOOK_BASE/api/attendance/entry" \
  -e PYTHONPATH=/workspace/deepstream_python_apps/bindings/build \
  -e PIPELINE_DISPLAY="$PIPELINE_DISPLAY" \
  "${DISPLAY_ARGS[@]}" \
  -v "$DS:/workspace" \
  -v "/home/meta/deploy/test/data/company_images/$FOLDER:/workspace/data/known_faces" \
  -v "$DS/config/companies/$USER.toml:/workspace/config/config_pipeline.toml" \
  -w /workspace "$IMG" \
  bash /workspace/tools/pipeline_entry.sh
  # NB: invoked as a script-file path, NOT `bash -c "..."`. The DeepStream image
  # entrypoint word-splits an unquoted $@, which silently breaks a multi-word -c
  # string (pip prints usage, exits 0, main_enterprise.py never runs). See
  # tools/pipeline_entry.sh.

# 4) regenerate MediaMTX paths so this company's annotated streams are published
#    as HLS/WebRTC. MediaMTX hot-reloads its config file, so no restart needed.
python3 "$DS/tools/gen_mediamtx_paths.py" || echo "WARN: MediaMTX path regen failed (is MediaMTX deployed?)"

echo "Started deepstream-$USER. Logs: docker logs -f deepstream-$USER"
