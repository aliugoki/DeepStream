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

# 1) generate the company config from the DB cameras
python3 "$DS/tools/gen_company_config.py" "$USER" --index "$INDEX"

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
docker run -d --name "deepstream-$USER" --restart always --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e COMPANY_ID="$CID" -e WEBHOOK_TOKEN="$APIKEY" \
  -e WEBHOOK_URL="$WEBHOOK_BASE/api/attendance/entry" \
  -e PYTHONPATH=/workspace/deepstream_python_apps/bindings/build \
  -v "$DS:/workspace" \
  -v "/home/meta/deploy/test/data/company_images/$FOLDER:/workspace/data/known_faces" \
  -v "$DS/config/companies/$USER.toml:/workspace/config/config_pipeline.toml" \
  -w /workspace "$IMG" \
  bash /workspace/tools/pipeline_entry.sh
  # NB: invoked as a script-file path, NOT `bash -c "..."`. The DeepStream image
  # entrypoint word-splits an unquoted $@, which silently breaks a multi-word -c
  # string (pip prints usage, exits 0, main_enterprise.py never runs). See
  # tools/pipeline_entry.sh.

echo "Started deepstream-$USER. Logs: docker logs -f deepstream-$USER"
