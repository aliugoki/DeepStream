#!/usr/bin/env bash
# In-container entry for a company pipeline.
#
# IMPORTANT: this MUST be invoked as a script file —
#     bash /workspace/tools/pipeline_entry.sh
# and NOT via `bash -c "pip install ... && python3 main_enterprise.py"`.
# The DeepStream image's entrypoint ends with an UNQUOTED `$@`
# (/opt/nvidia/deepstream/deepstream-7.1/entrypoint.sh: `nvidia_entrypoint.sh $@`),
# which word-splits any multi-word `-c` string. The result is that bash's -c
# command becomes just the first token (`pip`) — pip prints its usage and exits
# 0, so python3 main_enterprise.py never runs and the container crash-loops under
# --restart always. Passing a single script-file path survives the splitting.
set -euo pipefail

pip install -q --no-cache-dir requests pyds psycopg2-binary toml opencv-python-headless cuda-python
exec python3 main_enterprise.py
