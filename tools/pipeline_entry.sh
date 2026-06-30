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

# tensorrt==10.3.0 matches the image's TRT 10.3.0.26 C++ libs (TRT_VERSION) and
# provides the python bindings the image does not ship; the 'trt' embedder
# backend (utils/arcface_embedder.ArcFaceTRT) does `import tensorrt`.
#
# cuda-python MUST be pinned to the CUDA 12.x line: ArcFaceTRT uses its cudart
# (cuda.bindings.runtime) for cudaMalloc, and the host driver (575.x, CUDA 12.9
# max) is too old for the latest cuda-python (CUDA 13, needs driver >=580) —
# that mismatch fails cudaMalloc with cudaErrorInsufficientDriver (35).
#
# Skip the (slow, ~1 GB from pypi.nvidia.com) install when the deps are already
# present — the deepstream-facepipe image (Dockerfile.facepipe) bakes them in, so
# this is a no-op there and only runs on the bare DeepStream base image.
if ! python3 -c "import tensorrt, cuda.bindings.runtime" >/dev/null 2>&1; then
    echo "Installing pipeline deps (not baked into this image)..."
    pip install -q --no-cache-dir requests pyds psycopg2-binary toml opencv-python-headless \
        "cuda-python>=12.6,<13" tensorrt==10.3.0
fi
exec python3 main_enterprise.py
