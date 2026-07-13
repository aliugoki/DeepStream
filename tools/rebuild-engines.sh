#!/usr/bin/env bash
# Rebuild the GPU-specific TensorRT engines after moving to a new/different GPU.
# TensorRT plan files are compiled per-GPU; the ArcFace engine (arc1.engine) is
# loaded directly by the pipeline and never auto-rebuilds, so it MUST be rebuilt
# here or recognition silently fails. See DEPLOYMENT.md §12.
#
#   ./tools/rebuild-engines.sh            # rebuild ArcFace + reset YOLO engine
#   ./tools/rebuild-engines.sh --enroll   # also re-enroll every gallery (aligned, onnx)
#
# Requires: the deepstream-facepipe:latest image and a working GPU under docker
# (nvidia-container-toolkit). Env overrides: FACEPIPE_IMG, COMPANY_IMAGES_ROOT,
# ARCFACE_MAX_BATCH.
set -euo pipefail

DS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${FACEPIPE_IMG:-deepstream-facepipe:latest}"
MODELS="$DS/models"
IMAGES_ROOT="${COMPANY_IMAGES_ROOT:-/home/meta/deploy/test/data/company_images}"
MAXB="${ARCFACE_MAX_BATCH:-16}"

c()   { printf '\033[36m• %s\033[0m\n' "$*"; }
ok()  { printf '\033[32m✓ %s\033[0m\n' "$*"; }
die() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- preflight ------------------------------------------------------------- #
docker image inspect "$IMG" >/dev/null 2>&1 \
  || die "image '$IMG' not found — build it first:  docker build -t deepstream-facepipe:latest -f Dockerfile.facepipe ."
docker run --rm --gpus all --entrypoint true "$IMG" 2>/dev/null \
  || die "GPU not usable from docker (--gpus all). Install/enable nvidia-container-toolkit."
[ -f "$MODELS/arcface/arcface.onnx" ] \
  || die "missing $MODELS/arcface/arcface.onnx — copy the model files onto this machine first."

# --- 1) ArcFace engine (dynamic batch 1..MAXB, fp16) ----------------------- #
# Built via the TensorRT builder API (not `trtexec --onnx`) so we can set an
# optimization profile for the dynamic batch dimension the embedder requires,
# and read the real input tensor name/dims straight from the ONNX.
c "Rebuilding ArcFace engine for this GPU (fp16, batch 1..$MAXB) …"
docker run --rm -i --gpus all -e MAXB="$MAXB" -v "$MODELS":/m --entrypoint python3 "$IMG" - <<'PY' || die "ArcFace engine build failed (see errors above)."
import os, sys, tensorrt as trt
onnx_path, eng_path = "/m/arcface/arcface.onnx", "/m/arcface/arc1.engine"
maxb = int(os.environ.get("MAXB", "16"))
log = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(log)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(net, log)
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit("ONNX parse failed")
inp = net.get_input(0)
name, shp = inp.name, inp.shape
c = shp[1] if shp[1] > 0 else 3
h = shp[2] if shp[2] > 0 else 112
w = shp[3] if shp[3] > 0 else 112
cfg = builder.create_builder_config()
try:
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
except Exception:
    cfg.max_workspace_size = 2 << 30
if builder.platform_has_fast_fp16:
    cfg.set_flag(trt.BuilderFlag.FP16)
prof = builder.create_optimization_profile()
prof.set_shape(name, (1, c, h, w), (max(1, maxb // 2), c, h, w), (maxb, c, h, w))
cfg.add_optimization_profile(prof)
ser = builder.build_serialized_network(net, cfg)
if ser is None:
    sys.exit("engine build returned None")
with open(eng_path, "wb") as f:
    f.write(ser)
print(f"wrote {eng_path}  (input '{name}' {list(shp)} -> batch 1..{maxb})")
PY
ok "ArcFace engine rebuilt: $MODELS/arcface/arc1.engine"

# --- 2) YOLO-face engine: drop the stale one so nvinfer rebuilds from ONNX -- #
if [ -f "$MODELS/yolov8n_face/yolov8n-face2.engine" ]; then
  rm -f "$MODELS/yolov8n_face/yolov8n-face2.engine"
  ok "removed stale YOLO engine — DeepStream rebuilds it from ONNX on the next pipeline launch"
else
  c "no YOLO engine present — DeepStream will build it on first launch"
fi

# --- 3) optional: re-enroll every gallery so vectors match this machine ----- #
if [ "${1:-}" = "--enroll" ]; then
  [ -d "$IMAGES_ROOT" ] || die "gallery root not found: $IMAGES_ROOT (set COMPANY_IMAGES_ROOT)"
  shopt -s nullglob
  for dir in "$IMAGES_ROOT"/*/; do
    imgs=("$dir"*.png "$dir"*.jpg)
    [ ${#imgs[@]} -gt 0 ] || continue
    c "re-enrolling gallery: $(basename "$dir")"
    docker run --rm --gpus all -v "$DS":/workspace -v "$dir":/gallery -w /workspace \
      --entrypoint bash "$IMG" \
      -c "pip install -q onnxruntime 2>/dev/null; python3 tools/enroll.py --all --known-dir /gallery --embedder onnx" \
      2>&1 | grep -E "Enrolled|gallery_meta|WARNING" | tail -3 || true
  done
  ok "gallery re-enrollment complete"
else
  c "Skipped gallery re-enrollment — re-run with --enroll (or re-enroll per company)."
fi

echo
ok "Done. Relaunch each company's pipeline (dashboard → Pipeline → Restart) to load the new engines."
