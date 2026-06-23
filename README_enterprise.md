# Enterprise recognition upgrade (accuracy track)

New, self-contained modules that improve **recognition accuracy** without
touching the running `main_udp.py` / `utils/probe_git.py`. You can A/B the two
pipelines and roll back instantly.

## What changed and why

| Problem (legacy) | Fix | File |
|---|---|---|
| `RECOGNITION_THRESHOLD = 0.2` hardcoded; `rec_threshold` in config ignored; no margin check | Config-driven threshold **+ top-1/top-2 margin gate** | `utils/recognition.py` |
| Raw axis-aligned crops fed to ArcFace (big accuracy loss) | **5-point Umeyama alignment** to the canonical 112×112 template using the landmarks the YOLO-face PGIE already emits | `utils/face_align.py`, `utils/probe_enterprise.py` |
| O(N) Python loop over gallery keys | Single **vectorized matmul** against an `(N,512)` matrix | `utils/recognition.py` (`Gallery`) |
| `clear()+update()` gallery hot-reload races the stream thread | **Atomic swap** | `utils/recognition.py` + `main_enterprise.py` |
| Re-decides identity every frame; `TRACKED_OBJECTS` grows forever | **Recognize-once-per-track** voting + **TTL eviction** | `utils/recognition.py` (`TrackIdentityManager`) |
| Enrollment vs inference preprocessing could drift | **One shared embedder** for enrollment and live | `utils/arcface_embedder.py` |

## Prerequisite: re-enroll the gallery (one time)

Alignment changes the embedding space, so the legacy `*.npy` (made by the
unaligned SGIE) are **invalid** for the aligned pipeline. Regenerate them from
the source images:

```bash
# offline, CPU, no GPU/DeepStream needed
python3 tools/enroll.py --all                 # writes <id>.npy + gallery_meta.json
python3 tools/enroll.py --image data/known_faces/101.png --dry-run   # inspect one
```

The tool detects the face (YOLOv8n-face ONNX), aligns, embeds (ArcFace ONNX by
default; `--embedder trt` on-device), and writes `(512,1)` vectors matching
`load_faces`. It refuses to silently embed the wrong face when an enrollment
image is ambiguous (multiple similar-size faces) and skips images with no face
or poor landmark confidence.

## Run the enterprise pipeline

```bash
python3 main_enterprise.py     # reads config/config_pipeline.toml
```

Optional `[pipeline]` knobs (all have safe defaults — no config edit required):

```toml
[pipeline]
rec_threshold   = 0.35   # accept only if cosine >= this
rec_margin      = 0.05   # AND (top1 - top2) >= this
track_min_votes = 3      # frames agreeing before an identity is committed
track_ttl_sec   = 30     # evict idle tracks after this (bounds memory)
embedder        = "trt"  # "trt" (live) or "onnx" (debug)
arcface_engine  = "/workspace/models/arcface/arc1.engine"
```

## Offline validation already done (on your real gallery)

- `python3 utils/face_align.py` — Umeyama round-trip error 3e-6 px.
- `python3 utils/recognition.py` — threshold/margin, matmul match, voting, TTL.
- `python3 tools/enroll.py --all --dry-run` — all 5 faces detected & aligned.
- Aligned vs legacy embedding: same-person cosine ≈ 0.86–0.89 (alignment is
  doing real work → re-enrollment is mandatory). Aligned **cross-identity**
  cosine max 0.166 / mean 0.043 → `0.35`+margin is safely above impostors;
  the legacy `0.2` was dangerously close.

## Reliability / uptime (track 2)

`main_enterprise.py` now self-heals instead of dying on the first hiccup
(`utils/reliability.py`):

- **RTSP auto-reconnect** — inputs use `nvurisrcbin` (not `uridecodebin`), with
  `rtsp-reconnect-interval`/`-attempts`, forced TCP, and `latency` wired from
  your existing per-source config keys. A camera blip reconnects on its own.
- **Recoverable bus handler** — source errors are logged/counted and survived;
  only a fatal *core-element* error quits (so the supervisor can restart clean).
- **Health + metrics endpoint** — `GET :9108/healthz` (200 healthy / 503 not)
  and `:9108/metrics` (Prometheus: per-source frames, reconnects, errors,
  seconds-since-frame, stale flag). Point your existing Grafana/Prometheus at it.
  Port 9108 avoids the host's kafka-ui on :8080; override with `health_port`.
- **Frame-flow watchdog** — logs any source silent beyond `stale_after_sec`.
- **Process supervisor** — `deploy/deepstream-face.service` (systemd
  `Restart=always`) restarts the container on hard crash/OOM.

Optional `[pipeline]` knobs: `health_port` (9108), `stale_after_sec` (20),
`watchdog_interval_sec` (10).

## Must validate ON-DEVICE (no GPU in the build env)

1. `ArcFaceTRT` (`arc1.engine`) output cosine-matches `ArcFaceONNX` (> 0.99) for
   the same chip. If so, an ONNX-enrolled gallery is valid against the TRT live
   path; otherwise re-enroll with `--embedder trt`.
2. `get_nvds_buf_surface` returns an addressable RGBA frame on the RTX 3070 with
   `nvbuf-memory-type=2` (unified). If frames come back empty, also set
   `cudadec-memtype=2` on the decoder in `create_source_bin`.
3. `obj_meta.mask_params.get_mask_array()` returns the 15 landmark floats from
   the YOLO-face parser (confirm size/layout on a live frame).
4. Re-tune `rec_threshold`/`rec_margin` on the aligned distribution with several
   images per person (the offline numbers used one image each).
5. Live FPS on the 3070 — recognize-once-per-track should keep per-frame
   embedding rare at steady state.
6. `nvurisrcbin` reconnect: confirm the exact reconnect property names exist on
   your DS 7.1 build (the code sets them defensively via `find_property`, so
   absent ones are skipped — check the log for which applied), and verify a
   pulled camera cable reconnects without restarting the process.
```
