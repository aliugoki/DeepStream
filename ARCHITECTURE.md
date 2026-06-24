# Architecture & file map

This project accumulated ~12 `main_*.py` and ~31 `probe_*` variants over time.
This document marks the **canonical** files so the rest can be treated as
history/experiments. Nothing has been deleted; archive the legacy set when you
are confident the enterprise path is validated on-device.

## Canonical (enterprise) pipeline

| Role | File |
|---|---|
| Entry point | `main_enterprise.py` |
| Recognition probe (aligned ArcFace, in-probe embed) | `utils/probe_enterprise.py` |
| Face alignment (Umeyama, landmarks) | `utils/face_align.py` |
| Shared ArcFace embedder (ONNX + TRT) | `utils/arcface_embedder.py` |
| Gallery match + threshold/margin + per-track voting | `utils/recognition.py` |
| Reliability (nvurisrcbin reconnect, bus, health, watchdog) | `utils/reliability.py` |
| Hardened DB/webhook (pooled, env secrets, TLS) | `utils/db_service.py` |
| Offline gallery enrollment | `tools/enroll.py` |
| Config | `config/config_pipeline.toml`, `config/config_yolo.txt`, `config/config_arcface.txt`, `config/config_tracker_perf.txt` |
| Deploy | `deploy/deepstream-face.service`, `.env` (from `.env.example`) |
| Docs | `README_enterprise.md`, this file |

## Legacy (still runnable, kept for reference / rollback)

- `main_udp.py` — the previous production entry point (unaligned SGIE path).
- `utils/probe_git.py` — legacy recognition probe; the enterprise probe reuses
  its display/line-crossing helpers.
- `utils/posgres_service.py` — legacy DB layer (hardcoded secrets); superseded
  by `utils/db_service.py`.

## Experimental / dead (candidates for `archive/`)

`main.py main1.py main2.py main_8mp.py main_2026.py main_face_copy.py main_git.py
main_iaa.py main_kafka.py main_modular.py merged_main.py my.py my_old.py cap.py`
and the `utils/probe*` / `utils/probes/` variants other than `probe_git.py`.

Recommended next step (not done automatically — it moves your files):
```bash
mkdir -p archive && git mv <legacy_experimental_files> archive/
```

## Data flow (enterprise)

```
RTSP --nvurisrcbin(reconnect)--> nvstreammux --> nvinfer(YOLOv8n-face, +landmarks)
   --> nvtracker --> nvvideoconvert(RGBA, unified mem) --> [probe_enterprise]
   --> nvmultistreamtiler --> nvvideoconvert --> nvdsosd --> tee
       |-> display / fakesink
       |-> nvv4l2h264enc -> udpsink -> RTSP out

[probe_enterprise] per uncommitted track:
   mask_params landmarks -> inverse-letterbox -> Umeyama align 112x112
   -> ArcFace TRT embed -> Gallery matmul (threshold+margin)
   -> TrackIdentityManager vote/commit -> attendance queue -> db_service worker
   -> Postgres + webhook ; health.mark_frame() per frame
```
