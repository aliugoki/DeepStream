# Linking the face pipeline → VisionTrack (identity feed)

Goal: surface **who** a tracked person is (emp_id / name) inside VisionTrack,
without corrupting its body-ReID space.

## Why not the existing embeddings stream
VisionTrack's `vt:ds:embeddings:<tenant_id>` carries **body** ReID vectors
(PeopleNet/OSNet) and its `persons` are clustered in that space. Our ArcFace
**face** vectors are a different space, and the two pipelines have independent
trackers (track_ids don't correspond). So we publish a **separate** stream and
let VisionTrack correlate by space+time.

## Pipeline side (done — in this repo)
`utils/visiontrack_publisher.py` XADDs to `vt:face:identities:<tenant_id>`:

| field | meaning |
|---|---|
| `camera_id` | VisionTrack camera UUID (from config mapping) |
| `emp_id` | recognized employee id |
| `name` | display name (may be empty) |
| `score` | match confidence |
| `bbox` | JSON `[left, top, width, height]` in source-frame pixels |
| `captured_at_ms` | epoch millis |

Enable in `config/config_pipeline.toml`:

```toml
[visiontrack]
enabled = true
redis_url = "redis://localhost:6379/0"   # VisionTrack's Redis
throttle_sec = 2.0
# one entry per pipeline source (order matches [[sources]]):
[[visiontrack.cameras]]
source_id = 0
camera_id = "<visiontrack cameras.id UUID>"
tenant_id = "<visiontrack tenants.id UUID>"
```

It is safe: if Redis is down or a source is unmapped, publishing is a no-op —
the pipeline never blocks. Only the **enterprise** pipeline publishes (the probe
has the bbox + committed identity).

## VisionTrack side — IMPLEMENTED (in /home/meta/visiontrack/visiontrack)
Added (mirrors the existing embedding-consumer conventions; keeps ReID
untouched — face identity is a pure overlay):

- **Migration** `backend/alembic/versions/0017_face_identities.py` — new
  `person_identities` table (`tenant_id, person_id?, track_id?, camera_id,
  emp_id, name, source, confidence, votes, first/last_labeled_at`; unique
  `(tenant_id, person_id, emp_id)`). down_revision `0016_global_tracks`.
- **Model** `PersonIdentity` in `backend/app/modules/persons/models.py`
  (auto-registered via `app/core/models.py` -> persons).
- **Consumer** `backend/app/modules/persons/face_identity_consumer.py` —
  per-tenant `xreadgroup` on `vt:face:identities:<tenant_id>`, correlates each
  event to a track by camera + bbox-IoU + time window, resolves `person_id`,
  and upserts `person_identities` with running confidence + vote count (one bad
  correlation can't flip a label).
- **Config** `backend/app/core/config.py` — `FACE_IDENTITY_ENABLED`,
  `FACE_IDENTITY_STREAM_PREFIX`, `FACE_ID_CORRELATION_WINDOW_MS`,
  `FACE_ID_MIN_IOU`, `FACE_ID_CANDIDATE_LIMIT`.
- **Lifespan** `backend/app/main.py` — consumer started/stopped alongside the
  others, gated by `FACE_IDENTITY_ENABLED`.

Deploy (in the VisionTrack backend container/venv):
```bash
alembic upgrade head        # creates person_identities (now head 0017)
# restart the backend → the face-identity consumer auto-starts per tenant
```
Not yet done (left for you, optional): join `person_identities` into the
persons / live-wall API responses so the dashboard renders the employee name on
the tracked body. The data is captured; this is just the read/UI surface.

NOT committed — review the diff in the visiontrack repo first; it was not
runtime-tested against the live stack here (no DB/Redis), only compile-checked.

## Recommended alternative (higher fidelity, more work)
Run face recognition **inside** VisionTrack's `ai-worker-ds` as a face SGIE on
the *same* pipeline. Then body-ReID and face-identity share one track_id and no
cross-pipeline bbox correlation is needed. Choose this if you want identity
tightly bound to tracks rather than correlated after the fact.
