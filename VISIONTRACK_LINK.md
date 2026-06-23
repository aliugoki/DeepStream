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

## VisionTrack side (to add there — spec, not yet implemented)
VisionTrack's `persons` has no identity column, so add an association rather
than mutating ReID clustering:

1. **Migration** — a `person_identities` table (or nullable cols on `persons`):
   `person_id (FK), tenant_id, emp_id, name, source='face', confidence,
   first_labeled_at, last_labeled_at`. Keeps face identity separate from ReID.
2. **Consumer** — mirror `backend/app/modules/persons/consumer.py`: an
   `xreadgroup` loop on `vt:face:identities:<tenant_id>` per tenant.
3. **Correlation** — for each identity event, find the active track on
   `camera_id` whose latest `track_points.bbox` overlaps the event `bbox` at
   `captured_at_ms` (IoU over a small time window); resolve its `person_id`;
   upsert `person_identities`. Use a vote/decay so a single bad correlation
   doesn't relabel a person.
4. **API/UI** — join `person_identities` in the persons/live-wall responses so
   the dashboard shows the employee name on the tracked body.

This keeps ReID matching untouched and makes the face identity an overlay.

## Recommended alternative (higher fidelity, more work)
Run face recognition **inside** VisionTrack's `ai-worker-ds` as a face SGIE on
the *same* pipeline. Then body-ReID and face-identity share one track_id and no
cross-pipeline bbox correlation is needed. Choose this if you want identity
tightly bound to tracks rather than correlated after the fact.
