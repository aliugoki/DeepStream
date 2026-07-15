"""
Publish recognized face identities to VisionTrack via Redis Streams.

VisionTrack (/home/meta/visiontrack) is a separate multi-tenant people-tracking
platform. Its own DeepStream worker publishes *body* ReID embeddings to
`vt:ds:embeddings:<tenant_id>` and tracks anonymous `persons`. Our face pipeline
knows *who* people are (emp_id), but:
  * ArcFace face embeddings live in a DIFFERENT vector space than VisionTrack's
    body-ReID embeddings, so they must NOT be pushed into vt:ds:embeddings; and
  * the two pipelines use separate trackers, so track_ids don't correspond.

Therefore we publish to a DEDICATED stream `vt:face:identities:<tenant_id>`
carrying (camera_id, bbox, time, emp_id, name, score). VisionTrack correlates
these to its own active person-tracks by camera + bbox overlap + timestamp and
labels the matching person — keeping its ReID space clean. See
VISIONTRACK_LINK.md for the VisionTrack-side consumer spec.

Design notes:
  * Redis is optional: a failed connect/XADD is logged and dropped; the GStreamer
    pipeline is never blocked or crashed by VisionTrack being down.
  * Per (camera, emp_id) throttle avoids spamming the stream every frame for a
    committed track.
"""
import json
import time
import logging

log = logging.getLogger("visiontrack_publisher")

DEFAULT_STREAM_PREFIX = "vt:face:identities"
DEFAULT_STREAM_MAXLEN = 100_000


class VisionTrackPublisher:
    def __init__(self, redis_url, source_map, stream_prefix=DEFAULT_STREAM_PREFIX,
                 maxlen=DEFAULT_STREAM_MAXLEN, throttle_sec=2.0,
                 time_fn=time.monotonic):
        """
        source_map: {source_id(int) -> {"camera_id": <uuid str>, "tenant_id": <uuid str>}}
                    mapping each pipeline source to its VisionTrack camera/tenant.
        """
        self.redis_url = redis_url
        self.source_map = {int(k): v for k, v in (source_map or {}).items()}
        self.stream_prefix = stream_prefix
        self.maxlen = maxlen
        self.throttle_sec = throttle_sec
        self._now = time_fn
        self._last_pub = {}      # (camera_id, emp_id) -> monotonic ts
        self._redis = None
        self._connect()

    def _connect(self):
        try:
            import redis
            self._redis = redis.from_url(self.redis_url, socket_timeout=1.0,
                                         socket_connect_timeout=1.0)
            self._redis.ping()
            log.info("VisionTrack publisher connected: %s", self.redis_url)
        except Exception as e:
            self._redis = None
            log.warning("VisionTrack Redis unavailable (%s); identity publishing "
                        "disabled until reachable.", e)

    def publish(self, source_id, emp_id, name, score, bbox, captured_at_ms=None):
        """
        Publish one recognized identity. bbox is (left, top, width, height) in
        the source frame. No-op (and never raises) if the source isn't mapped,
        Redis is down, or the (camera, emp_id) throttle window hasn't elapsed.
        """
        m = self.source_map.get(int(source_id))
        if not m:
            return
        camera_id, tenant_id = m.get("camera_id"), m.get("tenant_id")
        # Multi-tenant: prefer the company_id stream key (FaceTrack has its own
        # company_id; VisionTrack routes vt:face:identities:<company_id> to the
        # mapped tenant). Falls back to tenant_id for single-tenant setups.
        stream_key = m.get("company_id") or tenant_id
        if not camera_id or not stream_key:
            return

        now = self._now()
        key = (camera_id, str(emp_id))
        last = self._last_pub.get(key)
        if last is not None and (now - last) < self.throttle_sec:
            return

        if self._redis is None:
            self._connect()
            if self._redis is None:
                return

        left, top, width, height = bbox
        fields = {
            "camera_id": str(camera_id),
            "emp_id": str(emp_id),
            "name": name or "",
            "score": f"{float(score):.4f}",
            "bbox": json.dumps([float(left), float(top), float(width), float(height)]),
            "captured_at_ms": str(captured_at_ms if captured_at_ms is not None
                                  else int(now * 1000)),
        }
        try:
            self._redis.xadd(f"{self.stream_prefix}:{stream_key}", fields,
                             maxlen=self.maxlen, approximate=True)
            self._last_pub[key] = now
        except Exception as e:
            log.warning("VisionTrack XADD failed (%s); dropping. Will retry.", e)
            self._redis = None  # force reconnect next time


def from_config(cfg):
    """
    Build a publisher from the [visiontrack] config section, or return None if
    disabled/absent. Expected TOML:

        [visiontrack]
        enabled = true
        redis_url = "redis://localhost:6379/0"
        throttle_sec = 2.0
        # one [[visiontrack.cameras]] per pipeline source, in source order
        [[visiontrack.cameras]]
        source_id = 0
        camera_id = "<visiontrack camera uuid>"
        # Multi-tenant: set company_id (this company's id) so VisionTrack routes to
        # the right tenant. For single-tenant, set tenant_id instead.
        company_id = "<company id>"
        # tenant_id = "<visiontrack tenant uuid>"
    """
    vt = cfg.get("visiontrack") if isinstance(cfg, dict) else None
    if not vt or not vt.get("enabled"):
        return None
    source_map = {int(c["source_id"]): {"camera_id": c.get("camera_id"),
                                        "tenant_id": c.get("tenant_id"),
                                        "company_id": c.get("company_id")}
                  for c in vt.get("cameras", []) if "source_id" in c}
    if not source_map:
        log.warning("[visiontrack] enabled but no cameras mapped; publisher off.")
        return None
    return VisionTrackPublisher(
        redis_url=vt.get("redis_url", "redis://localhost:6379/0"),
        source_map=source_map,
        stream_prefix=vt.get("stream_prefix", DEFAULT_STREAM_PREFIX),
        maxlen=int(vt.get("stream_maxlen", DEFAULT_STREAM_MAXLEN)),
        throttle_sec=float(vt.get("throttle_sec", 2.0)),
    )
