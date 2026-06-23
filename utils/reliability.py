"""
Reliability / uptime layer for the DeepStream face-recognition pipeline.

The legacy pipeline (main_udp.py) has three single-points-of-failure:
  1. inputs use plain `uridecodebin` with NO RTSP reconnection -- a camera blip
     permanently kills that stream;
  2. `bus_call` calls `loop.quit()` on ANY error, tearing down the whole
     pipeline (and process) on a single source hiccup;
  3. there is no health signal, so an orchestrator can't tell a hung pipeline
     from a healthy one.

This module fixes all three without new dependencies:
  * `create_resilient_source_bin` -> `nvurisrcbin` with native RTSP reconnect
    wired from the existing per-source config keys
    (rtsp-reconnect-interval-sec, num-retry, latency);
  * `make_resilient_bus_call` -> recovers from source errors (lets nvurisrcbin
    reconnect) and only quits on genuinely fatal core-element errors;
  * `HealthState` + `start_health_server` + `start_watchdog` -> per-source
    frame-flow tracking, a /healthz + /metrics HTTP endpoint, and a watchdog
    that flags silent sources.

Pair with a process supervisor (systemd `Restart=always`, see
`deploy/deepstream-face.service`) so a hard crash also self-heals.
"""
import sys
import time
import json
import logging
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

log = logging.getLogger("reliability")

# select-rtp-protocol value: 4 = force TCP (most reliable over lossy/NAT links).
RTP_PROTOCOL_TCP = 4


# --------------------------------------------------------------------------- #
# Resilient source: nvurisrcbin with RTSP auto-reconnect
# --------------------------------------------------------------------------- #
def _safe_set(element, prop, value):
    """Set a property only if the element actually exposes it (DS-version safe)."""
    try:
        if element.find_property(prop) is not None:
            element.set_property(prop, value)
            return True
    except Exception as e:
        log.debug("could not set %s=%r: %s", prop, value, e)
    return False


def _on_source_pad_added(_bin, pad, ghost):
    """Link the dynamic video pad to the source bin's ghost pad (NVMM only)."""
    try:
        caps = pad.get_current_caps() or pad.query_caps()
        name = caps.get_structure(0).get_name().lower()
        if "video" not in name:
            return
        if not caps.get_features(0).contains("memory:NVMM"):
            log.error("source pad is not NVMM (%s); nvdec did not engage", name)
            return
        if not ghost.set_target(pad):
            log.error("failed to set ghost pad target")
    except Exception:
        log.error("pad-added handler failed:\n%s", traceback.format_exc())


def create_resilient_source_bin(index, src_cfg, gpu_id=0, cudadec_memtype=2):
    """
    Build a source bin backed by `nvurisrcbin` with RTSP reconnection.

    Reads the same per-source config keys the project already uses:
      uri, num-retry, rtsp-reconnect-interval-sec, latency.
    Returns a Gst.Bin with a (target-less) "src" ghost pad, mirroring the
    structure of the legacy create_source_bin so the rest of the graph is
    unchanged.
    """
    uri = src_cfg["uri"]
    bin_ = Gst.Bin.new(f"source-bin-{index}")
    src = Gst.ElementFactory.make("nvurisrcbin", f"nvurisrc-{index}")
    if not src:
        raise RuntimeError("nvurisrcbin unavailable (is this the DeepStream container?)")

    src.set_property("uri", uri)
    is_rtsp = uri.startswith(("rtsp://", "rtspt://"))
    if is_rtsp:
        # Reconnect window in seconds; >0 makes nvurisrcbin re-establish a
        # stalled RTSP connection on its own.
        _safe_set(src, "rtsp-reconnect-interval",
                  int(src_cfg.get("rtsp-reconnect-interval-sec", 10)))
        # -1 = retry forever; otherwise honor configured num-retry.
        attempts = int(src_cfg.get("num-retry", -1))
        _safe_set(src, "rtsp-reconnect-attempts", attempts if attempts > 0 else -1)
        _safe_set(src, "select-rtp-protocol", RTP_PROTOCOL_TCP)
        _safe_set(src, "latency", int(src_cfg.get("latency", 200)))
        _safe_set(src, "udp-buffer-size", 2 * 1024 * 1024)
    _safe_set(src, "drop-on-latency", True)
    _safe_set(src, "cudadec-memtype", cudadec_memtype)
    _safe_set(src, "gpu-id", gpu_id)

    bin_.add(src)
    ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    bin_.add_pad(ghost)
    src.connect("pad-added", _on_source_pad_added, ghost)
    return bin_


# --------------------------------------------------------------------------- #
# Health / metrics state
# --------------------------------------------------------------------------- #
class HealthState:
    """Thread-safe pipeline health snapshot, shared across probe/bus/watchdog."""

    def __init__(self, sources, stale_after_sec=20.0, time_fn=time.monotonic):
        self._lock = threading.Lock()
        self._now = time_fn
        self.stale_after = stale_after_sec
        self.started_at = self._now()
        self.pipeline_state = "NULL"
        start = self._now()
        self.sources = {
            i: {"id": s.get("id", f"cam{i}"), "frames": 0,
                "last_frame": start, "reconnects": 0, "errors": 0}
            for i, s in enumerate(sources)
        }

    def mark_frame(self, source_id):
        with self._lock:
            s = self.sources.get(source_id)
            if s:
                s["frames"] += 1
                s["last_frame"] = self._now()

    def mark_reconnect(self, source_id):
        with self._lock:
            s = self.sources.get(source_id)
            if s:
                s["reconnects"] += 1

    def mark_error(self, source_id):
        with self._lock:
            s = self.sources.get(source_id)
            if s:
                s["errors"] += 1

    def set_pipeline_state(self, state):
        with self._lock:
            self.pipeline_state = state

    def snapshot(self):
        now = self._now()
        with self._lock:
            srcs = {}
            healthy = self.pipeline_state == "PLAYING"
            for i, s in self.sources.items():
                stale = (now - s["last_frame"]) > self.stale_after
                healthy = healthy and not stale
                srcs[i] = {**s, "stale": stale,
                           "seconds_since_frame": round(now - s["last_frame"], 1)}
            return {"healthy": healthy, "pipeline_state": self.pipeline_state,
                    "uptime_sec": round(now - self.started_at, 1), "sources": srcs}


def _prometheus_text(snap):
    lines = [
        "# HELP ds_pipeline_up 1 if pipeline is PLAYING and all sources fresh",
        "# TYPE ds_pipeline_up gauge",
        f"ds_pipeline_up {1 if snap['healthy'] else 0}",
        "# TYPE ds_uptime_seconds gauge",
        f"ds_uptime_seconds {snap['uptime_sec']}",
    ]
    for i, s in snap["sources"].items():
        lbl = f'source="{i}",id="{s["id"]}"'
        lines += [
            f"ds_source_frames_total{{{lbl}}} {s['frames']}",
            f"ds_source_reconnects_total{{{lbl}}} {s['reconnects']}",
            f"ds_source_errors_total{{{lbl}}} {s['errors']}",
            f"ds_source_seconds_since_frame{{{lbl}}} {s['seconds_since_frame']}",
            f"ds_source_stale{{{lbl}}} {1 if s['stale'] else 0}",
        ]
    return "\n".join(lines) + "\n"


def start_health_server(health_state, port=9108):
    """Serve /healthz (200/503) and /metrics (Prometheus) on a daemon thread."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # silence default access logging
            pass

        def do_GET(self):
            snap = health_state.snapshot()
            if self.path == "/metrics":
                body = _prometheus_text(snap).encode()
                ctype = "text/plain; version=0.0.4"
                code = 200
            else:  # /healthz and everything else
                body = json.dumps(snap).encode()
                ctype = "application/json"
                code = 200 if snap["healthy"] else 503
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("health endpoint on :%d (/healthz, /metrics)", port)
    return srv


# --------------------------------------------------------------------------- #
# Resilient bus handler + watchdog
# --------------------------------------------------------------------------- #
def _is_source_message(message):
    src = message.src
    name = src.get_name() if src else ""
    # nvurisrcbin lives inside our "source-bin-<i>" wrappers.
    el = src
    while el is not None:
        if el.get_name().startswith("source-bin-"):
            try:
                return int(el.get_name().rsplit("-", 1)[1])
            except ValueError:
                return -1
        el = el.get_parent()
    return -1 if "source" not in name.lower() else -1


def make_resilient_bus_call(loop, health_state=None):
    """
    Bus handler that recovers from source-level failures instead of quitting.

    Source errors are logged and counted but NOT fatal -- nvurisrcbin reconnects
    on its own. Only errors from core elements (mux/infer/tracker/sinks) quit
    the loop so the process supervisor can restart cleanly.
    """
    def handler(_bus, message, _loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            log.warning("EOS received; quitting for supervisor restart")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            log.warning("%s: %s", err, dbg)
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            sidx = _is_source_message(message)
            if sidx >= 0:
                log.error("source %d error (recoverable, reconnecting): %s | %s",
                          sidx, err, dbg)
                if health_state:
                    health_state.mark_error(sidx)
            else:
                log.critical("fatal pipeline error: %s | %s", err, dbg)
                loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            s = message.get_structure()
            if s and health_state:
                nm = s.get_name() or ""
                # nvurisrcbin emits stream reset/reconnect element messages.
                if "reconnect" in nm.lower() or "stream-eos" in nm.lower():
                    sidx = _is_source_message(message)
                    if sidx >= 0:
                        health_state.mark_reconnect(sidx)
                        log.warning("source %d stream event: %s", sidx, nm)
        elif t == Gst.MessageType.STATE_CHANGED and health_state:
            if message.src and message.src.get_name() == "ds-enterprise":
                _, new, _ = message.parse_state_changed()
                health_state.set_pipeline_state(new.value_nick.upper())
        return True
    return handler


def start_watchdog(health_state, interval_sec=10):
    """Periodically log stale sources so silent feeds are visible/alertable."""
    def tick():
        snap = health_state.snapshot()
        for i, s in snap["sources"].items():
            if s["stale"]:
                log.warning("WATCHDOG: source %d (%s) silent for %.0fs",
                            i, s["id"], s["seconds_since_frame"])
        return True
    GLib.timeout_add_seconds(interval_sec, tick)
